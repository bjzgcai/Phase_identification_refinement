# stage1/train.py

import os
import sys
import json
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import ConcatDataset, Subset
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from ase.db import connect

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", ".."))

from single_phase_xrd_identification.common.dataset import XRDDataset
from single_phase_xrd_identification.common.model import PerceiverXRD
from single_phase_xrd_identification.common.serialization import safe_torch_load

# -----------------------------
# 配置（你按需改这几行）
# -----------------------------
DB_TRAIN = os.path.join(ROOT_DIR, "data", "trainD.db")
DB_VAL   = os.path.join(ROOT_DIR, "data", "valueV.db")

NUM_CLASSES = 100315   # ✅ Task A：结构ID分类 (0..100314)
BATCH_SIZE  = 64
NUM_WORKERS = 4

EPOCHS = 100
WARMUP_EPOCHS = 20

LR = 8e-5              #  默认 8e-5
WEIGHT_DECAY = 1e-4
MIN_LR = 1e-5

LOG_FILE = "log_stage1.json"
CKPT_DIR = "checkpoints_stage1"
FINAL_MODEL = "model_stage1_final.pth"


# -----------------------------
# Loss：Focal Loss（多分类 logits）
# -----------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# -----------------------------
# ✅ iteration 平滑版 LR：warmup + cosine
# -----------------------------
def adjust_lr_warmup_cosine_iter(
    optimizer,
    epoch_float: float,
    *,
    base_lr: float,
    epochs: int,
    warmup_epochs: int,
    min_lr: float = 0.0,
) -> float:
    if epoch_float < warmup_epochs:
        lr = base_lr * epoch_float / max(1e-9, warmup_epochs)
    else:
        lr = base_lr * 0.5 * (1.0 + math.cos(
            math.pi * (epoch_float - warmup_epochs) / max(1e-9, (epochs - warmup_epochs))
        ))

    if min_lr > 0:
        lr = max(lr, min_lr)

    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# -----------------------------
# DDP init / cleanup
# -----------------------------
def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    else:
        # 单卡调试兼容（注意：需要有 CUDA）
        print("⚠️ 未检测到 torchrun 环境，自动进入单卡模式（rank=0, world_size=1）")
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        os.environ["LOCAL_RANK"] = "0"
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(0)


def cleanup_ddp():
    dist.destroy_process_group()


def count_rows(db_path: str) -> int:
    with connect(db_path) as db:
        return db.count()


# -----------------------------
# Train / Eval
# -----------------------------
def run_one_epoch_train(
    model, loader, device, criterion, optimizer, scaler,
    *, epoch: int, epochs: int, warmup_epochs: int, base_lr: float, min_lr: float,
    is_master: bool, desc: str
):
    model.train()
    correct = 0
    total = 0
    loss_sum = 0.0

    iters = len(loader)
    iterator = tqdm(loader, desc=desc, unit="batch") if is_master else loader

    for i, batch in enumerate(iterator):
        epoch_float = (epoch - 1) + (i + 1) / max(1, iters)
        lr_now = adjust_lr_warmup_cosine_iter(
            optimizer,
            epoch_float,
            base_lr=base_lr,
            epochs=epochs,
            warmup_epochs=warmup_epochs,
            min_lr=min_lr,
        )

        x, peaks, elem, y = batch
        x = x.to(device, non_blocking=True)
        peaks = peaks.to(device, non_blocking=True)
        elem = elem.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()

        y_min = int(y.min().item())
        y_max = int(y.max().item())
        if not (0 <= y_min and y_max < NUM_CLASSES):
            raise RuntimeError(f"发现非法 label: min={y_min}, max={y_max}，期望 [0,{NUM_CLASSES-1}]")

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            out = model(x, peaks, elem)
            if out.size(1) != NUM_CLASSES:
                raise RuntimeError(f"模型输出 out.size(1)={out.size(1)} != NUM_CLASSES={NUM_CLASSES}")
            loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_sum += float(loss.item())
        pred = out.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.size(0))

        if is_master:
            acc = 100.0 * correct / max(1, total)
            iterator.set_postfix({
                "lr": f"{lr_now:.2e}",
                "loss": f"{float(loss.item()):.3f}",
                "acc": f"{acc:.4f}%"
            })

    avg_loss = loss_sum / max(1, len(loader))
    acc = 100.0 * correct / max(1, total)
    return avg_loss, acc


@torch.no_grad()
def run_one_epoch_eval(model, loader, device, criterion, *, is_master: bool, desc: str):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0

    iterator = tqdm(loader, desc=desc, unit="batch") if is_master else loader
    for batch in iterator:
        x, peaks, elem, y = batch
        x = x.to(device, non_blocking=True)
        peaks = peaks.to(device, non_blocking=True)
        elem = elem.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()

        with torch.amp.autocast("cuda"):
            out = model(x, peaks, elem)
            loss = criterion(out, y)

        loss_sum += float(loss.item())
        pred = out.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.size(0))

        if is_master:
            acc = 100.0 * correct / max(1, total)
            iterator.set_postfix({
                "loss": f"{float(loss.item()):.3f}",
                "acc": f"{acc:.4f}%"
            })

    avg_loss = loss_sum / max(1, len(loader))
    acc = 100.0 * correct / max(1, total)
    return avg_loss, acc

@torch.no_grad()
def run_one_epoch_eval_ddp(model, loader, device, criterion, *, is_master: bool, desc: str):
    """
    DDP 验证：每个 rank 跑自己的 val_loader 分片，
    然后 all_reduce 汇总成全量 val 的 avg_loss / acc
    """
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0

    iterator = tqdm(loader, desc=desc, unit="batch") if is_master else loader
    for batch in iterator:
        x, peaks, elem, y = batch
        x = x.to(device, non_blocking=True)
        peaks = peaks.to(device, non_blocking=True)
        elem = elem.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()

        with torch.amp.autocast("cuda"):
            out = model(x, peaks, elem)
            loss = criterion(out, y)

        bs = int(y.size(0))
        loss_sum += float(loss.item()) * bs
        pred = out.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += bs
        
        if is_master:
            acc_local = 100.0 * correct / max(1, total)
            avg_loss_local = loss_sum / max(1, total)
            iterator.set_postfix({
                "loss": f"{avg_loss_local:.3f}",
                "acc": f"{acc_local:.4f}%"
            })

    t = torch.tensor([loss_sum, correct, total], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)

    loss_sum_all, correct_all, total_all = t.tolist()
    avg_loss = loss_sum_all / max(1.0, total_all)
    acc = 100.0 * correct_all / max(1.0, total_all)
    return avg_loss, acc

def save_json(history: dict):
    with open(LOG_FILE, "w") as f:
        json.dump(history, f, indent=2)


# -----------------------------
# Main
# -----------------------------
def main():
    setup_ddp()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    is_master = (rank == 0)

    if is_master:
        print("\n✅ Stage1 (Task A) = 结构ID分类")
        print(f"   Train = {os.path.basename(DB_TRAIN)} | Val = {os.path.basename(DB_VAL)}")
        print(f"   world_size = {world_size} | batch = {BATCH_SIZE} x {world_size} = {BATCH_SIZE*world_size}")
        print(f"   num_classes = {NUM_CLASSES}")
        print(f"   base_lr = {LR} | min_lr = {MIN_LR} | warmup_epochs = {WARMUP_EPOCHS} | epochs = {EPOCHS}")
        print("   ✅ 方案启用：Train Val 都用 DDP")


    # 1) 模型 + DDP
    base_model = PerceiverXRD(num_classes=NUM_CLASSES).to(device)
    model = DDP(
        base_model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
        broadcast_buffers=False,   # ✅ 避免下一轮 forward 的小广播卡死
    )


    # 2) 数据：
    n_train = count_rows(DB_TRAIN)
    n_val   = count_rows(DB_VAL)
    train_ids = list(range(1, n_train + 1))
    val_ids   = list(range(1, n_val + 1))

    trainset = XRDDataset(
        DB_TRAIN, train_ids,
        return_id_str=False, return_elem_onehot=True, num_classes=NUM_CLASSES
    )
    valset = XRDDataset(
        DB_VAL, val_ids,
        return_id_str=False, return_elem_onehot=True, num_classes=NUM_CLASSES
    )

    # --- Train: DDP sampler ---
    train_sampler = DistributedSampler(trainset, shuffle=True)
    train_loader = DataLoader(
        trainset, batch_size=BATCH_SIZE, sampler=train_sampler,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
    )
    

    # --- Val: DDP sampler (all ranks) ---
    val_sampler = DistributedSampler(valset, shuffle=False, drop_last=False)
    val_loader = DataLoader(
        valset, batch_size=BATCH_SIZE, sampler=val_sampler,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False
    )

    # 3) 优化器 / loss / AMP
    criterion = FocalLoss(gamma=2.0, alpha=1.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda")

    os.makedirs(CKPT_DIR, exist_ok=True)

    # --- 断点续训逻辑 ---
    start_epoch = 0
    best_val_loss = float("inf")
    history = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    ckpt_path = os.path.join(CKPT_DIR, "checkpoint_0060.pth")
    if os.path.exists(ckpt_path):
        checkpoint = safe_torch_load(ckpt_path, map_location=device, trusted=True)

        model.module.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])

        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device, non_blocking=True)

        start_epoch = int(checkpoint.get("epoch", 0))

        if is_master and os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f:
                history = json.load(f)
            if history.get("val_loss", []):
                best_val_loss = min(history["val_loss"])

        dist.barrier()
        if is_master:
            print(f"🚀 恢复成功！从 Epoch {start_epoch + 1} 接力，当前最佳 val_loss: {best_val_loss:.4f}")

    # 4) 训练：每个 epoch 都 TrainD -> ValV（DDP：Train/Val 都分片跑 + all_reduce 汇总）
    for epoch in range(start_epoch + 1, EPOCHS + 1):
        train_sampler.set_epoch(epoch)

        if is_master:
            print(f"\n---------------- Epoch {epoch}/{EPOCHS} | lr(now)={optimizer.param_groups[0]['lr']:.2e} ----------------")

        # ---- TrainD (DDP, all ranks) ----
        train_loss, train_acc = run_one_epoch_train(
            model, train_loader, device, criterion, optimizer, scaler,
            epoch=epoch, epochs=EPOCHS, warmup_epochs=WARMUP_EPOCHS,
            base_lr=LR, min_lr=MIN_LR,
            is_master=is_master,
            desc=f"[TrainD] Epoch {epoch}/{EPOCHS}",
        )

        # ---- ValV (DDP, all ranks) ----
        val_sampler.set_epoch(epoch)  # 可写可不写，shuffle=False 时也安全
        val_loss, val_acc = run_one_epoch_eval_ddp(
            model, val_loader, device, criterion,
            is_master=is_master,
            desc=f"[valueV-DDP] Epoch {epoch}/{EPOCHS}",
        )

        # 只有 master 记录日志、保存模型
        if is_master:
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            save_json(history)

            print(f"✅ train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | train_acc={train_acc:.4f}% | val_acc={val_acc:.4f}%")

            # checkpoint
            if epoch == 1 or epoch % 5 == 0:
                ckpt_path = os.path.join(CKPT_DIR, f"checkpoint_{epoch:04d}.pth")
                torch.save({
                    "epoch": epoch,
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                }, ckpt_path)
                print(f"💾 checkpoint saved: {ckpt_path}")

            # best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(CKPT_DIR, "model_best.pth")
                torch.save(model.module.state_dict(), best_path)
                print(f"🏆 best model updated (val_loss={best_val_loss:.4f}): {best_path}")

        dist.barrier()

    if is_master:
        torch.save(model.module.state_dict(), FINAL_MODEL)
        print(f"\n🎉 训练完成：最终模型已保存 -> {FINAL_MODEL}")

    cleanup_ddp()


if __name__ == "__main__":
    main()