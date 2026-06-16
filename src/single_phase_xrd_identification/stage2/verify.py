# verify_stage3.py
#
# 运行示例：
#   torchrun --nproc_per_node=1 verify_stage3.py 
#
import os, sys, json, re, csv, argparse
from typing import Dict, List, Set, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "..", ".."))

from single_phase_xrd_identification.common.dataset_real import XRDDatasetStrict
from single_phase_xrd_identification.common.model import PerceiverXRD
from single_phase_xrd_identification.common.serialization import safe_torch_load

# ---------------- DDP ----------------
def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def is_ddp():
    return dist.is_available() and dist.is_initialized()

def get_rank_world():
    if is_ddp():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1

def get_device():
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")

# ---------------- entries_dict：label -> mpid ----------------
def load_entries(entries_dict_path: str) -> Dict[int, str]:
    with open(entries_dict_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    out: Dict[int, str] = {}
    for k, v in obj.items():
        try:
            lab = int(k)
        except Exception:
            continue
        if isinstance(v, dict) and "value" in v:
            mp = str(v["value"]).strip()
        else:
            mp = str(v).strip()
        if mp.endswith(".cif"):
            mp = mp[:-4]
        out[lab] = mp
    return out

# ---------------- label_formula.csv：label -> formula ----------------
def load_label2formula_from_csv(csv_path: str) -> Dict[int, str]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到 label_formula.csv：{csv_path}")

    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

    label_keys = {"label", "classid", "class", "idx", "id", "y"}
    formula_keys = {"formula", "chemicalformulasum", "chemicalformula", "composition", "formulasum"}

    label2formula: Dict[int, str] = {}

    with open(csv_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        first_line = sample.splitlines()[0] if sample.splitlines() else ""
        has_header = False
        if first_line and (("label" in first_line.lower()) or ("formula" in first_line.lower())):
            has_header = True

        if has_header:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                has_header = False
            else:
                fields = [norm(x) for x in reader.fieldnames]
                label_col = None
                formula_col = None
                for raw_name, nname in zip(reader.fieldnames, fields):
                    if nname in label_keys and label_col is None:
                        label_col = raw_name
                    if nname in formula_keys and formula_col is None:
                        formula_col = raw_name

                for row in reader:
                    try:
                        if label_col is None:
                            k0 = reader.fieldnames[0]
                            lab = int(str(row.get(k0, "")).strip())
                        else:
                            lab = int(str(row.get(label_col, "")).strip())
                    except Exception:
                        continue

                    if formula_col is None:
                        if len(reader.fieldnames) < 2:
                            continue
                        k1 = reader.fieldnames[1]
                        formula = str(row.get(k1, "")).strip()
                    else:
                        formula = str(row.get(formula_col, "")).strip()

                    if formula:
                        label2formula[lab] = formula

        if not has_header:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 2:
                    continue
                try:
                    lab = int(str(row[0]).strip())
                except Exception:
                    continue
                formula = str(row[1]).strip()
                if formula:
                    label2formula[lab] = formula

    if len(label2formula) == 0:
        raise RuntimeError(f"label_formula.csv 未解析到任何有效映射：{csv_path}")
    return label2formula

# ---------------- 元素解析：从 formula 抽元素集合 ----------------
_ELEM_RE = re.compile(r"([A-Z][a-z]?)")
def formula_to_elems(formula: str) -> Set[str]:
    if not isinstance(formula, str):
        return set()
    formula = formula.strip()
    if not formula:
        return set()
    return set(_ELEM_RE.findall(formula))

# --------- 从 *_CIF.txt 解析实验谱元素集合（优先 ATOM 表，失败兜底 token）---------
_ELEM_TOKEN_RE = re.compile(r"\b([A-Z][a-z]?)\b")
def parse_elements_from_rruff_cif_txt(txt_path: str) -> Set[str]:
    elems: Set[str] = set()
    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return elems

    start = -1
    for i, line in enumerate(lines):
        if "ATOM" in line and "OCCUPANCY" in line:
            start = i + 1
            break

    if start != -1:
        for j in range(start, min(start + 500, len(lines))):
            s = lines[j].strip()
            if not s:
                break
            tok = s.split()
            if not tok:
                continue
            sym = tok[0]
            if re.match(r"^[A-Z][a-z]?$", sym):
                elems.add(sym)
            if "2-THETA" in s or "X-RAY" in s:
                break

    if not elems:
        for line in lines:
            for sym in _ELEM_TOKEN_RE.findall(line):
                if re.match(r"^[A-Z][a-z]?$", sym):
                    elems.add(sym)

    return elems

# ---------------- 模型加载 ----------------
def load_model(model_path: str, device: torch.device, num_classes: int) -> PerceiverXRD:
    model = PerceiverXRD(num_classes=num_classes).to(device)
    if device.type == "cuda":
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        ckpt = safe_torch_load(model_path, map_location={"cuda:0": f"cuda:{local_rank}"})
    else:
        ckpt = safe_torch_load(model_path, map_location=device)

    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./model_best.pth", help="model_best.pth / checkpoint")
    parser.add_argument("--strict_dir", default="../data/Exp_data", help="strict directory containing *.csv and *_CIF.txt")
    parser.add_argument("--entries_dict", default="../data/entries_dict.json", help="entries_dict.json")
    parser.add_argument("--label_formula_csv", default="./label_formula.csv", help="label_formula.csv")
    parser.add_argument("--num_classes", type=int, default=100315)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--raw_topn", type=int, default=50000)
    parser.add_argument("--topk_keep", type=int, default=10)
    parser.add_argument("--out_dir", default=os.path.join(THIS_DIR, "analysis_results"))
    parser.add_argument("--use_elem_cond", type=int, default=1, choices=[0,1],
                        help="1: elem_onehot参与推理(FiLM ON); 0: elem_onehot全零(FiLM OFF)")
    
    # 默认路径假设你在 stage3 目录直接运行：
    #   ./model_best.pth
    #   ./label_formula.csv
    #   ../data/entries_dict.json
    #   ../data/MP-match/strict
    # 若你的目录不同，用命令行参数覆盖即可。

    args = parser.parse_args()

    setup_ddp()
    rank, world_size = get_rank_world()
    device = get_device()
    is_master = (rank == 0)
    os.makedirs(args.out_dir, exist_ok=True)

    temp_csv_pattern = os.path.join(args.out_dir, "temp_rank_{rank}.csv")
    top10_csv = os.path.join(args.out_dir, "top10_candidates.csv")
    summary_json = os.path.join(args.out_dir, "summary.json")

    if is_master:
        print("🔍 [Stage3.1+ | Logits TopN | NO elem filter in retrieval | Eval by elemset EXACT match]")
        print(f"   use_elem_cond={args.use_elem_cond} (FiLM/elem_onehot {'ON' if args.use_elem_cond else 'OFF'})")
        print(f"   strict_dir={args.strict_dir}")
        print(f"   entries_dict={args.entries_dict}")
        print(f"   label_formula_csv={args.label_formula_csv}")
        print(f"   model={args.model}")
        print(f"   world_size={world_size} | batch={args.batch_size} | workers={args.num_workers} | device={device}")
        print(f"   RAW_TOPN={args.raw_topn} | TOPK_KEEP={args.topk_keep}")

    # model
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"找不到模型权重：{args.model}")
    model = load_model(args.model, device, args.num_classes)
    if is_master:
        print("✅ 模型加载成功")

    # mappings
    label2mpid = load_entries(args.entries_dict)
    label2formula = load_label2formula_from_csv(args.label_formula_csv)
    if is_master:
        print(f"✅ label->formula loaded: n={len(label2formula)}")

    # dataset
    if not os.path.isdir(args.strict_dir):
        raise FileNotFoundError(f"找不到 strict 目录：{args.strict_dir}")

    rruff2mp_dict = {}
    for fn in os.listdir(args.strict_dir):
        if fn.endswith(".csv"):
            rid = fn[:-4]
            rruff2mp_dict[rid] = 0  # dummy label

    ds = XRDDatasetStrict(
        args.strict_dir,
        rruff2mp_dict,
        num_classes=args.num_classes,
    )

    sampler = DistributedSampler(ds, shuffle=False, drop_last=False) if is_ddp() else None
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False if sampler is not None else False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # per-rank write candidates
    temp_csv = temp_csv_pattern.format(rank=rank)
    fcsv = open(temp_csv, "w", newline="", encoding="utf-8")
    writer = csv.writer(fcsv)
    writer.writerow(["rruff_id", "rank", "pred_label", "pred_mpid", "pred_formula", "pred_elems", "exp_elems"])

    iterator = tqdm(loader, desc=f"Rank {rank} verifying", unit="batch") if is_master else loader
    use_amp = (device.type == "cuda")
    zeros_elem = None

    for batch in iterator:
        # dataset_real(新) 返回: x, peaks, elem_onehot, label, name(rruff_id)
        if len(batch) == 5:
            x, peaks, elem_onehot, _y, rruff_id = batch

        # 兼容 4 字段 dataset（可留可不留）
        elif len(batch) == 4:
            a, b, c, d = batch
            x, peaks = a, b
            elem_onehot = None
            _y = None
            rruff_id = d

            if torch.is_tensor(c) and (c.numel() == 118 or (c.dim() == 2 and c.size(-1) == 118)):
                elem_onehot = c
            else:
                _y = c
        else:
            raise RuntimeError(f"XRDDatasetStrict 返回字段数异常: {len(batch)}")


        x = x.to(device, non_blocking=True)
        peaks = peaks.to(device, non_blocking=True)
        bs = x.size(0)

        if args.use_elem_cond == 1 and elem_onehot is not None:
            elem_onehot = elem_onehot.to(device, non_blocking=True)
        else:
            # FiLM OFF / 或 dataset 不提供 elem_onehot：用全零
            if zeros_elem is None or zeros_elem.size(0) != bs:
                zeros_elem = torch.zeros((bs, 118), dtype=torch.float32, device=device)
            elem_onehot = zeros_elem

        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(x, peaks, elem_onehot)
        else:
            logits = model(x, peaks, elem_onehot)

        k_search = min(args.raw_topn, logits.size(1))
        _, topj = torch.topk(logits, k=k_search, dim=1)

        for i in range(bs):
            rid = str(rruff_id[i])
            exp_txt = os.path.join(args.strict_dir, f"{rid}_CIF.txt")
            exp_elems = set(parse_elements_from_rruff_cif_txt(exp_txt))
            exp_elems_str = " ".join(sorted(exp_elems))

            # ✅ 不做元素过滤，只做 mpid 去重
            seen_mpid: Set[str] = set()
            kept: List[Tuple[int, str, str, Set[str]]] = []

            for t in range(k_search):
                lab = int(topj[i, t].item())
                formula = label2formula.get(lab, "")
                if not formula:
                    continue
                mpid = label2mpid.get(lab, "")
                if not mpid:
                    continue
                if mpid in seen_mpid:
                    continue
                seen_mpid.add(mpid)

                pred_elems = formula_to_elems(formula)
                kept.append((lab, mpid, formula, pred_elems))
                if len(kept) >= args.topk_keep:
                    break

            for rnk, (lab, mpid, formula, pred_elems) in enumerate(kept, start=1):
                pred_elems_str = " ".join(sorted(pred_elems))
                writer.writerow([rid, rnk, lab, mpid, formula, pred_elems_str, exp_elems_str])

    fcsv.close()
    if is_ddp():
        dist.barrier()

    if is_master:
        # merge
        with open(top10_csv, "w", newline="", encoding="utf-8") as fout:
            wout = csv.writer(fout)
            wout.writerow(["rruff_id", "rank", "pred_label", "pred_mpid", "pred_formula", "pred_elems", "exp_elems"])
            for r in range(world_size):
                p = temp_csv_pattern.format(rank=r)
                if not os.path.exists(p):
                    continue
                with open(p, "r", encoding="utf-8") as fin:
                    reader = csv.reader(fin)
                    _ = next(reader, None)
                    for row in reader:
                        wout.writerow(row)
        print(f"✅ 输出候选: {top10_csv}")

        # compute elemset accuracy
        rid2rows: Dict[str, List[Tuple[Set[str], Set[str]]]] = {}
        with open(top10_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid = row["rruff_id"].strip()
                rnk = int(row["rank"])
                pred_elems = set(row["pred_elems"].split()) if row["pred_elems"].strip() else set()
                exp_elems = set(row["exp_elems"].split()) if row["exp_elems"].strip() else set()
                if rid not in rid2rows:
                    rid2rows[rid] = []
                while len(rid2rows[rid]) < rnk - 1:
                    rid2rows[rid].append((set(), set()))
                if len(rid2rows[rid]) == rnk - 1:
                    rid2rows[rid].append((pred_elems, exp_elems))

        strict_rids = sorted(rruff2mp_dict.keys())
        total = 0
        missing_exp_elems = 0
        no_candidate = 0
        top1_hit = 0
        top5_hit = 0
        top10_hit = 0

        for rid in strict_rids:
            exp_txt = os.path.join(args.strict_dir, f"{rid}_CIF.txt")
            exp_elems = set(parse_elements_from_rruff_cif_txt(exp_txt))
            if not exp_elems:
                missing_exp_elems += 1
                continue

            total += 1
            preds = rid2rows.get(rid, [])
            preds = [(pe, ee) for (pe, ee) in preds if pe]
            if len(preds) == 0:
                no_candidate += 1
                continue

            def hit_any(k: int) -> bool:
                for pe, _ in preds[:k]:
                    if pe == exp_elems:
                        return True
                return False

            if preds[0][0] == exp_elems:
                top1_hit += 1
            if hit_any(5):
                top5_hit += 1
            if hit_any(10):
                top10_hit += 1

        top1_acc = top1_hit / max(1, total)
        top5_acc = top5_hit / max(1, total)
        top10_acc = top10_hit / max(1, total)

        summary = {
            "mode": "stage3_1_plus_logits_topN_no_elem_filter_eval_by_elemset_exact",
            "use_elem_cond": int(args.use_elem_cond),
            "strict_dir": args.strict_dir,
            "model": args.model,
            "entries_dict": args.entries_dict,
            "label_formula_csv": args.label_formula_csv,
            "raw_topn": int(args.raw_topn),
            "topk_keep": int(args.topk_keep),
            "total_strict_samples": len(strict_rids),
            "total_used_in_eval": total,
            "missing_exp_elems_count": missing_exp_elems,
            "no_candidate_count": no_candidate,
            "top1_elemset_acc": top1_acc,
            "top5_elemset_acc": top5_acc,
            "top10_elemset_acc": top10_acc,
            "outputs": {"top10_candidates_csv": top10_csv, "summary_json": summary_json},
        }
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print("\n📊 [Stage3.1+ 元素集合命中率（FiLM/elem_onehot 参与推理；不做元素过滤检索）]")
        print(f"   total_used_in_eval = {total} (missing_exp_elems={missing_exp_elems})")
        print(f"   no_candidate       = {no_candidate}")
        print(f"   Top-1 Elem Acc     = {top1_acc*100:.2f}%")
        print(f"   Top-5 Elem Acc     = {top5_acc*100:.2f}%")
        print(f"   Top-10 Elem Acc    = {top10_acc*100:.2f}%")
        print(f"✅ Summary: {summary_json}")

    cleanup_ddp()


if __name__ == "__main__":
    main()
