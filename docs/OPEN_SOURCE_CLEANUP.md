# Open Source Cleanup Plan

This repository currently mixes source code, experimental data, generated candidates, model weights, checkpoints, and refinement outputs. Before publishing, keep the public repository focused on reproducible code and small examples, and move large artifacts to a release asset, Zenodo/OSF, Hugging Face, or an institutional data archive.

## Recommended Public Layout

```text
single_phase_XRD_identification/
  README.md
  LICENSE
  pyproject.toml or requirements.txt
  .gitignore
  src/xrd_identification/
    common/
    stage1/
    stage2/
    refinement/
    xqueryer/
  scripts/
    build_rruff_data.py
    run_stage1_train.py
    run_stage2_verify.py
    run_refinement.py
  configs/
    stage1.yaml
    stage2.yaml
    refinement.yaml
  examples/
    mini_exp_data/
    mini_rruff_data/
  docs/
    OPEN_SOURCE_CLEANUP.md
    DATA.md
    REFINEMENT.md
  tests/
```

## What Should Stay In Git

- Core Python source files: model, dataset loaders, training, verification, refinement.
- Small toy/example data only.
- Config files and command examples.
- Documentation explaining how to obtain full data and pretrained weights.
- Scripts that regenerate candidate CIF folders from downloaded data.

## What Should Not Stay In Git

- `data/mp_spacegroup.json` because it is about 500 MB.
- `Exp_data_refinement/RRUFF_data/` because it is generated candidate CIF data.
- `Exp_data_refinement/result_RL_rank/` because it is generated refinement output.
- `*.pth`, `*.pt`, `*.ckpt`, and training checkpoints.
- Large CSV analysis reports under `analysis_results/`.
- `__pycache__`, logs, `.done`, images, and intermediate `.xy` outputs.

## Suggested Migration Map

| Current path | Public destination | Notes |
| --- | --- | --- |
| `common/` | `src/xrd_identification/common/` | Shared datasets and model code. |
| `stage1/` | `src/xrd_identification/stage1/` | Keep source; move outputs/checkpoints out. |
| `stage2/` | `src/xrd_identification/stage2/` | Keep source; publish small example ranking CSV only. |
| `Base_Xqueryer/` | `src/xrd_identification/xqueryer/` | Rename for style consistency. |
| `Exp_data_refinement/XRDRefinement.py` | `src/xrd_identification/refinement/xrd_refinement.py` | Rename to snake_case later. |
| `Exp_data_refinement/build_rruff_data.py` | `scripts/build_rruff_data.py` | Utility script. |
| `Ablation/` | `experiments/ablation/` or `docs/ablation.md` | Keep only reproducible configs/scripts. |
| `data/Exp_data/` | external data archive | Provide download instructions. |
| `train_space/` | external training workspace or `experiments/` | Do not publish checkpoints. |

## Low-Risk Cleanup Steps

1. Keep a private full workspace exactly as it is.
2. Create a clean public branch or copy.
3. Apply `.gitignore` before adding files.
4. Move generated data/results outside the repo or leave them untracked.
5. Add `requirements.txt` or `environment.yml`.
6. Convert `Readme.txt` to `README.md` and replace internal notes with public run commands.
7. Add a tiny example dataset so users can run a smoke test without downloading hundreds of MB.
8. Add tests for CIF parsing, dataset loading, and Pearson main-phase selection.

## Immediate Checks Before Publishing

- `find . -size +50M` should only show files intentionally tracked with Git LFS or excluded by `.gitignore`.
- `find . -name '__pycache__' -o -name '*.pyc'` should be empty in the public copy.
- No absolute cluster paths should appear in scripts or docs.
- No generated result folder should be committed.
- Full data and pretrained weights should have clear download links and checksums.
