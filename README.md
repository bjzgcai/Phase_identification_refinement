# Single-Phase XRD Identification

A research codebase for single-phase X-ray diffraction identification, candidate retrieval, and Rietveld-style refinement on experimental RRUFF spectra.

This public layout includes the project source code, a curated `data/Exp_data/` RRUFF experimental subset, `data/match.txt`, and the Stage 1 pretrained weights needed to reproduce the documented workflow. The complete upstream datasets are not mirrored here; third-party data and model artifacts remain subject to their original licenses, database terms, and citation requirements. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [docs/DATA.md](docs/DATA.md).

## Repository Layout

```text
src/single_phase_xrd_identification/
  common/        Shared datasets and PerceiverXRD model code
  stage1/        Theoretical-domain training and verification
  stage2/        Experimental-spectrum candidate retrieval
  refinement/    Candidate CIF refinement and Pearson main-phase selection
scripts/         Utility scripts, including RRUFF candidate generation
docs/            Data and cleanup documentation
examples/        Tiny example assets only
tests/           Smoke/unit tests
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For a pinned reference environment, use `requirements-lock.txt`. The checked reference environment is Python 3.10.19 with PyTorch 2.11.0+cu128/CUDA 12.8. For GPU training, install the PyTorch build matching your CUDA environment from the official PyTorch instructions.

For development smoke tests, install the small test dependency set and run:

```bash
pip install -r requirements-dev.txt
PYTHONPATH=src pytest tests
```

## Security Notes

Only load checkpoints from trusted sources. Public verification scripts use `torch.load(..., weights_only=True)` through a compatibility wrapper when supported by PyTorch. Local training resume checkpoints may include optimizer/scaler state and are treated as trusted local artifacts. The Stage 2 memory bank is a locally generated pickle file; do not accept or load externally supplied pickle files.

## Computational Workflow

The project follows a three-stage identification-to-refinement workflow.

1. **Stage 1: theoretical-pattern training and verification**

   `src/single_phase_xrd_identification/stage1/` trains and verifies the PerceiverXRD model on theoretical/simulated XRD patterns. The pretrained checkpoint is expected at `src/single_phase_xrd_identification/stage1/checkpoints_stage1/model_best.pth`.

   ```bash
   python -m single_phase_xrd_identification.stage1.train
   python -m single_phase_xrd_identification.stage1.verify
   ```

2. **Stage 2: experimental-spectrum candidate retrieval**

   `src/single_phase_xrd_identification/stage2/` applies the Stage 1 model to experimental RRUFF spectra and writes the top candidate MP IDs. The downstream refinement workflow uses the top-10 candidates, especially `temp_rank_0.csv` or the merged top-10 candidate CSV under the Stage 2 analysis output directory.

   ```bash
   python -m single_phase_xrd_identification.stage2.verify \
     --model src/single_phase_xrd_identification/stage1/checkpoints_stage1/model_best.pth \
     --strict_dir data/Exp_data \
     --entries_dict data/entries_dict.json \
     --topk_keep 10
   ```

3. **Refinement: build CIF candidate folders and run Rietveld-style refinement**

   `src/single_phase_xrd_identification/refinement/build_rruff_data.py` converts the Stage 2 top-10 retrieval output into per-sample CIF folders. The documented refinement workflow uses the model-retrieved MP candidates: the rank-1 MP CIF is used as the fixed main phase, and ranks 2-10 are placed in `model_top1_impurities/` as impurity candidates. A reconstructed RRUFF-derived CIF may be retained in the sample folder for provenance/evaluation metadata, but it is not used as a refinement candidate in the model-Top-1 workflow.

   ```bash
   python -m single_phase_xrd_identification.refinement.build_rruff_data \
     --rank-csv src/single_phase_xrd_identification/stage2/analysis_results/temp_rank_0.csv \
     --mp-json data/mp_spacegroup.json \
     --exp-dir data/Exp_data \
     --out-dir data/RRUFF_data \
     --top-k 10

   MAIN_CIF=$(sed -n '1p' data/RRUFF_data/R040009/model_top1_main.txt)
   python -m single_phase_xrd_identification.refinement.xrd_refinement \
     --xy data/Exp_data/R040009.csv \
     --main "$MAIN_CIF" \
     --imp data/RRUFF_data/R040009/model_top1_impurities \
     --wl 1.541838 \
     --main-selection fixed
   ```

## Space-Group and Crystal-System Analysis

Additional analysis scripts report top-1/top-10 accuracy by space group and crystal system for both stages. Missing MP/space-group records are written to separate files and skipped from the accuracy denominators rather than being assigned to P1. Because the script filenames contain a hyphen, run them by file path rather than `python -m`.

Stage 1 analysis reads the Stage 1 top-k CSV, maps labels to MP IDs through `entries_dict.json`, then compares predicted and true space-group / crystal-system metadata from `mp_spacegroup.json`.

```bash
python src/single_phase_xrd_identification/stage1/spacegroup_crystal-system_accuracy_analysis.py \
  --temp-rank-csv src/single_phase_xrd_identification/stage1/analysis_results/temp_rank_0.csv \
  --entries-json data/entries_dict.json \
  --mp-spacegroup-json data/mp_spacegroup.json \
  --output-dir src/single_phase_xrd_identification/stage1/analysis
```

Stage 2 analysis reads the experimental-spectrum top-10 candidate CSV, parses the true RRUFF space group from `data/Exp_data/*_CIF.txt`, and compares it with each predicted MP candidate.

```bash
python src/single_phase_xrd_identification/stage2/spacegroup_crystal-system_accuracy_analysis.py \
  --top10-csv src/single_phase_xrd_identification/stage2/analysis_results/top10_candidates.csv \
  --exp-data-dir data/Exp_data \
  --mp-spacegroup-json data/mp_spacegroup.json \
  --output-dir src/single_phase_xrd_identification/stage2/analysis
```

## Data and Artifacts

This repository intentionally includes the reproducibility artifacts below:

- `data/Exp_data/`: 222 RRUFF experimental spectra plus 222 matching `*_CIF.txt` structure-text files used by the documented Stage 2/refinement workflow.
- `data/match.txt`: strict RRUFFID <-> MPID matching pairs used for evaluation/reference.
- `src/single_phase_xrd_identification/stage1/checkpoints_stage1/model_best.pth`: Stage 1 pretrained `state_dict` checkpoint.

The repository still does not mirror the complete upstream datasets. The following files/directories remain external and must be obtained from their original sources when needed:

- `data/mp_spacegroup.json`
- `data/entries_dict.json`
- `data/MP_data/`
- `src/single_phase_xrd_identification/xqueryer/`

`data/MP_data/` can be large and contains many small files. Keep the original MP_data download link available for users:
[MP_data.zip on Hugging Face](https://huggingface.co/datasets/caobin/PyXplore/resolve/main/MP_data.zip?download=true).

For the broader XQueryer/PyXplore dataset, use the official XQueryer dataset link:
[XQueryer dataset on OneDrive](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL2YvYy81ZDg2MjYyMzg0NzBiNDllL0V1d09VMTNQM2JoSHNiU2lEMTRON3hZQmZCTEdCYTFjX0VhVkhrbGZUajRxZXc%5FZT0xa3liaFg&id=5D8626238470B49E%21s5d530eecddcf47b8b1b4a20f5e0def16&cid=5D8626238470B49E).

Use `docs/ARTIFACTS_SHA256SUMS.txt` to verify redistributed artifacts. See [docs/DATA.md](docs/DATA.md) for the local data layout and artifact policy.

## Attribution and Academic Use

Parts of the data preparation and baseline comparison workflow are based on or compatible with **XQueryer: An Intelligent Crystal Structure Identifier for Powder X-ray Diffraction**. Please consult and cite the official XQueryer resources when using those components or datasets:

- Paper: <https://doi.org/10.1093/nsr/nwaf421>
- Website: <https://xqueryer.caobin.asia/about>
- Benchmarks: <https://github.com/WPEM/XqueryerBench>
- Official dataset: [OneDrive](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL2YvYy81ZDg2MjYyMzg0NzBiNDllL0V1d09VMTNQM2JoSHNiU2lEMTRON3hZQmZCTEdCYTFjX0VhVkhrbGZUajRxZXc%5FZT0xa3liaFg&id=5D8626238470B49E%21s5d530eecddcf47b8b1b4a20f5e0def16&cid=5D8626238470B49E)

The MIT license in this repository applies to project source code. Redistributed RRUFF experimental spectra, Materials Project-derived reference metadata, XQueryer/PyXplore-derived matching information, and pretrained weights remain subject to their original licenses, database terms, and citation requirements. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before redistributing these artifacts.

Note: All data are subject to certain tolerances, as no two crystals are perfectly identical. The documented refinement workflow uses model-retrieved MP structures as inputs; RRUFF-derived structure files are retained only for provenance and evaluation metadata.

## Build RRUFF Candidate CIF Folders

```bash
python -m single_phase_xrd_identification.refinement.build_rruff_data   --rank-csv path/to/temp_rank_0.csv   --mp-json path/to/mp_spacegroup.json   --exp-dir path/to/Exp_data   --out-dir path/to/RRUFF_data   --top-k 10
```

## Run Refinement

```bash
MAIN_CIF=$(sed -n '1p' path/to/RRUFF_data/R040009/model_top1_main.txt)
python -m single_phase_xrd_identification.refinement.xrd_refinement   --xy path/to/R040009.csv   --main "$MAIN_CIF"   --imp path/to/RRUFF_data/R040009/model_top1_impurities   --wl 1.541838   --main-selection fixed
```

The refinement code reads the fixed model Top-1 MP main CIF plus the remaining model-retrieved MP impurity candidates.

## License

See [LICENSE](LICENSE).

The refinement code reads the main CIF plus candidate CIFs and automatically reorders phases by Pearson correlation between the observed pattern and each simulated profile.

## License

See [LICENSE](LICENSE).
