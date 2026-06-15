# Single-Phase XRD Identification

A research codebase for single-phase X-ray diffraction identification, candidate retrieval, and Rietveld-style refinement on experimental RRUFF spectra.

This public layout includes the project source code and Stage 1 pretrained weights. Large or third-party data files are not redistributed in Git; download them from their original sources as described below.

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

For GPU training, install the PyTorch build matching your CUDA environment from the official PyTorch instructions.

## Data

The full experimental/reference data used by this project comes from external datasets and should be downloaded separately. In particular, the public repository does not redistribute:

- `data/MP_data/`
- `src/single_phase_xrd_identification/xqueryer/`

`data/MP_data/` can be large and contains many small files. Keep the original MP_data download link available for users:
[MP_data.zip on Hugging Face](https://huggingface.co/datasets/caobin/PyXplore/resolve/main/MP_data.zip?download=true).

For the broader XQueryer/PyXplore dataset, use the official XQueryer dataset link:
[XQueryer dataset on OneDrive](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL2YvYy81ZDg2MjYyMzg0NzBiNDllL0V1d09VMTNQM2JoSHNiU2lEMTRON3hZQmZCTEdCYTFjX0VhVkhrbGZUajRxZXc%5FZT0xa3liaFg&id=5D8626238470B49E%21s5d530eecddcf47b8b1b4a20f5e0def16&cid=5D8626238470B49E).

See [docs/DATA.md](docs/DATA.md) for the expected local data layout after download.

## Attribution and Academic Use

Parts of the data preparation and baseline comparison workflow are based on or compatible with **XQueryer: An Intelligent Crystal Structure Identifier for Powder X-ray Diffraction**. Please consult and cite the official XQueryer resources when using those components or datasets:

- Paper: <https://doi.org/10.1093/nsr/nwaf421>
- Website: <https://xqueryer.caobin.asia/about>
- Benchmarks: <https://github.com/WPEM/XqueryerBench>
- Official dataset: [OneDrive](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL2YvYy81ZDg2MjYyMzg0NzBiNDllL0V1d09VMTNQM2JoSHNiU2lEMTRON3hZQmZCTEdCYTFjX0VhVkhrbGZUajRxZXc%5FZT0xa3liaFg&id=5D8626238470B49E%21s5d530eecddcf47b8b1b4a20f5e0def16&cid=5D8626238470B49E)

RRUFF experimental spectra and Materials Project-derived reference data remain subject to their original licenses, database terms, and citation requirements. This repository only provides code and documentation needed to reproduce the workflow with locally obtained data.

## Build RRUFF Candidate CIF Folders

```bash
python scripts/build_rruff_data.py   --rank-csv path/to/temp_rank_0.csv   --mp-json path/to/mp_spacegroup.json   --exp-dir path/to/Exp_data   --out-dir path/to/RRUFF_data
```

## Run Refinement

```bash
python -m single_phase_xrd_identification.refinement.xrd_refinement   --xy path/to/R040009.csv   --main path/to/RRUFF_data/R040009/R040009.cif   --imp path/to/RRUFF_data/R040009   --wl 1.541838
```

The refinement code reads the main CIF plus candidate CIFs and automatically reorders phases by Pearson correlation between the observed pattern and each simulated profile.

## License

See [LICENSE](LICENSE).
