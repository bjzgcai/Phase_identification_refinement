# Data Files

This repository includes a curated RRUFF experimental subset and a pretrained checkpoint for reproducibility. It does not mirror the complete upstream XQueryer/PyXplore, RRUFF, or Materials Project datasets. Download the remaining external data separately from the original sources and place it in the layout below when running Stage 2 candidate retrieval and refinement locally.

## Included Paths

```text
data/
  Exp_data/              # included: 444 RRUFF experimental spectra and *_CIF.txt files
  MP_data/               # external: MP-derived simulated/reference pattern files
  mp_spacegroup.json     # external: MP metadata and CIF strings, not redistributed
  entries_dict.json      # external: label -> mp-id mapping, not redistributed
  match.txt              # included: strict RRUFFID <-> MPID matching pairs
src/single_phase_xrd_identification/stage1/checkpoints_stage1/
  model_best.pth         # included: Stage 1 pretrained state_dict checkpoint
```

## Download Sources

- XQueryer/PyXplore official dataset: [OneDrive](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL2YvYy81ZDg2MjYyMzg0NzBiNDllL0V1d09VMTNQM2JoSHNiU2lEMTRON3hZQmZCTEdCYTFjX0VhVkhrbGZUajRxZXc%5FZT0xa3liaFg&id=5D8626238470B49E%21s5d530eecddcf47b8b1b4a20f5e0def16&cid=5D8626238470B49E)
- MP_data fallback archive: [MP_data.zip on Hugging Face](https://huggingface.co/datasets/caobin/PyXplore/resolve/main/MP_data.zip?download=true)

The two JSON files under `data/` are intentionally ignored by Git because they contain third-party metadata/CIF content with redistribution constraints.

## Data Availability and License

`data/Exp_data/` contains RRUFF experimental spectra and matching structure files derived from external datasets. These data are redistributed as a curated reproducibility subset and remain subject to the original RRUFF/XQueryer data licenses, database terms, and citation requirements.

Note: All data are subject to certain tolerances, as no two crystals are perfectly identical. The reference structure derived from **XQueryer** serves as the input for further structure determination during the refinement step.

The repository MIT license covers project source code only. See `../THIRD_PARTY_NOTICES.md` for source attribution, citation links, artifact hashes, and redistribution notes.

## Artifact Verification

Redistributed data and model artifacts are listed in `docs/ARTIFACTS_SHA256SUMS.txt`. Verify them with:

```bash
sha256sum -c docs/ARTIFACTS_SHA256SUMS.txt
```

The key single-file artifacts are:

```text
b115fb536826790aaaabaa844034e310ee95ed4d31761fbcd67391519ddd9680  src/single_phase_xrd_identification/stage1/checkpoints_stage1/model_best.pth
e190ae856a11a2bcd5d5a999f8984bed05e744307b4e49ee690032dd3e6fb52a  data/match.txt
```

## Generated Candidate CIFs

`RRUFF_data/` is generated and remains ignored by Git. Regenerate it with:

```bash
python scripts/build_rruff_data.py   --rank-csv stage2/analysis_results/temp_rank_0.csv   --mp-json data/mp_spacegroup.json   --exp-dir data/Exp_data   --out-dir data/RRUFF_data
```

If you want `temp_rank_0.csv` included in a release, place it under a tracked data or release-artifact location and update the command accordingly.

## Checkpoint Distribution

`model_best.pth` is intentionally included for reproducibility. It is a Stage 1 `state_dict` checkpoint and should be loaded only from trusted repository/release sources. If a hosting service rejects large Git objects, publish the same file through Git LFS, a release asset, Hugging Face, Zenodo, or an institutional archive and keep the SHA256 hash above unchanged.

## Still Excluded

- Generated refinement result folders
- Runtime logs and `.done` markers
- Newly generated checkpoints such as `checkpoint_*.pth`
- `data/mp_spacegroup.json` and `data/entries_dict.json`
- `data/MP_data/`
- `src/single_phase_xrd_identification/xqueryer/`
- Generated `RRUFF_data/` candidate folders unless you explicitly decide to publish them separately
