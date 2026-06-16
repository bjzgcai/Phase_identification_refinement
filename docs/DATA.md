# Data Files

This public repository does not redistribute the full external datasets. Download the data separately from the original sources and place it in the layout below when running Stage 2 candidate retrieval and refinement locally.

## Included Paths

```text
data/
  Exp_data/              # RRUFF experimental spectra and *_CIF.txt files, external
  MP_data/               # MP-derived simulated/reference pattern files, external
  mp_spacegroup.json     # MP metadata and CIF strings, external and not redistributed
  entries_dict.json      # label -> mp-id mapping, external and not redistributed
  match.txt              # strict RRUFFID <-> MPID matching pairs, if permitted
src/single_phase_xrd_identification/stage1/checkpoints_stage1/
  model_best.pth         # Stage 1 pretrained checkpoint
```

## Download Sources

- XQueryer/PyXplore official dataset: [OneDrive](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL2YvYy81ZDg2MjYyMzg0NzBiNDllL0V1d09VMTNQM2JoSHNiU2lEMTRON3hZQmZCTEdCYTFjX0VhVkhrbGZUajRxZXc%5FZT0xa3liaFg&id=5D8626238470B49E%21s5d530eecddcf47b8b1b4a20f5e0def16&cid=5D8626238470B49E)
- MP_data fallback archive: [MP_data.zip on Hugging Face](https://huggingface.co/datasets/caobin/PyXplore/resolve/main/MP_data.zip?download=true)

The two JSON files under `data/` are intentionally ignored by Git because they contain third-party metadata/CIF content with redistribution constraints.

## Data Availability and License

`data/Exp_data/` contains RRUFF experimental spectra and matching structure files derived from external datasets. These data remain subject to the original RRUFF/XQueryer data licenses, database terms, and citation requirements.

Note: All data are subject to certain tolerances, as no two crystals are perfectly identical. The reference structure derived from **XQueryer** serves as the input for further structure determination during the refinement step.

## Generated Candidate CIFs

`RRUFF_data/` is generated and remains ignored by Git. Regenerate it with:

```bash
python scripts/build_rruff_data.py   --rank-csv stage2/analysis_results/temp_rank_0.csv   --mp-json data/mp_spacegroup.json   --exp-dir data/Exp_data   --out-dir data/RRUFF_data
```

If you want `temp_rank_0.csv` included in a release, place it under a tracked data or release-artifact location and update the command accordingly.

## Git LFS Recommendation

Use Git LFS for large artifacts that you are explicitly allowed to redistribute, such as model checkpoints:

```bash
git lfs install
git lfs track "*.pth"
```

Do not use Git LFS as a workaround for data that should not be redistributed.

## Still Excluded

- Generated refinement result folders
- Runtime logs and `.done` markers
- Newly generated checkpoints such as `checkpoint_*.pth`
- `data/mp_spacegroup.json` and `data/entries_dict.json`
- `data/MP_data/`
- `src/single_phase_xrd_identification/xqueryer/`
- Generated `RRUFF_data/` candidate folders unless you explicitly decide to publish them separately
