# Data Directory

This directory stores the data artifacts used by the local workflow. The repository intentionally includes a curated `Exp_data/` subset and `match.txt` for reproducibility, while the complete upstream datasets remain external.

## Included or Expected

```text
Exp_data/             included: 444 RRUFF experimental spectra and matching *_CIF.txt files
MP_data/              external: MP-derived reference spectra / CIF text files
mp_spacegroup.json    external: MP metadata and CIF strings, ignored by Git
entries_dict.json     external: label-to-mp-id mapping, ignored by Git
match.txt             included: RRUFF-to-MP strict matching pairs
```

## Notes

* `Exp_data`:
  Contains diffraction and structure data of **strictly matched** RRUFF entries. Identified structures share the same elements and lattice constants. These files are redistributed here as a curated reproducibility subset, not as a complete mirror of the upstream datasets.

  Data availability and license: RRUFF experimental spectra and the associated structure files remain subject to the original RRUFF/XQueryer data licenses, database terms, and citation requirements. Note: All data are subject to certain tolerances, as no two crystals are perfectly identical. The reference structure derived from **XQueryer** serves as the input for further structure determination during the refinement step.

* `match.txt`:
  Records strict RRUFFID <-> MPID matches used for evaluation/reference. Treat this file as derived third-party matching metadata and cite the relevant sources listed in `../THIRD_PARTY_NOTICES.md`.

* `MP_data`:
  Contains diffraction and structure data of MP entries and is not mirrored in this repository.
  **If an unstable connection interrupts the download of this folder, a ZIP archive is also available on [HuggingFace](https://huggingface.co/datasets/caobin/PyXplore/resolve/main/MP_data.zip?download=true).**

- `MP_data/` is large and contains many small files.
- `mp_spacegroup.json` and `entries_dict.json` are not redistributed because they contain third-party metadata/CIF content.
- Generated refinement candidate folders such as `RRUFF_data/` are not included here by default. They can be regenerated with `single_phase_xrd_identification.refinement.build_rruff_data`.
- Verify redistributed artifacts with `../docs/ARTIFACTS_SHA256SUMS.txt`.
