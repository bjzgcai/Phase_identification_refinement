# Data Directory

This directory is the expected local location for external data. The public repository does not redistribute the full datasets.

## Included

```text
Exp_data/             RRUFF experimental spectra and matching *_CIF.txt files, external
MP_data/              MP-derived reference spectra / CIF text files, external
mp_spacegroup.json    MP metadata and CIF strings, external and ignored by Git
entries_dict.json     label-to-mp-id mapping, external and ignored by Git
match.txt             RRUFF-to-MP strict matching pairs, if permitted
```

## Notes

* `Exp_data`:
  Contains diffraction and structure data of **strictly matched** RRUFF entries. Identified structures share the same elements and lattice constants.

  Data availability and license: RRUFF experimental spectra and the associated structure files remain subject to the original RRUFF/XQueryer data licenses, database terms, and citation requirements. Note: All data are subject to certain tolerances, as no two crystals are perfectly identical. The reference structure derived from **XQueryer** serves as the input for further structure determination during the refinement step.

* `MP_data`:
  Contains diffraction and structure data of MP entries.
  **If an unstable connection interrupts the download of this folder, a ZIP archive is also available on [HuggingFace](https://huggingface.co/datasets/caobin/PyXplore/resolve/main/MP_data.zip?download=true).**

- `MP_data/` is large and contains many small files.
- `mp_spacegroup.json` and `entries_dict.json` are not redistributed because they contain third-party metadata/CIF content.
- `match.txt` records strict RRUFFID <-> MPID matches used for evaluation/reference.
- Generated refinement candidate folders such as `RRUFF_data/` are not included here by default. They can be regenerated with `single_phase_xrd_identification.refinement.build_rruff_data`.
