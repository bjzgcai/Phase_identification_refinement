# Third-Party Notices

The MIT license in `LICENSE` applies to this repository's original source code. It does not override the license, database terms, citation requirements, or redistribution conditions of third-party data, derived metadata, pretrained model weights, or upstream software referenced by this project.

## Redistributed Artifacts

The repository intentionally includes the following artifacts for reproducibility:

| Path | Contents | Redistribution note |
| --- | --- | --- |
| `data/Exp_data/` | 444 RRUFF experimental spectra and matching `*_CIF.txt` structure text files | Curated experimental subset derived from external RRUFF/XQueryer/PyXplore sources; subject to original source terms and citation requirements. |
| `data/match.txt` | Strict RRUFFID <-> MPID matching pairs | Derived matching metadata used for evaluation/reference; cite the upstream RRUFF, Materials Project, and XQueryer/PyXplore sources. |
| `src/single_phase_xrd_identification/stage1/checkpoints_stage1/model_best.pth` | Stage 1 pretrained PyTorch `state_dict` checkpoint | Included to reproduce the documented workflow; load only from trusted repository/release sources and verify the SHA256 hash. |

The complete upstream datasets are not mirrored here. `data/MP_data/`, `data/mp_spacegroup.json`, `data/entries_dict.json`, generated `data/RRUFF_data/`, and the external XQueryer baseline code are excluded unless separately obtained under their own terms.

## Source Attribution

| Source | Project use | Links and citation notes |
| --- | --- | --- |
| RRUFF Project | Experimental powder XRD spectra and associated reference structure information used in `data/Exp_data/`. | Website: <https://rruff.info/>. Follow the RRUFF database citation and usage terms for any reuse. |
| Materials Project | MP IDs, structures, and reference metadata used for candidate retrieval and matching workflows. | Website: <https://materialsproject.org/>. Follow the Materials Project terms of use and citation guidance for any reuse. |
| XQueryer / PyXplore | Dataset organization, strict matching workflow, and baseline comparison context. | Paper: <https://doi.org/10.1093/nsr/nwaf421>; website: <https://xqueryer.caobin.asia/about>; benchmarks: <https://github.com/WPEM/XqueryerBench>; dataset link is documented in `README.md` and `docs/DATA.md`. |
| PyTorch and scientific Python dependencies | Model training/inference and data processing. | Dependency versions are listed in `requirements-lock.txt`; package licenses should be reviewed with the locked environment before release. |

## Artifact Hashes

Verify redistributed artifacts with:

```bash
sha256sum -c docs/ARTIFACTS_SHA256SUMS.txt
```

Key single-file artifacts:

```text
b115fb536826790aaaabaa844034e310ee95ed4d31761fbcd67391519ddd9680  src/single_phase_xrd_identification/stage1/checkpoints_stage1/model_best.pth
e190ae856a11a2bcd5d5a999f8984bed05e744307b4e49ee690032dd3e6fb52a  data/match.txt
```

## Security and Trust Boundary

PyTorch checkpoints and pickle files can execute code when loaded through legacy pickle deserialization. Public model-loading paths in this repository use a `weights_only=True` compatibility wrapper when supported by PyTorch. Do not load checkpoints or `memory_bank.pkl` files from untrusted sources. The Stage 2 memory bank pickle is a local generated artifact and is not an interchange format.

## Notes on Scientific Tolerances

All data are subject to certain tolerances, as no two crystals are perfectly identical. The reference structure derived from **XQueryer** serves as the input for further structure determination during the refinement step.
