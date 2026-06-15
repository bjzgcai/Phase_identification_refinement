# Refinement Workflow

The refinement module accepts one experimental pattern, one entry CIF, and a folder of candidate CIFs.

```bash
python -m single_phase_xrd_identification.refinement.xrd_refinement   --xy data/Exp_data/R040009.csv   --main data/RRUFF_data/R040009/R040009.cif   --imp data/RRUFF_data/R040009   --wl 1.541838
```

The `--main` argument is an entry point. During initialization, the code computes Pearson correlation between the observed pattern and each candidate simulated profile, then automatically places the highest-correlation phase first.
