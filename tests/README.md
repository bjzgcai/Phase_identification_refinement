# Tests

Run the lightweight tests without the full external datasets:

```bash
PYTHONPATH=src pytest tests
```

Current coverage focuses on:

- Parsing a small RRUFF `_CIF.txt` fixture.
- Sorting and truncating Stage 2 candidate rank CSV rows.
- Loading a local PyTorch state dict through the safe checkpoint wrapper.

Broader integration tests that require the full MP/RRUFF datasets should be kept separate from this smoke suite.
