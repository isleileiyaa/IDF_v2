# third_party/uni2ts

Vendored subset of [SalesforceAIResearch/uni2ts](https://github.com/SalesforceAIResearch/uni2ts)
(commit: `main` as of 2026-07-26), Apache-2.0 licensed (see `uni2ts/LICENSE.txt`).

## Why vendored instead of `pip install uni2ts`

The official PyPI package pins `torch<2.5`, `gluonts~=0.14.3`, `jax`, `lightning`, etc.
Installing it with its declared dependencies would downgrade the `tsrag` conda
env's torch 2.6/CUDA 12.4 build and could break other installed packages
(chronos-forecasting, faiss-gpu). RIDDE only needs `Moirai2Module`, the raw
backbone `nn.Module` used to load `Salesforce/moirai-2.0-R-small` weights and
run its forward pass — none of the GluonTS/Lightning/JAX training or
forecasting-pipeline code (`Moirai2Forecast`, data transforms, etc.) is used.

Everything under this directory is the minimal, pure-PyTorch dependency
closure of `uni2ts.model.moirai2.Moirai2Module`:

```
uni2ts/
  __init__.py, __about__.py
  common/torch_util.py               # packed_causal_attention_mask
  model/moirai2/module.py            # Moirai2Module
  model/moirai2/__init__.py          # trimmed: does NOT import forecast.py (avoids gluonts/lightning)
  module/transformer.py, attention.py, ffn.py, norm.py,
         packed_scaler.py, ts_embed.py, position/*.py
```

Only extra runtime dependency needed beyond what TS-RAG already has:
`jaxtyping` (used purely for type annotations, no heavy transitive deps).

`model/moirai2/__init__.py` is the one file intentionally modified from
upstream (see the comment inside it) — every other file here is an unmodified
copy of the corresponding upstream file.

## How it's imported

`TS-RAG/models/Moirai2.py` inserts `TS-RAG/third_party` onto `sys.path` and
then does `from uni2ts.model.moirai2 import Moirai2Module`, so this package
resolves as a normal top-level `uni2ts` import without touching the `tsrag`
env's installed packages.

## Updating

If upstream `uni2ts` changes `Moirai2Module`'s architecture, re-fetch the
files listed above from the `uni2ts` GitHub repo and re-apply the same trim
to `model/moirai2/__init__.py`.
