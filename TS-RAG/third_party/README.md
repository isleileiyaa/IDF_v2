# third_party/

Backbone dependencies that don't ship as clean, narrowly-scoped PyPI packages
live here, one subdirectory per backbone. See each backbone's own section
below for why it's handled the way it is.

## third_party/uni2ts

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

## Updating (uni2ts)

If upstream `uni2ts` changes `Moirai2Module`'s architecture, re-fetch the
files listed above from the `uni2ts` GitHub repo and re-apply the same trim
to `model/moirai2/__init__.py`.

## third_party/timesfm

**Not vendored** -- a real `git clone` of
[google-research/timesfm](https://github.com/google-research/timesfm),
`pip install -e`'d into the `tsrag` env (see setup below). Unlike `uni2ts`,
this repo's base + `[torch]` extras (`numpy`, `huggingface_hub`,
`safetensors`, `torch>=2.0.0` with no upper pin) don't conflict with anything
already installed, so there was no need to hand-pick files.
`TS-RAG/third_party/timesfm/` is gitignored (it's a live pip-editable-install
target, not something we've reviewed file-by-file the way `uni2ts` was).

Setup, if this directory is missing on a fresh checkout:
```bash
cd TS-RAG/third_party
git clone https://github.com/google-research/timesfm.git
cd timesfm && pip install -e ".[torch]"
```

`models/TimesFM25.py` loads `TimesFM_2p5_200M_torch` from
`timesfm.timesfm_2p5.timesfm_2p5_torch`, checkpoint
`google/timesfm-2.5-200m-pytorch`. This is deliberately **not** the
HuggingFace `transformers` port (`transformers.TimesFm2_5ModelForPrediction`,
checkpoint `google/timesfm-2.5-200m-transformers`) -- same architecture,
different checkpoint repo and loading code; don't mix the two.
