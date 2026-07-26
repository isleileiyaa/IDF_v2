#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# NOTE: this is a trimmed copy of uni2ts/model/moirai2/__init__.py.
# The upstream version also does `from .forecast import Moirai2Forecast`,
# which pulls in gluonts/lightning at import time. RIDDE only needs the raw
# Moirai2Module backbone (see TS-RAG/third_party/README.md), so that import
# is intentionally omitted here to keep this vendored subset dependency-free
# beyond torch/einops/jaxtyping/huggingface_hub.

from .module import Moirai2Module

__all__ = [
    "Moirai2Module",
]
