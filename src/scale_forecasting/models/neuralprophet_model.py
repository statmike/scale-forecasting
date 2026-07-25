"""NeuralProphet — the one model that benefits from a GPU (DESIGN §11).

One model, one file (CONTRACTS §1). Owned by BUILD step 2.5: implements BaseModel
and ends with a register(...) call. Runtime: python.
"""

from __future__ import annotations
