"""The prediction-interval convention the interval metrics score against.

Shared by `pinball` and `interval_score`, which are the two metrics whose *value* depends on which
quantiles the bounds represent — a pinball loss computed at the wrong q is a different number, not
a wrong one, so nothing downstream would notice. Kept in one file so the two can never disagree.

These mirror `models.base_model.DEFAULT_QUANTILES`, which is where a model that builds its band
from residual quantiles gets the bounds it writes. If that changes, this changes with it.
"""

from __future__ import annotations

# Lower bound at the 0.1 quantile, upper at the 0.9.
LOWER_Q = 0.1
UPPER_Q = 0.9

# The nominal miss rate of the [0.1, 0.9] interval — the α the Winkler interval score penalises
# with. Derived from the bounds above so the two can never drift apart.
INTERVAL_ALPHA = 2.0 * LOWER_Q
