"""Leakage guard.

Placeholder until the as-of feature builder exists (week 2). The contract:
every feature row for target season T must be derived only from data dated
before the forecast date for T. The builder must expose, per feature row,
`max_source_season` (the latest season of any input) and the test asserts
max_source_season < target_season for every row.
"""

import pytest


@pytest.mark.skip(reason="feature builder not implemented yet (week 2)")
def test_features_use_only_prior_seasons():
    raise NotImplementedError
