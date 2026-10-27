"""Leakage guard.

Contract: every feature row for target season T is derived only from data dated
before the forecast date for T. The feature builder must expose, per row,
`max_source_season` (latest season of any input); this test asserts
max_source_season < target_season for every row.
"""

import pytest


@pytest.mark.skip(reason="feature builder not implemented yet (week 2)")
def test_features_use_only_prior_seasons():
    raise NotImplementedError
