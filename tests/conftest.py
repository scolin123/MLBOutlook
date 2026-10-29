import pytest

from synthetic import make_seasons


@pytest.fixture
def seasons_df():
    return make_seasons()
