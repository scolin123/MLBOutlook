"""Streamlit-cached wrappers around `components.data`, so a page rerun doesn't reread files."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components import data


@st.cache_data(show_spinner=False)
def seasons() -> pd.DataFrame:
    return data.seasons()


@st.cache_data(show_spinner=False)
def forecasts() -> pd.DataFrame:
    return data.forecasts()


@st.cache_data(show_spinner=False)
def artifact(name: str) -> pd.DataFrame:
    return data.artifact(name)


@st.cache_data(show_spinner=False)
def holdout_artifact(name: str) -> pd.DataFrame:
    return data.holdout_artifact(name)


@st.cache_data(show_spinner="Finding comps...")
def comps(player_id: int, k: int = 10):
    return data.comps(seasons(), player_id, k=k)


def require_files() -> None:
    try:
        data.check_files()
    except data.MissingArtifacts as exc:
        st.error(str(exc))
        st.stop()
