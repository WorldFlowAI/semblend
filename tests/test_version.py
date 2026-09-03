"""The package's __version__ must be the installed distribution version.

Regression: the 0.3.17 wheel shipped __version__ = "0.3.13" because the
attribute was a hand-maintained literal. Engine integrations (the SGLang
fuzzy-match provider) gate on semblend.__version__, so the drift turned a
correct install into an ImportError at scheduler start.
"""

from __future__ import annotations

import importlib.metadata

import semblend


def test_dunder_version_matches_distribution_metadata():
    assert semblend.__version__ == importlib.metadata.version("semblend")


def test_dunder_version_is_not_the_stale_literal():
    assert semblend.__version__ != "0.3.13" or importlib.metadata.version("semblend") == "0.3.13"
