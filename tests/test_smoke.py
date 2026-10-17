"""Smoke test: every top-level package imports."""

import importlib

import pytest


@pytest.mark.parametrize("name", ["brain", "fly_saber", "training", "server"])
def test_import(name: str) -> None:
    assert importlib.import_module(name).__doc__
