"""Smoke test: the package and every subpackage import cleanly."""

import importlib

import pytest

SUBPACKAGES = ["census", "sites", "harmonize", "evaluation", "federation", "policy", "agent"]


def test_package_imports():
    pkg = importlib.import_module("fedkg_spacebio")
    assert pkg.__version__


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name):
    mod = importlib.import_module(f"fedkg_spacebio.{name}")
    assert mod.__doc__, f"{name} should document its responsibility"


def test_nvflare_recipe_api_available():
    """FLARE's Job Recipe API is importable (guards against a broken or wrong-version install)."""
    recipe = importlib.import_module("nvflare.recipe")
    assert recipe is not None
