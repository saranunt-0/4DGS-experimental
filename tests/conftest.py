import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def small_tree():
    from gs4d.procedural_tree import TreeParams, generate_tree

    params = TreeParams(children=(4, 3, 3, 2), leaves_per_branch=(10, 16), segments=(6, 4, 3, 3, 2))
    return generate_tree(params)
