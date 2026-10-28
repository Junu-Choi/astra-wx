import gzip
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

DATA = ROOT / "data"
EXPECTED = DATA / "expected"
TEXT_KEYS = {"ens_pairing", "ref_on", "ours_definition", "ref_definition", "note", "notes", "label", "coverage",
             "source"}


def golden(name):
    with gzip.open(EXPECTED / name, "rt") as f:
        return json.load(f)


def as_json(obj):
    from astra_eval025.scorecard import _json_default
    return json.loads(json.dumps(obj, default=_json_default))


def diff(a, b, path="", out=None, skip=()):
    """Deep comparison of JSON-like objects: floats equal within math.isclose(rel_tol=1e-9, abs_tol=1e-12), NaN
    equal to NaN; everything else exactly."""
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k in skip:
                continue
            if k not in a or k not in b:
                out.append(f"{path}/{k}: missing")
            else:
                diff(a[k], b[k], f"{path}/{k}", out, skip)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diff(x, y, f"{path}[{i}]", out, skip)
    elif isinstance(a, float) and isinstance(b, float):
        if not (math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12) or (math.isnan(a) and math.isnan(b))):
            out.append(f"{path}: {a!r} != {b!r}")
    elif a != b:
        out.append(f"{path}: {a!r} != {b!r}")
    return out


@pytest.fixture(scope="session")
def P():
    from astra_eval025 import reproduce
    return reproduce.compute(DATA, log=lambda *a: None)
