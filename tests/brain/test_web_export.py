"""The browser must receive the exact anatomy packaged with the trained checkpoint."""

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.web.export_web import bundle_subgraph, export_web

BUNDLE = Path("data/bundles/real_fast")
WEB = Path("frontend/public/data/brain")


def test_committed_edges_match_every_bundle_row() -> None:
    sub = bundle_subgraph(BUNDLE)  # checks manifest hashes and checkpoint metadata
    neurons = pd.read_parquet(sub / "neurons.parquet")
    edges = pd.read_parquet(sub / "edges.parquet")
    exported = json.loads((WEB / "edges.json").read_text())
    assert exported["body_id"] == neurons["body_id"].tolist()
    assert exported["body_id"] == json.loads((WEB / "neurons.json").read_text())["body_id"]
    assert exported["n_edges"] == len(edges) == 340_922
    assert exported["n_synapses"] == int(edges["weight"].sum()) == 6_175_540
    for field in ("pre", "post", "weight", "sign"):
        assert exported[field] == edges[field].tolist()


def test_bundle_export_is_reproducible(tmp_path: Path) -> None:
    export_web(bundle_subgraph(BUNDLE), tmp_path)
    for file in ("neurons.json", "edges.json", "groups.json", "columns.json", "controls.json"):
        assert (tmp_path / file).read_bytes() == (WEB / file).read_bytes()


def test_bundle_export_rejects_mismatched_anatomy(tmp_path: Path) -> None:
    # Reuse the real tables while corrupting the recorded fingerprint.
    (tmp_path / "subgraph").symlink_to((BUNDLE / "subgraph").resolve(), target_is_directory=True)
    (tmp_path / "bc_950.npz").symlink_to((BUNDLE / "bc_950.npz").resolve())
    manifest = json.loads((BUNDLE / "MANIFEST.json").read_text())
    manifest["subgraph_fingerprint"]["neurons_sha256"] = "wrong-order"
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="neurons_sha256"):
        bundle_subgraph(tmp_path)
