"""Load MaleCNS v1.0 neurons and edges (spec 5.1).

Owner: Person 1. File formats are documented in docs/DATA_NOTES.md.
"""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

ANNOTATIONS = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS = "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
FILES = (ANNOTATIONS, NEUROTRANSMITTERS, WEIGHTS)
BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"


def load_neurons(raw_dir: str | Path) -> pd.DataFrame:
    """Traced, typed neurons indexed by body id.

    Columns: type, superclass, side, x, y, z, hex1, hex2, nt.
    Position is the soma in 8 nm voxels, NaN when missing.
    """
    raw_dir = Path(raw_dir)
    ann = pd.read_feather(raw_dir / ANNOTATIONS)
    ann = ann[(ann["status"] == "Traced") & ann["type"].notna()].set_index("bodyId")

    # Photoreceptors have no somaSide; their rootSide is set instead.
    side = ann["somaSide"].fillna(ann["rootSide"])
    soma = ann["somaLocation"].apply(lambda p: list(p) if p is not None else [None] * 3)
    xyz = pd.DataFrame(soma.tolist(), index=ann.index, columns=["x", "y", "z"], dtype="float64")

    nt = pd.read_feather(raw_dir / NEUROTRANSMITTERS, columns=["body", "consensus_nt"])
    nt = nt.drop_duplicates("body").set_index("body")["consensus_nt"]

    out = pd.DataFrame(
        {
            "type": ann["type"].astype(str),
            "superclass": ann["superclass"].astype(str),
            "side": side,
            "hex1": ann["assignedOlHex1"],
            "hex2": ann["assignedOlHex2"],
            "nt": nt.reindex(ann.index).fillna("unclear"),
        }
    ).join(xyz)
    out.index.name = "body_id"
    return out


def load_edges(raw_dir: str | Path, body_ids: pd.Index, min_synapses: int) -> pd.DataFrame:
    """Edges between `body_ids` with at least `min_synapses` synapses.

    Filtering the 152M-row file takes a minute, so the result is cached
    next to the raw data, keyed by threshold and the neuron set size.
    """
    raw_dir = Path(raw_dir)
    cache = raw_dir / f"edges_cache_w{min_synapses}_n{len(body_ids)}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    table = feather.read_table(raw_dir / WEIGHTS)
    table = table.filter(pc.greater_equal(table["weight"], min_synapses))
    ids = pa.array(body_ids.to_numpy())
    keep = pc.and_(pc.is_in(table["body_pre"], ids), pc.is_in(table["body_post"], ids))
    edges = table.filter(keep).to_pandas()
    edges.to_parquet(cache)
    return edges
