"""Cell-type based visual-to-motor subgraph extraction (spec 5.2).

Owner: Person 1.

Pipeline:
1. Give columnar optic lobe neurons that lack a hex column (photoreceptors,
   T4, T5) the column of their strongest partner that has one.
2. Keep columnar types in every `column_stride`-th column of each eye.
3. Keep all visual projection neurons and all descending neurons.
4. Add central brain bridge neurons between projection neurons and DNs.
5. Induce edges, fill missing positions, and reindex to 0..N-1.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from brain.config import SubgraphConfig
from brain.connectome.signs import nt_to_sign

ROLES = ("columnar", "projection", "descending", "intermediate")


@dataclass(frozen=True)
class Subgraph:
    neurons: pd.DataFrame  # index 0..N-1; body_id, type, superclass, side, x, y, z, ...
    edges: pd.DataFrame  # pre, post (subgraph indices), weight, sign


def match_types(types: pd.Series, patterns: list[str]) -> pd.Series:
    """True where `types` fully matches any regex in `patterns`."""
    return types.str.fullmatch("|".join(f"(?:{p})" for p in patterns))


def _strongest_partner(edges: pd.DataFrame, value: pd.Series) -> pd.Series:
    """For each body, the `value` of its strongest partner (either direction) that has one."""
    both = pd.concat(
        [
            edges.rename(columns={"body_pre": "body", "body_post": "partner"}),
            edges.rename(columns={"body_post": "body", "body_pre": "partner"}),
        ]
    )[["body", "partner", "weight"]]
    both = both[both["partner"].isin(value.dropna().index)]
    best = both.sort_values("weight").groupby("body").tail(1).set_index("body")["partner"]
    return pd.Series(value.reindex(best.to_numpy()).to_numpy(), index=best.index)


def inherit_hex(
    neurons: pd.DataFrame, edges: pd.DataFrame, superclasses: list[str], passes: int
) -> pd.DataFrame:
    """Fill missing hex1/hex2 from the strongest hex-assigned partner, `passes` times."""
    out = neurons.copy()
    eligible = out["superclass"].isin(superclasses)
    for _ in range(passes):
        missing = eligible & out["hex1"].isna()
        if not missing.any():
            break
        key = (
            out["hex1"].astype("Int64").astype(str) + "," + out["hex2"].astype("Int64").astype(str)
        )
        key = key.where(out["hex1"].notna())
        inherited = _strongest_partner(edges, key).reindex(out.index[missing]).dropna()
        hex12 = inherited.str.split(",", expand=True).astype(float)
        if hex12.empty:
            break
        out.loc[hex12.index, "hex1"] = hex12[0]
        out.loc[hex12.index, "hex2"] = hex12[1]
    return out


def select_neurons(neurons: pd.DataFrame, edges: pd.DataFrame, cfg: SubgraphConfig) -> pd.Series:
    """Role per kept body id (see ROLES). Bodies not kept are absent."""
    columnar = match_types(neurons["type"], cfg.columnar_types)
    sampled = (neurons["hex1"] % cfg.column_stride == 0) & (
        neurons["hex2"] % cfg.column_stride == 0
    )
    projection = match_types(neurons["type"], cfg.projection_types)
    descending = neurons["superclass"] == cfg.descending_superclass

    role = pd.Series(pd.NA, index=neurons.index, dtype="object")
    role[columnar & sampled] = "columnar"
    role[projection] = "projection"
    role[descending] = "descending"
    seeds = role.dropna()

    budget = min(cfg.max_intermediates, cfg.max_neurons - len(seeds))
    if budget > 0:
        proj_ids = seeds.index[seeds == "projection"]
        dn_ids = seeds.index[seeds == "descending"]
        in_from_proj = edges[edges["body_pre"].isin(proj_ids)].groupby("body_post")["weight"].sum()
        out_to_dn = edges[edges["body_post"].isin(dn_ids)].groupby("body_pre")["weight"].sum()
        # A missing side means zero synapses, not "skip": bridges need both.
        score = pd.concat([in_from_proj, out_to_dn], axis=1).fillna(0).min(axis=1)
        score = score[~score.index.isin(seeds.index) & (score > 0)]
        role[score.nlargest(budget).index] = "intermediate"

    return role.dropna()


def _fill_positions(neurons: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Neurons without a soma take their strongest partner's position (for the 3D view)."""
    out = neurons.copy()
    has_pos = out["x"].notna()
    out["position_inherited"] = False
    for _ in range(2):
        missing = out.index[out["x"].isna()]
        if len(missing) == 0:
            break
        pos = out[["x", "y", "z"]].dropna()
        key = pd.Series(np.arange(len(pos)), index=pos.index, dtype="float64")
        src = _strongest_partner(edges, key).reindex(missing).dropna().astype(int)
        out.loc[src.index, ["x", "y", "z"]] = pos.to_numpy()[src.to_numpy()]
    out["position_inherited"] = ~has_pos & out["x"].notna()
    return out


def build_subgraph(neurons: pd.DataFrame, edges: pd.DataFrame, cfg: SubgraphConfig) -> Subgraph:
    """Run the full extraction. `edges` must already be thresholded."""
    neurons = inherit_hex(neurons, edges, cfg.hex_inherit_superclasses, cfg.hex_inherit_passes)
    role = select_neurons(neurons, edges, cfg)

    kept = neurons.loc[role.index].assign(role=role, role_order=role.map(ROLES.index))
    kept = kept.sort_values(["role_order", "type", "side"]).drop(columns="role_order")
    sub_edges = edges[edges["body_pre"].isin(kept.index) & edges["body_post"].isin(kept.index)]
    kept = _fill_positions(kept, sub_edges)

    kept["sign"] = nt_to_sign(kept["nt"])
    index_of = pd.Series(np.arange(len(kept)), index=kept.index)
    out_edges = pd.DataFrame(
        {
            "pre": index_of[sub_edges["body_pre"]].to_numpy(dtype="int32"),
            "post": index_of[sub_edges["body_post"]].to_numpy(dtype="int32"),
            "weight": sub_edges["weight"].to_numpy(dtype="int32"),
        }
    )
    out_edges["sign"] = kept["sign"].to_numpy()[out_edges["pre"].to_numpy()]
    out_edges = out_edges.sort_values(["post", "pre"], ignore_index=True)

    out_neurons = kept.reset_index()
    return Subgraph(neurons=out_neurons, edges=out_edges)
