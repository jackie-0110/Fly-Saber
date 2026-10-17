"""Neurotransmitter to synaptic sign mapping (spec 5.3).

Owner: Person 1.

Glutamate is inhibitory in the fly central brain via GluCl (a simplification;
it can be excitatory at some synapses). Histamine is inhibitory via HisCl1;
photoreceptors are histaminergic, so mapping it to 0 would disconnect the eyes.
Modulatory transmitters get sign 0 for fast transmission and are grouped for
the mood sliders instead.
"""

import pandas as pd

SIGNS: dict[str, int] = {
    "acetylcholine": 1,
    "gaba": -1,
    "glutamate": -1,
    "histamine": -1,
    "dopamine": 0,
    "serotonin": 0,
    "octopamine": 0,
    "unclear": 0,
}

MODULATORY: tuple[str, ...] = ("dopamine", "serotonin", "octopamine")


def nt_to_sign(nt: pd.Series, signs: dict[str, int] = SIGNS) -> pd.Series:
    """Map neurotransmitter names to +1, -1, or 0. Unknown names map to 0."""
    return nt.map(signs).fillna(0).astype("int8")
