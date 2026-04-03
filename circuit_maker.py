from __future__ import annotations

import numpy as np


def create_xor_circuit_text(params: np.ndarray) -> str:
    rounded = np.round(np.asarray(params, dtype=float), 3)
    p0, p1, p2, p3 = rounded.tolist()
    return "\n".join(
        [
            f"q_0: --Rx({p0})--*--Rx({p2})----------------",
            f"q_1: --Rx({p1})--X--Rx({p3})--Measure P(1)--",
        ]
    )
