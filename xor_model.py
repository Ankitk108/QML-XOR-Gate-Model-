from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qutip import basis, expect, qeye, tensor
from qutip.qip.operations import cnot, rx
from scipy.optimize import minimize


TRAINING_DATA = {
    (0, 0): 0,
    (0, 1): 1,
    (1, 0): 1,
    (1, 1): 0,
}


@dataclass
class XorTrainingResult:
    params: np.ndarray
    loss: float
    records: list[tuple[int, int, int, float, int]]


def make_xor_circuit():
    projector = tensor(qeye(2), basis(2, 1) * basis(2, 1).dag())

    def circuit_expectation(params: np.ndarray, bits: tuple[int, int]) -> float:
        x1, x2 = bits
        state = tensor(basis(2, x1), basis(2, x2))
        layer_1 = tensor(rx(params[0]), rx(params[1]))
        entangler = cnot(2, 0, 1)
        layer_2 = tensor(rx(params[2]), rx(params[3]))
        final = (layer_2 * entangler * layer_1) * state
        return float(expect(projector, final).real)

    return circuit_expectation


def train_xor_model(seed: int = 412, max_iter: int = 250) -> XorTrainingResult:
    circuit_expectation = make_xor_circuit()

    def loss(params: np.ndarray) -> float:
        errors = []
        for bits, target in TRAINING_DATA.items():
            prediction = circuit_expectation(params, bits)
            errors.append((prediction - target) ** 2)
        return float(np.mean(errors))

    rng = np.random.default_rng(seed)
    init = rng.uniform(0, 2 * np.pi, 4)
    result = minimize(loss, init, bounds=[(0, 2 * np.pi)] * 4, options={"maxiter": max_iter})

    records = []
    for bits, target in TRAINING_DATA.items():
        probability = circuit_expectation(result.x, bits)
        prediction = int(round(probability))
        records.append((bits[0], bits[1], target, float(probability), prediction))

    return XorTrainingResult(params=result.x, loss=float(result.fun), records=records)


if __name__ == "__main__":
    trained = train_xor_model()
    print("Learned params:", np.round(trained.params, 3))
    for x1, x2, target, probability, prediction in trained.records:
        print(f"({x1}, {x2}) -> P(1)={probability:.4f}, pred={prediction}, target={target}")
