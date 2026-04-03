# QML XOR Gate Model

This project trains a 2-qubit variational circuit to reproduce the XOR truth table.

## What changed

- Added `xor_model.py` so the training logic can be run outside the notebook.
- Added `circuit_maker.py` to provide the missing circuit rendering helper from a clean clone.
- Removed the notebook's dependency on Qiskit-only display code for simple circuit inspection.

## Run

```bash
python xor_model.py
```

## Notebook note

`XOR.ipynb` now uses the lightweight helper in `circuit_maker.py` to print an ASCII view of the learned circuit.
