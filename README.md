# PINN Code

Physics-Informed Neural Networks for solving partial differential equations.

---

## Files

### Improved Architecture (Sin-Tanh Hybrid Activation + Adaptive Gradient Clipping)

| File         | Problem                                                                                      |
| ------------ | -------------------------------------------------------------------------------------------- |
| `code1.py` | **1D Heat Conduction** $u_t = \alpha u_{xx}$                                         |
| `code2.py` | **2D Flow Around a Cylinder** Navier-Stokes (streamfunction)                           |
| `code3.py` | **Hydrogen Molecule H₂** Time-independent Schrödinger equation (bonding/antibonding) |
| `code4.py` | **1D Quantum Tunneling** Time-dependent Schrödinger equation                          |

### Standard Architecture (Tanh Activation)

| File            | Problem                   |
| --------------- | ------------------------- |
| `sd_code1.py` | 1D Heat Conduction        |
| `sd_code2.py` | 2D Flow Around a Cylinder |
| `sd_code3.py` | Hydrogen Molecule H₂     |
| `sd_code4.py` | 1D Quantum Tunneling      |

### Numerical Reference Solutions

| File                   | Method                   | Problem                |
| ---------------------- | ------------------------ | ---------------------- |
| `CFD_code.py`        | Finite Difference Method | Flow Around a Cylinder |
| `numerical_code3.py` | FDM + Eigenvalue Solver  | H₂ Molecule           |

### Utilities

| File                                        | Description                     |
| ------------------------------------------- | ------------------------------- |
| `Code for duplication rate comparison.py` | Code similarity comparison tool |

---

## Usage

```bash
pip install torch numpy pandas scipy openpyxl

# Run any file independently
python code1.py
python sd_code1.py   # standard version
```

Each file trains for 30000 epochs, outputs `.xlsx` results and `.pth` model weights.
