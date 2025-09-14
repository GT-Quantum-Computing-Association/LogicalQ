# LogicalQ

Georgia Tech Quantum Computing Association's Quantum Error Correction Project

This is the repository where we store all code, tests, and demos for the implementation, simulation, and final hardware-ready code for our QEC toolkit.

The end goal of this project is to provide researchers with an easy-to-learn tool for working with LogicalCircuits, an extension of Qiskit's QuantumCircuits, with generalized and optimized QEC methodology.

## Installation

To install LogicalQ, clone this repository and install via `pip`:
```py
git clone https://github.com/GT-Quantum-Computing-Association/LogicalQ.git
cd LogicalQ
python -m virtualenv .venv
.venv/bin/activate
python -m pip install -e .
```

## Usage

LogicalQ is really easy to use, like this:
```py
from LogicalQ.Logical import LogicalCircuit, LogicalStatevector
from LogicalQ.Library.QECCs import steane_code

lqc = LogicalCircuit(2, **steane_code)
lqc.encode([0,1], initial_states=[1,0])

lqc.h(0)
lqc.cx(0, 1)

lqc.append_qec_cycle()

lqc.measure_all()

lqc.draw("mpl")

lsv = LogicalStatevector(lqc)
lsv.draw("latex")
```
See the `demos` directory for more examples!


