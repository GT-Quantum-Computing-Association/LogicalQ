# LogicalQ

Georgia Tech Quantum Computing Association's Quantum Error Correction Project

This is the repository where we store all code, tests, and demos for the implementation, simulation, and final hardware-ready code for our QEC toolkit.

The end goal of this project is to provide researchers with an easy-to-learn tool for working with `LogicalCircuit`s, an extension of Qiskit's `QuantumCircuit`s with generalized and optimized QEC methodology.

For more information about `LogicalQ`, please refer to our documentation: [https://logicalq.readthedocs.io/](https://logicalq.readthedocs.io/).

## Installation

To install LogicalQ, clone this repository and install via `pip`:
```py
git clone https://github.com/GT-Quantum-Computing-Association/LogicalQ.git
cd LogicalQ
python -m pip install virtualenv
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

## Contributing

We welcome contributions from anyone! A GitHub Issue may be used for making requests or reporting bugs. Any contributions should be merged from a branch (preferred for internal contributors from QCA) or forks (default for external contributors outside QCA). Please read our [contributing guidelines](./CONTRIBUTING.md)

