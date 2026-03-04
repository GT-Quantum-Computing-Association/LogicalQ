# LogicalQ

Georgia Tech Quantum Computing Association's Quantum Error Correction Project

This is the repository where we store all code, tests, and demos for the implementation, simulation, and final hardware-ready code for our QEC toolkit.

The end goal of this project is to provide researchers with an easy-to-learn tool for working with `LogicalCircuit`s, an extension of `Qiskit`'s `QuantumCircuit`s with generalized and optimized QEC methodology.

For more information about `LogicalQ`, please refer to our documentation: [https://logicalq.readthedocs.io/](https://logicalq.readthedocs.io/).

## Installation

To install the latest stable release of LogicalQ, install directly from the PyPI index via `pip`:
```py
python -m pip install virtualenv
python -m virtualenv .venv
.venv/bin/activate
python -m pip install LogicalQ
```

To install the latest nightly version of LogicalQ, clone this repository and install via `pip`:
```py
git clone https://github.com/GT-Quantum-Computing-Association/LogicalQ.git
cd LogicalQ
python -m pip install virtualenv
python -m virtualenv .venv
.venv/bin/activate
python -m pip install -e .
```

`LogicalQ` is written to function best on Unix/POSIX-like system, but it can also be installed on Windows and any other operating system which supports Python 3 and the necessary dependencies. We do our best to ensure compatibility with all major Linux, macOS, and Windows distributions. However, we recommend installing it in a Unix/POSIX-like environment for the highest-quality experience and guaranteed support of all features. If you use Windows, we recommend using WSL, which is documented at [https://learn.microsoft.com/en-us/windows/wsl/](https://learn.microsoft.com/en-us/windows/wsl/).

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

We welcome contributions from anyone! A GitHub Issue may be used for making requests or reporting bugs. Any contributions should be merged from a branch (preferred for internal contributors from GT QCA) or forks (default for external contributors outside GT QCA). Please read our [contributing guidelines](./CONTRIBUTING.md).

## Acknowledgements

`LogicalQ` wouldn't've been possible without the amazing members of Georgia Tech Quantum Computing Association who contributed to its development. We are also very grateful to all the developers of `Qiskit`, `Stim`, `Stac`, `PECOS`, `MQT`, and `tqec`, all of which have been invaluable references in developing this codebase. We are especially grateful to IBM Quantum for maintaining `Qiskit` as an open-source library which we use as the parent class for many of our own classes. *Please cite these authors* if you use their code or parts of their code which are present in `LogicalQ`.

## Contact and feedback

If you have any questions, comments, or concerns regarding `LogicalQ`, feel free to contact the GT QCA team at [aliceandbob@gatechquantum.com](<mailto:aliceandbob@gatechquantum.com>).

