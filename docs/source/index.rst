LogicalQ Documentation
======================

LogicalQ
-----

**LogicalQ** is a free/open-source toolkit for quantum circuit development with built-in, generalized quantum error correction, developed by Georgia Tech students and alumni. It is distributed under the `MIT License`_.

.. _MIT License: https://en.wikipedia.org/wiki/MIT_License

Installation
-------------------------

LogicalQ is still in the development stage, so the latest version of LogicalQ is available on the GitHub under the `main branch`_.

.. _main branch: https://github.com/GT-Quantum-Computing-Association/LogicalQ.git

LogicalQ is geared towards any Unix/POSIX-like system, but it can also be installed on Windows and any other operating system which supports Python 3 and the necessary dependencies. We do our best to ensure compatibility with all major Linux, macOS, and Windows distributions. However, we recommend installing it in a Unix/POSIX-like environment for the highest-quality experience and guaranteed support of all features. If you use Windows, we recommend using WSL, which is documented at `https://learn.microsoft.com/en-us/windows/wsl/ <https://learn.microsoft.com/en-us/windows/wsl/>`__.

LogicalQ can be installed as follows:

.. code:: bash

    git clone https://github.com/GT-Quantum-Computing-Association/LogicalQ.git
    cd LogicalQ
    python -m pip install virtualenv
    python -m virtualenv .venv
    .venv/bin/activate # `Scripts` instead of `bin` on Windows
    python -m pip install -e .

Documentation and Mailing Lists
-------------------------------

Documentation is hosted online on ReadtheDocs at `https://logicalq.readthedocs.io <https://logicalq.readthedocs.io>`__.

Acknowledgements
----------------
LogicalQ wouldn't've been possible without the amazing members of Georgia Tech Quantum Computing Association who contributed to its development. We are also very grateful to all the developers of Qiskit, Stim, Staq, PECOS, and MQT, which have been invaluable references in developing this codebase. We are especially grateful to IBM Quantum for maintaining Qiskit as an open-source library which we use as the parent class for many of our own classes. *Please cite these authors* if you use their code or parts of their code which are present in LogicalQ.

Contact and Feedback
--------------------

For bug reports and feature requests, please `file a GitHub Issue`_.

.. _file a GitHub Issue: https://github.com/GT-Quantum-Computing-Association/LogicalQ/issues

If you have any questions, comments, or concerns regarding LogicalQ, feel free to contact the GT QCA team at `aliceandbob@gatechquantum.com <mailto:aliceandbob@gatechquantum.com>`__.

.. toctree::

   Home <self>
   API Reference

