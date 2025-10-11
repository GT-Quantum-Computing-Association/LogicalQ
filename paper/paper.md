---
title: "LogicalQ: A toolkit for quantum circuit development with built-in, generalized quantum error correction"
tags:
  - Python
  - quantum computing
  - quantum error correction
authors:
  - name: Rasmit Devkota
    orcid: 
    affiliation: "1,2"
  - name: Ben Hagan
    affiliation: 3
  - name: Noah Song
    affiliation: 2
  - name: Zhixin Song
    affiliation: 1
  - name: David Lloyd George
    affiliation: 4
  - name: Younos Hashem
    affiliation: 1
  - name: Nolan Heffner
    affiliation: "1,2"
  - name: Fiyin Makinde
    affiliation: 5
  - name: Richard Yu
    affiliation: "2,6"
affiliations:
 - name: School of Physics, College of Science, Georgia Institute of Technology, Atlanta, GA 30332, USA
   index: 1
 - name: School of Mathematics, College of Science, Georgia Institute of Technology, Atlanta, GA 30332, USA
   index: 2
 - name: Ming Hsieh Department of Electrical and Computer Engineering, Viterbi School of Engineering, University of Southern California, Los Angeles, CA 90089, United States
   index: 3
 - name: Department of Physics, Duke University, Durham, North Carolina 27708, United States
   index: 4
 - name: School of Electrical and Computer Engineering, College of Engineering, Georgia Institute of Technology, Atlanta, GA 30332, USA
   index: 5
 - name: School of Computer Science, College of Computing, Georgia Institute of Technology, Atlanta, GA 30332, USA
   index: 6
date: XX October 2025
bibliography: paper.bib
---

# Summary

`LogicalQ` is a Python toolkit for quantum circuit development with built-in, generalized quantum error mitigation, detection, and correction (QEMDAC). `LogicalQ` inherits many of its data structures from, and thus is designed to interface well with, the `Qiskit` and `Qiskit Aer` packages [@Javadi-Abhari:2024].

The source code for `LogicalQ` is available on GitHub at https://github.com/GT-Quantum-Computing-Association/LogicalQ/. It can be installed via `pip` from the `pypi` index at https://pypi.org/project/LogicalQ/. Its documentation is hosted publicly at https://logicalq.readthedocs.io/.

# Statement of need

Quantum computing presents a new model for computation which may significantly accelerate scientific discovery in many fields, from physics to finance. However, the current era of quantum hardware is still noisy, and quantum error mitigation (QEM), quantum error detection (QED), and quantum error correction (QEC) techniques are necessary for the execution of utility-scale algorithms. Moreover, the broader scientific consensus is that these techniques will always be necessary to some extent. Although there exist a number of libraries which allow researchers with some background to run experiments focused on QEC, there is a lack of a unified framework for general QEMDAC which is accessible to quantum computing researchers of any background and simultaneously supports diverse algorithms.

Many of the necessary components for quantum error control have been formalized mathematically such that algorithms can be designed to construct these components for general classes of error control techniques.

`LogicalQ` was designed to accelerate the quantum error correction workflow.
The combination of generalized quantum error correction functionality, compatibility with libraries such as Qiskit, existence of numerous demo notebooks, and overall usability will increase accessibility to quantum error corrected-research and enable deeper study into the application of quantum error correction.

Furthermore, QEMDAC techniques can make analysis of quantum computation results difficult because they utilize overhead resources which exponentially increase the size of experiment outputs. There is a need for tools which can parse QEMDAC results without requiring researchers to understand the often-complex mathematics of these techniques.

Although many of the necessary tools are not particularly lengthy or convoluted in their implementation, `LogicalQ` provides a single toolkit which handles the complexities of the QEMDAC workflow to avoid user error when constructing circuit components or performing mathematical analyses.

# Functionality

`LogicalQ` consists of various modules which provide functionality for the construction of QEMDAC components as well as their application and analysis.

The `Logical` module lies at the heart of the library with the implementation of the `LogicalCircuit` class. This class inherits from the `QuantumCircuit` class in `Qiskit` and . This module also contains the implementations of the `LogicalStatevector` and `LogicalDensityMatrix` classes (which inherit from the `Statevector` and `DensityMatrix` classes in `Qiskit`, respectively), which enable direct analysis of quantum states at either the logical level or physical level. Importantly, these classes allow for experimental measurement data to be represented using formal quantum state representations and subsequently visualized in terms of logical basis vectors and matrices, respectively. `Logical` also contains the `logical_state_fidelity` function which is designed to support complex fidelity comparisons, such as the fidelity of a physical density matrix and a logical statevector. Such a tool is essential for 

An important feature of the `LogicalCircuit` class is its `from_physical_circuit` class method, which allows users to input a `QuantumCircuit` object from `Qiskit` and output a `LogicalCircuit` which has all of its QEMDAC functionality. This interoperability means that researchers that already use `Qiskit` (or, in fact, any tool which can export `OpenQASM` code, which can then be imported into a `QuantumCircuit`) can readily apply QEMDAC techniques to their work. Because of the inheritance structure, most of the familiar class attributes and methods are available `LogicalCircuit`, including many which have been overrided with special behavior for QEMDAC. This includes logical realizations of common quantum gates, many of which can be realized using different methods.

The `Benchmarks` module contains constructors for many of the most commonly-used benchmarking circuits in quantum computation, including randomized benchmarking and quantum volume. These functions expose parameters such as qubit counts, circuit lengths, and random selection seeds to the user so that they can be directly integrated into controlled tests and experiments.

The `Experiments` module contains a variety of experiments which can be used to study QEMDAC techniques. It also contains the `execute_circuits` method, which provides a unified interface with both simulator and hardware backends. Experiment data can be analyzed with functions from the `Analysis` module, as long as they conform to the expected data structures for each analysis function.

# Acknowledgements

We acknowledge contributions from the Quantum Computing Association at Georgia Tech.

We would like to thank Jeff Young for serving as the advisor of the Quantum Computing Association at Georgia Tech and this project.

# References


