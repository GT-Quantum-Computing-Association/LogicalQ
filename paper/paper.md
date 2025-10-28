---
title: "LogicalQ: A toolkit for quantum circuit development with generalized quantum error mitigation, detection, and correction"
tags:
  - Python
  - quantum
  - quantum information
  - quantum computing
  - quantum computation
  - quantum computer
  - quantum algorithms
  - quantum simulation
  - quantum error correction
  - quantum error-correcting code
  - quantum stabilizer code
  - css code
  - qecc
  - qiskit
  - optimization
  - physics
authors:
  - name: Rasmit Devkota
    orcid: 0009-0009-3294-638X
    affiliation: "1,2"
  - name: Ben Hagan
    affiliation: 3
  - name: Nolan Heffner
    orcid: 0009-0005-6311-6013
    affiliation: "1,2"
  - name: Younos Hashem
    orcid: 0009-0005-6704-6745
    affiliation: 1
  - name: Noah Song
    orcid: 0009-0008-5561-6853
    affiliation: 2
  - name: Arhan Deshpande
    affiliation: 1
  - name: Fiyin Makinde
    affiliation: 4
  - name: Richard Yu
    affiliation: "2,5"
  - name: Alisa Petrusinskaia
    orcid: 0009-0000-8913-0650
    affiliation: 5
  - name: Zhixin Song
    affiliation: 1
  - name: David Lloyd George
    affiliation: 6
  - name: Lance Lampert
    affiliation: 6
affiliations:
 - name: School of Physics, College of Sciences, Georgia Institute of Technology, Atlanta, GA
   index: 1
 - name: School of Mathematics, College of Science, Georgia Institute of Technology, Atlanta, GA
   index: 2
 - name: Ming Hsieh Department of Electrical and Computer Engineering, Viterbi School of Engineering, University of Southern California, Los Angeles, CA
   index: 3
 - name: School of Electrical and Computer Engineering, College of Engineering, Georgia Institute of Technology, Atlanta, GA
   index: 4
 - name: School of Computer Science, College of Computing, Georgia Institute of Technology, Atlanta, GA
   index: 5
 - name: Department of Physics, Duke University, Durham, NC
   index: 6
date: 26 October 2025
bibliography: paper.bib
---

# Summary

`LogicalQ` is a Python toolkit for quantum circuit development with generalized quantum error mitigation, detection, and correction (QEMDAC). `LogicalQ` inherits many of its data structures from, and thus is designed to interface well with, the `Qiskit` and `Qiskit Aer` packages [@Javadi-Abhari2024].

The source code for `LogicalQ` is available on [GitHub](https://github.com/GT-Quantum-Computing-Association/LogicalQ/). It can be installed via `pip` from the [`pypi` index](https://pypi.org/project/LogicalQ/). Its [documentation](https://logicalq.readthedocs.io/) is hosted publicly.

# Statement of need

Quantum computing presents a new model for computation which may significantly accelerate discovery in many fields, from physics [@Feynman1982] to cryptography [@Shor1994] to economics [@Herman2023]. However, the current era of quantum hardware is still noisy, and quantum error mitigation (QEM) [@Viola1998], quantum error detection (QED) [@Leung1999], and quantum error correction (QEC) [@Shor1995] techniques are necessary for the execution of utility-scale algorithms with any reasonable fidelity. Although there exist a number of libraries which allow researchers with some background to run experiments focused on QEC, there is no unified framework for general QEMDAC which is accessible to quantum computing researchers of any background and simultaneously supports the convenient implementation of quantum algorithms.

Many of the necessary components for QEMDAC have been formalized mathematically such that algorithms can be designed to construct these components for general classes of error control techniques [@Gottesman1997]. `LogicalQ`, like many existing QEMDAC libraries, uses such generalized constructions to meet any use case and application.

A comparison of existing libraries is made in Table 1. We choose to compare features which may be desirable to researchers in quantum algorithms. Note that we define external two-way interoperability to be with any external general-purpose quantum computing tool such as `QASM` or `Qiskit`, but not just another QEMDAC tool.

|Feature|`LogicalQ`|`stim`|`mqt-qecc`|`PEC0S`|`stac`|`tqec`|
|------------------------------------|-------------|------------|------------|------------|------------|------------|
|StabilizercodeQEC|$\checkmark$|$\checkmark$|$\checkmark$|$\checkmark$|$\checkmark$|$\checkmark$|
|qLDPC-orientedQEC|$\checkmark$|$\times$|$\checkmark$|$\times$|$\times$|$\times$|
|ArbitrarylogicalCliffordgates|$\checkmark$|$\times$|$\times$|$\checkmark$|$\checkmark$|$\times$|
|Arbitrarylogicalnon-Cliffordgates|$\checkmark$|$\times$|$\times$|$\checkmark$|$\checkmark$|$\times$|
|Advanceddecoders|$\times$|$\checkmark$|$\checkmark$|$\checkmark$|$\times$|$\checkmark$|
|Arbitrarynoisemodelsupport|$\checkmark$|$\times$|$\times$|$\checkmark$|$\times$|$\checkmark$|
|OptimizedQECcyclescheduling|$\checkmark$|$\times$|$\times$|$\times$|$\times$|$\times$|
|Experimentsuite|$\checkmark$|$\times$|$\times$|$\checkmark$|$\times$|$\checkmark$|
|Logicalstateanalysis|$\checkmark$|$\times$|$\times$|$\times$|$\times$|$\checkmark$|
|Externaltwo-wayinteroperability|$\checkmark$|$\checkmark$|$\times$|$\times$|$\checkmark$|$\times$|
|Cloudhardwareinterfaces|$\checkmark$|$\times$|$\times$|$\times$|$\times$|$\times$|
Table: Comparison of `LogicalQ` with other major QEMDAC packages; `stim` is due to [@Gidney2021], `mqt-qecc` is due to [@Wille2024], `PECOS` is due to [@RyanAndersonPECOS], `stac` is due to [@Stac2024], and `tqec` is due to [@TQEC].

In summary, many of the existing libraries are notable for their high-performance simulations and advanced implementations of certain features, but none support the full functionality required for QEMDAC applied to quantum algorithms research, especially on cloud hardware. `LogicalQ` is also unique in that it has a suite of experiments for testing QEMDAC which serves as a quick set of tests for researchers studying noise control.

`LogicalQ` was designed to accelerate the application of QEMDAC in quantum algorithm development, so its core design principle is maximizing user capability for implementing complex quantum circuits and using QEMDAC. The combination of generalized quantum error correction functionality, compatibility with libraries such as Qiskit, existence of numerous demo notebooks, and overall usability will increase accessibility to quantum error-corrected research and enable deeper study into the application of quantum error correction.

Furthermore, QEMDAC techniques can make analysis of quantum computation results difficult because they utilize overhead resources which exponentially increase the size of experiment outputs. There is a need for tools which can parse QEMDAC results without requiring researchers to understand the often-complex mathematics of these techniques.

Although many of the necessary tools are not particularly lengthy or convoluted in their implementation, `LogicalQ` provides a single toolkit which handles the complexities of the QEMDAC workflow to avoid user error when constructing circuit components or performing mathematical analyses.

# Functionality

`LogicalQ` consists of various modules which provide functionality for the construction of QEMDAC components as well as their application and analysis.

A general flowchart of library structure is shown in Figure \ref{fig:flowchart}.

![LogicalQ Architecture\label{fig:flowchart}](./flowchart.svg){ width=100% }

The `Logical` module lies at the heart of the library with the `LogicalCircuit` class, which inherits from the `QuantumCircuit` class in `Qiskit` and extends it with a variety of QEMDAC features. A `LogicalCircuit` can be constructed from a `Qiskit` `QuantumCircuit` via the `from_physical_circuit` method, which enables easy integration of `LogicalQ` into existing workflows. The `optimize_qec_cycle_indices` method of `LogicalCircuit` performs cost accounting based on a constraint model and effective threshold in order to construct an optimal list of QEC cycle indices.

The `Logical` module also contains the `LogicalStatevector` and `LogicalDensityMatrix` classes, which inherit from the `Statevector` and `DensityMatrix` classes in `Qiskit` respectively and enable representation and analysis of quantum states at either the logical level or physical level. `Logical` also contains the `logical_state_fidelity` function which is designed to support mixed-type fidelity comparisons, such as the fidelity of a physical density matrix and a logical statevector.

The `Benchmarks` module contains constructors for many of the most commonly-used benchmarking circuits in quantum computation, including randomized benchmarking and quantum volume. These functions expose parameters such as qubit counts, circuit lengths, and random selection seeds to the user so that they can be directly integrated into controlled tests and experiments.

The `Experiments` module contains a variety of experiments which can be used to study QEMDAC techniques. Experiment data can be analyzed with functions from the `Analysis` module.

The `Execution` module contains the `execute_circuits` function, which provides a single interface for both simulator and hardware backends with smart handling of complex aspects such as backend communication, hardware models, and transpilation.

The `Estimators` module contains special experiments which are used in QED and QEC cycle scheduling. In particular, this includes effective threshold estimation and constraint model construction.

The `Library` modules contain utilties such as quantum codes for QED and QEC, hardware models for modelling quantum devices, special gates for benchmarking, and dynamical decoupling sequences for QEM.

# Scholarly Work

`LogicalQ` development has largely been driven by an ongoing research project to optimize the scheduling of QEMDAC components in quantum circuits, with the motivation of performing fault-tolerant Hamiltonian simulations of lattice gauge theories and other physical models on quantum hardware. This involves code switching between QEC and QED codes depending on the error-criticality of a part of a circuit, made less complex by `LogicalQ`'s generalized framework for stabilizer codes. There is also ongoing work on genetic algorithm-based optimization of physical and logical dynamical decoupling sequences for these applications and others in science.

# Acknowledgements

We acknowledge contributions from the Quantum Computing Association at Georgia Tech.

We would like to thank Jeff Young for serving as the advisor of the Quantum Computing Association at Georgia Tech and this project.

This research was supported in part through research cyberinfrastructure resources and services provided by the Partnership for an Advanced Computing Environment (PACE) at the Georgia Institute of Technology, Atlanta, Georgia, USA [@PACE].

# References



