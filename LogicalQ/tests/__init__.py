import inspect

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import RXGate, RYGate, RZGate
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit.transpiler.passes import SolovayKitaev

FIDELITY_ATOL: float = inspect.signature(np.isclose).parameters["atol"].default


def sk_rotation_atol(
    recursion_degree: int = 1,
    depth: int = 10,
    basis_gates=None,
    thetas=None,
) -> float:
    if basis_gates is None:
        basis_gates = ["s", "sdg", "t", "tdg", "h", "x", "y", "z", "cz"]

    if thetas is None:
        thetas = list(np.linspace(0, 2*np.pi, 8, endpoint=False))

    sk = SolovayKitaev(recursion_degree=recursion_degree, basis_gates=basis_gates, depth=depth)

    worst_infidelity = 0.0
    for gate_cls in (RXGate, RYGate, RZGate):
        for theta in thetas:
            qc_ideal = QuantumCircuit(1)
            qc_ideal.append(gate_cls(theta), [0])

            qc_sk = sk(qc_ideal)

            sv_ideal = Statevector.from_int(0, dims=2).evolve(qc_ideal)
            sv_sk = Statevector.from_int(0, dims=2).evolve(qc_sk)

            infidelity = 1.0 - state_fidelity(sv_ideal, sv_sk, validate=False)
            worst_infidelity = max(worst_infidelity, infidelity)

    return float(worst_infidelity)
