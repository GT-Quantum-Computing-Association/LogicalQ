import numpy as np

from LogicalQ.Logical import LogicalCircuit
from LogicalQ.Logical import LogicalStatevector, logical_state_fidelity

from qiskit.quantum_info import Statevector
from qiskit.circuit.library import (
    XGate, YGate, ZGate, HGate, SGate, SdgGate, TGate, TdgGate,
    RXGate, RYGate, RZGate, RXXGate, RYYGate, RZZGate, CXGate,
)
from qiskit import transpile
from qiskit.transpiler import PassManager

from qiskit_aer import AerSimulator

from LogicalQ.Transpilation.ClearQEC import ClearQEC
from LogicalQ.Transpilation.UnBox import UnBox

from LogicalQ.Library.QECCs import implemented_codes

from LogicalQ.tests import FIDELITY_ATOL, sk_rotation_atol

# @TODO - find expected results in the form of statevectors, density matrices, etc.

def TestSingleQubitGate(gate_name, lqc_method, reference_gate, qeccs=None, init_states=None, **gate_kwargs):
    if qeccs is None:
        qeccs = implemented_codes

    if init_states is None:
        init_states = [0, 1]

    all_successful = True
    for qecc in qeccs:
        # @TODO - LogicalStatevector doesn't recognize LogicalCircuitGeneral, so we can't do other codes for this test
        if qecc["label"] != (7,1,3):
            print(f"WARNING - Test{gate_name} does not fully work for non-Steane codes, skipping")
            continue

        for init_state in init_states:
            n, k, d = qecc["label"]

            lqc = LogicalCircuit(k, **qecc)
            lqc.encode(list(range(k)), max_iterations=0, initial_states=[init_state]*k)
            lqc_method(lqc, list(range(k)), **gate_kwargs)

            try:
                lsv = LogicalStatevector(lqc)
            except Exception as e:
                print(f"Test{gate_name} failed for {qecc['label']} and initial state |{init_state}>: LogicalStatevector construction error: {e}")
                all_successful = False
                continue

            sv_expected = Statevector.from_int(init_state, dims=2**k).evolve(reference_gate)
            fidelity = logical_state_fidelity(lsv, sv_expected)

            if np.isclose(fidelity, 1.0, atol=FIDELITY_ATOL):
                print(f"Test{gate_name} succeeded for {qecc['label']} and initial state |{init_state}> with fidelity {fidelity}")
            else:
                print(f"Test{gate_name} failed for {qecc['label']} and initial state |{init_state}> with fidelity {fidelity}")
                all_successful = False

    return all_successful

def TestSingleQubitRotationGate(gate_name, lqc_method, reference_gate_factory, qeccs=None, thetas=None, recursion_degree=1, depth=10, atol=None):
    if qeccs is None:
        qeccs = implemented_codes

    if thetas is None:
        thetas = list(np.linspace(0, 2*np.pi, 5, endpoint=False))

    if atol is None:
        atol = sk_rotation_atol(recursion_degree=recursion_degree, depth=depth, thetas=thetas)

    all_successful = True
    for qecc in qeccs:
        # @TODO - LogicalStatevector doesn't recognize LogicalCircuitGeneral, so we can't do other codes for this test
        if qecc["label"] != (7,1,3):
            print(f"WARNING - Test{gate_name} does not fully work for non-Steane codes, skipping")
            continue

        for theta in thetas:
            n, k, d = qecc["label"]

            lqc = LogicalCircuit(k, **qecc)
            lqc.encode(list(range(k)), max_iterations=0, initial_states=[0]*k)
            lqc_method(lqc, theta, list(range(k)))

            try:
                lsv = LogicalStatevector(lqc)
            except Exception as e:
                print(f"Test{gate_name} failed for {qecc['label']} and theta={theta:.4f}: LogicalStatevector construction error: {e}")
                all_successful = False
                continue

            sv_expected = Statevector.from_int(0, dims=2**k).evolve(reference_gate_factory(theta))
            fidelity = logical_state_fidelity(lsv, sv_expected)

            if np.isclose(fidelity, 1.0, atol=atol):
                print(f"Test{gate_name} succeeded for {qecc['label']} and theta={theta:.4f} with fidelity {fidelity}")
            else:
                print(f"Test{gate_name} failed for {qecc['label']} and theta={theta:.4f} with fidelity {fidelity}")
                all_successful = False

    return all_successful

def TestMultiQubitGateConstruction(gate_name, lqc_method, n_logical_qubits, qeccs=None, **gate_kwargs):
    if qeccs is None:
        qeccs = implemented_codes

    all_successful = True
    for qecc in qeccs:
        n, k, d = qecc["label"]

        try:
            lqc = LogicalCircuit(n_logical_qubits, **qecc)
            lqc.encode(list(range(n_logical_qubits)), max_iterations=0)

            lqc_method(lqc)

            pm_unbox = PassManager([ClearQEC(), UnBox()])
            while "box" in lqc.count_ops():
                lqc = pm_unbox.run(lqc)

            simulator = AerSimulator()
            lqc_transpiled = transpile(lqc, simulator)

            if lqc_transpiled.num_qubits <= 0 or len(lqc_transpiled.data) == 0:
                raise RuntimeError("transpiled circuit is empty")
        except Exception as e:
            print(f"Test{gate_name} failed for {qecc['label']}: {e}")
            all_successful = False
            continue

        print(f"Test{gate_name} succeeded for {qecc['label']}")

    return all_successful

def TestX(qeccs=None):
    return TestSingleQubitGate(
        "X",
        lambda lqc, targets: lqc.x(targets),
        XGate(),
        qeccs=qeccs,
    )

def TestY(qeccs=None):
    return TestSingleQubitGate(
        "Y",
        lambda lqc, targets: lqc.y(targets),
        YGate(),
        qeccs=qeccs,
    )

def TestZ(qeccs=None):
    return TestSingleQubitGate(
        "Z",
        lambda lqc, targets: lqc.z(targets),
        ZGate(),
        qeccs=qeccs,
    )

def TestH(qeccs=None):
    return TestSingleQubitGate(
        "H",
        lambda lqc, targets: lqc.h(targets),
        HGate(),
        qeccs=qeccs,
    )

def TestS(qeccs=None):
    return TestSingleQubitGate(
        "S",
        lambda lqc, targets: lqc.s(targets),
        SGate(),
        qeccs=qeccs,
    )

def TestSdg(qeccs=None):
    return TestSingleQubitGate(
        "Sdg",
        lambda lqc, targets: lqc.sdg(targets),
        SdgGate(),
        qeccs=qeccs,
    )

def TestT(qeccs=None):
    return TestSingleQubitGate(
        "T",
        lambda lqc, targets: lqc.t(targets),
        TGate(),
        qeccs=qeccs,
    )

def TestTdg(qeccs=None):
    return TestSingleQubitGate(
        "Tdg",
        lambda lqc, targets: lqc.tdg(targets),
        TdgGate(),
        qeccs=qeccs,
    )

def TestCX(qeccs=None):
    return TestMultiQubitGateConstruction(
        "CX",
        lambda lqc: lqc.cx(0, 1),
        n_logical_qubits=2,
        qeccs=qeccs,
    )

def TestRX(qeccs=None, thetas=None):
    return TestSingleQubitRotationGate(
        "RX",
        lambda lqc, theta, targets: lqc.rx(theta, targets),
        lambda theta: RXGate(theta),
        qeccs=qeccs,
        thetas=thetas,
    )

def TestRY(qeccs=None, thetas=None):
    return TestSingleQubitRotationGate(
        "RY",
        lambda lqc, theta, targets: lqc.ry(theta, targets),
        lambda theta: RYGate(theta),
        qeccs=qeccs,
        thetas=thetas,
    )

def TestRZ(qeccs=None, thetas=None):
    return TestSingleQubitRotationGate(
        "RZ",
        lambda lqc, theta, targets: lqc.rz(theta, targets),
        lambda theta: RZGate(theta),
        qeccs=qeccs,
        thetas=thetas,
    )

def TestRXX(qeccs=None):
    return TestMultiQubitGateConstruction(
        "RXX",
        lambda lqc: lqc.rxx(np.pi/4, [0, 1]),
        n_logical_qubits=2,
        qeccs=qeccs,
    )

def TestRYY(qeccs=None):
    return TestMultiQubitGateConstruction(
        "RYY",
        lambda lqc: lqc.ryy(np.pi/4, [0, 1]),
        n_logical_qubits=2,
        qeccs=qeccs,
    )

def TestRZZ(qeccs=None):
    return TestMultiQubitGateConstruction(
        "RZZ",
        lambda lqc: lqc.rzz(np.pi/4, [0, 1]),
        n_logical_qubits=2,
        qeccs=qeccs,
    )

def TestRotationGates(qeccs=None):
    if all([
        TestRX(qeccs),
        TestRY(qeccs),
        TestRZ(qeccs),
        TestRXX(qeccs),
        TestRYY(qeccs),
        TestRZZ(qeccs),
    ]):
        print(f"TestRotationGates succeeded")
        return True
    else:
        print(f"TestRotationGates failed")
        return False

def TestPauliGates(qeccs=None):
    if all([
        TestX(qeccs),
        TestY(qeccs),
        TestZ(qeccs),
    ]):
        print(f"TestPauliGates succeeded")
        return True
    else:
        print(f"TestPauliGates failed")
        return False

def TestCliffordGates(qeccs=None):
    if all([
        TestPauliGates(qeccs),
        TestH(qeccs),
        TestS(qeccs),
        TestSdg(qeccs),
        TestCX(qeccs),
    ]):
        print(f"TestCliffordGates succeeded")
        return True
    else:
        print(f"TestCliffordGates failed")
        return False

def TestNonCliffordGates(qeccs=None):
    if all([
        TestT(qeccs),
        TestTdg(qeccs),
    ]):
        print(f"TestNonCliffordGates succeeded")
        return True
    else:
        print(f"TestNonCliffordGates failed")
        return False

def TestAllGates(qeccs=None):
    if all([
        TestCliffordGates(qeccs),
        TestNonCliffordGates(qeccs),
        TestRotationGates(qeccs),
    ]):
        print(f"TestAllGates succeeded")
        return True
    else:
        print(f"TestAllGates failed")
        return False

