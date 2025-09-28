# from LogicalQ.Logical import LogicalCircuit
import numpy as np
from LogicalQ.Experiments import execute_circuits
from LogicalQ.Logical import LogicalStatevector, logical_state_fidelity
from LogicalQ.LogicalGeneral import LogicalCircuitGeneral as LogicalCircuit
from LogicalQ.Library.QECCs import implemented_codes

from qiskit.quantum_info import partial_trace, DensityMatrix
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import PassManager
from LogicalQ.Transpilation.ClearQEC import ClearQEC
from LogicalQ.Transpilation.UnBox import UnBox
from LogicalQ.Library.HardwareModels import hardware_models_Quantinuum

from qiskit_aer import AerSimulator

# @TODO - find expected results in the form of statevectors, density matrices, etc.

def TestX(qeccs=None):
    if qeccs is None:
        qeccs = implemented_codes

    for qecc in qeccs:
        n, k, d = qecc["label"]

        lqc_x = LogicalCircuit(k, **qecc)

        targets = list(range(k))
        lqc_x.x(targets)

        simulator = AerSimulator()
        lqc_x_transpiled = transpile(lqc_x_transpiled, simulator)
        result = simulator.run(tqc).result()
        final_dm = result.data(0)["density_matrix"]

        rho = DensityMatrix(final_dm)
        reduced = partial_trace(rho, list(range(k, n)))

    print(f"WARNING - TestX has not been fully implemented, returning True")
    return True

def TestY(qeccs=None):
    if qeccs is None:
        qeccs = implemented_codes

    for qecc in qeccs:
        n, k, d = qecc["label"]

        lqc_y = LogicalCircuit(k, **qecc)

        targets = list(range(k))
        lqc_y.y(targets)

        lqc_y.measure_all()

    print(f"WARNING - TestY has not been fully implemented, returning True")
    return True

def TestZ(qeccs=None):
    if qeccs is None:
        qeccs = implemented_codes

    for qecc in qeccs:
        n, k, d = qecc["label"]

        lqc_z = LogicalCircuit(k, **qecc)

        targets = list(range(k))
        lqc_z.z(targets)

        lqc_z.measure_all()

    print(f"WARNING - TestZ has not been fully implemented, returning True")
    return True

def TestH(qeccs=None):
    if qeccs is None:
        qeccs = implemented_codes

    for qecc in qeccs:
        n, k, d = qecc["label"]

        lqc_h = LogicalCircuit(k, **qecc)

        targets = list(range(k))
        lqc_h.h(targets)

        lqc_h.measure_all()

    print(f"WARNING - TestH has not been fully implemented, returning True")
    return True

def TestS(qeccs=None):
    if qeccs is None:
        qeccs = implemented_codes

    for qecc in qeccs:
        n, k, d = qecc["label"]

        lqc_s = LogicalCircuit(k, **qecc)

        targets = list(range(k))
        lqc_s.s(targets)

        lqc_s.measure_all()

    print(f"WARNING - TestS has not been fully implemented, returning True")
    return True

def TestT(qeccs=None):
    if qeccs is None:
        qeccs = implemented_codes

    for qecc in qeccs:
        n, k, d = qecc["label"]

        lqc_t = LogicalCircuit(k, **qecc)

        targets = list(range(k))
        lqc_t.t(targets)

        lqc_t.measure_all()

    print(f"WARNING - TestT has not been fully implemented, returning True")
    return True

def TestCX(qeccs=None):
    if qeccs is None:
        qeccs = implemented_codes

    for qecc in qeccs:
        n, k, d = qecc["label"]

        lqc_cx = LogicalCircuit(k, **qecc)

        control = 0
        targets = list(range(1, k))
        lqc_cx.cx(control, *targets)

        lqc_cx.measure_all()

    print(f"WARNING - TestCX has not been fully implemented, returning True")
    return True

def TestRX(qeccs=None):
    if qeccs is None:
        qeccs = implemented_codes

    for qecc in qeccs:
        n, k, d = qecc["label"]

        for theta in np.linspace(0, 2*np.pi, 1024):
            lqc_rx = LogicalCircuit(k, **qecc)

            targets = list(range(k))

            lqc_rx.rx(theta, 0)

            simulator = AerSimulator()
            lqc_rx_transpiled = transpile(lqc_rx, simulator)
            result = simulator.run(lqc_rx_transpiled).result()
            final_dm = result.data(0)["density_matrix"]

            rho = DensityMatrix(final_dm)
            reduced = partial_trace(rho, list(range(k, n)))

    print(f"WARNING - TestX has not been fully implemented, returning True")
    return True

def TestRY(qeccs=None):
    print(f"WARNING - TestRY has not been fully implemented, returning True")
    return True

def TestRZ(qeccs=None):
    print(f"WARNING - TestRZ has not been fully implemented, returning True")
    return True

def TestRXX(qeccs=None):
    print(f"WARNING - TestRXX has not been fully implemented, returning True")
    return True

def TestRYY(qeccs=None):
    print(f"WARNING - TestRYY has not been fully implemented, returning True")
    return True

def TestRZZ(qeccs=None):
    print(f"WARNING - TestRZZ has not been fully implemented, returning True")
    return True

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
    ]):
        print(f"TestNonCliffordGates succeeded")
        return True
    else:
        print(f"TestNonCliffordGates failed")
        return False

def TestAllGates(qeccs=None):
    print(f"WARNING - TestAllGates has not been fully implemented, returning True")
    return True

    if all([
        TestCliffordGates(qeccs),
        TestNonCliffordGates(qeccs),
    ]):
        print(f"TestAllGates succeeded")
        return True
    else:
        print(f"TestAllGates failed")
        return False

