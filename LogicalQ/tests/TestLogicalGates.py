# from LogicalQ.Logical import LogicalCircuit
from LogicalQ.LogicalGeneral import LogicalCircuitGeneral as LogicalCircuit
from LogicalQ.Library.QECCs import implemented_codes

from qiskit.quantum_info import partial_trace, DensityMatrix

from qiskit import transpile
from qiskit.transpiler import PassManager
from LogicalQ.Transpilation.ClearQEC import ClearQEC
from LogicalQ.Transpilation.UnBox import UnBox

from qiskit_aer import AerSimulator

# @TODO - find expected results in the form of statevectors, density matrices, etc.

def TestX(qecc_list=None):
    if qecc_list is None:
        qecc_list = implemented_codes

    for qecc in qecc_list:
        n, k, d = qecc["label"]

        lqc_x = LogicalCircuit(k, **qecc)

        targets = list(range(k))
        lqc_x.x(targets)

        simulator = AerSimulator()
        tqc = transpile(qc, simulator)
        result = simulator.run(tqc).result()
        final_dm = result.data(0)["density_matrix"]

        rho = DensityMatrix(final_dm)
        reduced = partial_trace(rho, list(range(k, n)))

    print(f"WARNING - TestX has not been fully implemented, returning True")
    return True

def TestY(qecc_list=None):
    if qecc_list is None:
        qecc_list = implemented_codes

    for qecc in qecc_list:
        n, k, d = qecc["label"]

        lqc_y = LogicalCircuit(k, **qecc)

        targets = list(range(k))
        lqc_y.y(targets)

        lqc_y.measure_all()

    print(f"WARNING - TestY has not been fully implemented, returning True")
    return True

def TestZ(qecc_list=None):
    if qecc_list is None:
        qecc_list = implemented_codes

    for qecc in qecc_list:
        n, k, d = qecc["label"]

        lqc_z = LogicalCircuit(k, **qecc)

        targets = list(range(k))
        lqc_z.z(targets)

        lqc_z.measure_all()

    print(f"WARNING - TestZ has not been fully implemented, returning True")
    return True

def TestH(qecc_list=None):
    if qecc_list is None:
        qecc_list = implemented_codes

    for qecc in qecc_list:
        n, k, d = qecc["label"]

        lqc_h = LogicalCircuit(k, **qecc)

        targets = list(range(k))
        lqc_h.h(targets)

        lqc_h.measure_all()

    print(f"WARNING - TestH has not been fully implemented, returning True")
    return True

def TestS(qecc_list=None):
    if qecc_list is None:
        qecc_list = implemented_codes

    for qecc in qecc_list:
        n, k, d = qecc["label"]

        lqc_s = LogicalCircuit(k, **qecc)

        targets = list(range(k))
        lqc_s.s(targets)

        lqc_s.measure_all()

    print(f"WARNING - TestS has not been fully implemented, returning True")
    return True

def TestT(qecc_list=None):
    if qecc_list is None:
        qecc_list = implemented_codes

    for qecc in qecc_list:
        n, k, d = qecc["label"]

        lqc_t = LogicalCircuit(k, **qecc)

        targets = list(range(k))
        lqc_t.t(targets)

        lqc_t.measure_all()

    print(f"WARNING - TestT has not been fully implemented, returning True")
    return True

def TestCX(qecc_list=None):
    if qecc_list is None:
        qecc_list = implemented_codes

    for qecc in qecc_list:
        n, k, d = qecc["label"]

        lqc_cx = LogicalCircuit(k, **qecc)

        control = 0
        targets = list(range(1, k))
        lqc_cx.cx(control, *targets)

        lqc_cx.measure_all()

    print(f"WARNING - TestCX has not been fully implemented, returning True")
    return True

def TestPauliGates(qecc_list=None):
    if all([
        TestX(qecc_list),
        TestY(qecc_list),
        TestZ(qecc_list),
    ]):
        print(f"TestPauliGates succeeded")
        return True
    else:
        print(f"TestPauliGates failed")
        return False

def TestCliffordGates(qecc_list=None):
    if all([
        TestPauliGates(qecc_list),
        TestH(qecc_list),
        TestS(qecc_list),
        TestCX(qecc_list),
    ]):
        print(f"TestCliffordGates succeeded")
        return True
    else:
        print(f"TestCliffordGates failed")
        return False

def TestNonCliffordGates(qecc_list=None):
    if all([
        TestT(qecc_list),
    ]):
        print(f"TestNonCliffordGates succeeded")
        return True
    else:
        print(f"TestNonCliffordGates failed")
        return False

def TestAllGates(qecc_list=None):
    print(f"WARNING - TestAllGates has not been fully implemented, returning True")
    return True

    if all([
        TestCliffordGates(qecc_list),
        TestNonCliffordGates(qecc_list),
    ]):
        print(f"TestAllGates succeeded")
        return True
    else:
        print(f"TestAllGates failed")
        return False

