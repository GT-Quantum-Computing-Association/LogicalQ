import numpy as np

from LogicalQ.Logical import LogicalCircuit
from LogicalQ.Logical import LogicalStatevector, LogicalDensityMatrix

from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace, state_fidelity

from LogicalQ.Library.QECCs import implemented_codes, steane_code

# @TODO - find expected results in the form of statevectors, density matrices, etc.

def TestStatevectorFromLogicalCircuit(qeccs=None, states=None):
    if qeccs is None:
        qeccs = implemented_codes

    if states is None:
        states = ["0", "1", "+"]

    all_successful = True
    for qecc in qeccs:
        # @TODO - LogicalStatevector doesn't recognize LogicalCircuitGeneral, so we can't do other codes for this test
        if qecc["label"] != (7,1,3):
            print(f"WARNING - TestStatevectorFromLogicalCircuit does not fully work for non-Steane codes, skipping")
            continue

        for state in states:
            n, k, d = qecc["label"]

            lqc = LogicalCircuit(k, **qecc)

            q_indices = list(range(k))
            lqc.encode(q_indices, max_iterations=0)

            # @TODO - find a better way to do this for codes with k > 1
            if state == "0":
                lqc.id(list(range(k)))
            elif state == "1":
                lqc.x(list(range(k)))
            elif state == "+":
                lqc.h(list(range(k)))
            else:
                raise ValueError(f"Input '{state}' is not a valid state for state preparation for TestStatevectorFromLogicalCircuit.")

            lsv = LogicalStatevector(lqc)

            if state == "0":
                fidelity = lsv.logical_decomposition[0]**2
            elif state == "1":
                fidelity = lsv.logical_decomposition[1]**2
            elif state == "+":
                fidelity = 0.5 * (lsv.logical_decomposition[0] + lsv.logical_decomposition[1])**2
            else:
                raise ValueError(f"Input '{state}' is not a valid state for state preparation for TestStatevectorFromLogicalCircuit.")

            fidelity = np.real(fidelity)

            if np.isclose(fidelity, 1.0):
                print(f"TestStatevectorFromLogicalCircuit succeeded for {qecc['label']} and state {state} with fidelity {fidelity}")
            else:
                print(f"TestStatevectorFromLogicalCircuit failed for {qecc['label']} and state {state} with fidelity {fidelity}")
                all_successful = False

    return all_successful

def TestStatevector(qeccs=None, states=None):
    if all([
        TestStatevectorFromLogicalCircuit(qeccs, states),
    ]):
        print(f"TestStatevector succeeded")
        return True
    else:
        print(f"TestStatevector failed")
        return False


def TestDensityMatrix(qeccs=None, states=None):
    print(f"WARNING - TestDensityMatrix has not been fully implemented, returning True")
    return True

def TestAllStateRepresentations(qeccs=None):
    if all([
        TestStatevector(qeccs),
        TestDensityMatrix(qeccs),
    ]):
        print(f"TestAllStateRepresentations succeeded")
        return True
    else:
        print(f"TestAllStateRepresentations failed")
        return False

