import numpy as np

# from LogicalQ.Logical import LogicalCircuit
from LogicalQ.LogicalGeneral import LogicalCircuitGeneral as LogicalCircuit

from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace, state_fidelity

from LogicalQ.Library.QECCs import implemented_codes

from LogicalQ.tests.TestQEC_References import TestEncode_Reference

from qiskit import transpile
from qiskit.transpiler import PassManager
from LogicalQ.Transpilation.ClearQEC import ClearQEC
from LogicalQ.Transpilation.UnBox import UnBox

from qiskit_aer import AerSimulator

# @TODO - find expected results in the form of statevectors, density matrices, etc.

def TestEncode(qeccs=None):
    if qeccs is None:
        qeccs = implemented_codes

    all_successful = True
    for qecc in qeccs:
        # LogicalQ
        n, k, d = qecc["label"]

        lqc_encode = LogicalCircuit(k, **qecc)

        q_indices = list(range(k))
        lqc_encode.encode(q_indices, max_iterations=0)

        # pm_unbox = PassManager([ClearQEC(), UnBox()])
        # while "box" in lqc_encode.count_ops():
        #     lqc_encode = pm_unbox.run(lqc_encode)
        # lqc_encode = lqc_encode.remove_final_measurements()
        # dm_encode_logicalq = DensityMatrix(lqc_encode)

        # lqc_encode.save_density_matrix()
        lqc_encode.save_statevector()
        backend = AerSimulator()
        lqc_encode_transpiled = transpile(lqc_encode, backend)
        # result = backend.run(lqc_encode_transpiled, method="density_matrix", shots=1E2).result()
        result = backend.run(lqc_encode_transpiled, method="statevector", shots=1E2).result()
        # dm_encode_logicalq = result.data()["density_matrix"]
        dm_encode_logicalq = result.data()["statevector"]

        non_data_qubits = list(range(n, lqc_encode.num_qubits))
        dm_encode_logicalq = partial_trace(dm_encode_logicalq, non_data_qubits)

        # Reference
        qc_encode_reference = TestEncode_Reference(qecc)
        dm_encode_reference = DensityMatrix(qc_encode_reference)
        # dm_encode_reference = Statevector(qc_encode_reference)

        # Fidelity comparison
        fidelity = state_fidelity(dm_encode_logicalq, dm_encode_reference)

        if np.isclose(fidelity, 1.0):
            print(f"TestEncode succeeded for {qecc['label']} with fidelity {fidelity}")
        else:
            print(f"TestEncode failed for {qecc['label']} with fidelity {fidelity}")
            all_successful = False

    return all_successful

def TestUnflaggedSyndromeMeasurement(qeccs=None):
    print(f"WARNING - TestUnflaggedSyndromeMeasurement has not been fully implemented, returning True")
    return True

def TestFlaggedSyndromeMeasurement(qeccs=None):
    print(f"WARNING - TestFlaggedSyndromeMeasurement has not been fully implemented, returning True")
    return True

def TestDecoding(qeccs=None):
    print(f"WARNING - TestDecoding has not been fully implemented, returning True")
    return True

def TestQECCycle(qeccs=None):
    if all([
        TestUnflaggedSyndromeMeasurement(qeccs),
        TestFlaggedSyndromeMeasurement(qeccs),
        TestDecoding(qeccs),
    ]):
        print(f"TestQECCycle succeeded")
        return True
    else:
        print(f"TestQECCycle failed")
        return False

def TestAllQEC(qeccs=None):
    if all([
        TestEncode(qeccs),
        TestQECCycle(qeccs),
    ]):
        print(f"TestAllQEC succeeded")
        return True
    else:
        print(f"TestAllQEC failed")
        return False

