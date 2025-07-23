from LogicalQ.Logical import LogicalCircuit
from LogicalQ.Library.QECCs import stabilizer_codes

from qiskit.quantum_info import partial_trace, DensityMatrix

# @TODO - find expected results in the form of statevectors, density matrices, etc.

def TestEncode():
    for stabilizer_code in stabilizer_codes:
        n, k, d = stabilizer_code["label"]
        
        lqc_encode = LogicalCircuit(**stabilizer_code)
        
        q_indices = list(range(k))
        lqc_encode.encode(q_indices)
        
        lqc_encode.measure_all()

def TestUnflaggedSyndromeMeasurement():
    raise NotImplementedError()

def TestDecoding():
    raise NotImplementedError()

def TestQECCycle():
    TestDecoding()
    TestUnflaggedSyndromeMeasurement()
