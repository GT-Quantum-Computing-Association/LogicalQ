from LogicalQ.Logical import LogicalCircuit
from LogicalQ.Library.QECCs import stabilizer_codes

from qiskit.quantum_info import partial_trace, DensityMatrix

# @TODO - find expected results in the form of statevectors, density matrices, etc.
        
def TestX():
    for stabilizer_code in stabilizer_codes:
        n, k, d = stabilizer_code["label"]
        
        lqc_x = LogicalCircuit(**stabilizer_code)
        
        targets = list(range(k))
        lqc_x.x(targets)
        
        qc = lqc_x.circuit
        
        simulator = AerSimulator()
        tqc = transpile(qc, simulator)
        result = simulator.run(tqc).result()
        final_dm = result.data(0)["density_matrix"]
        
        rho = DensityMatrix(final_dm)
        reduced = partial_trace(rho, list(range(k, n)))
        print(reduced.draw(output="text"))

def TestY():
    for stabilizer_code in stabilizer_codes:
        n, k, d = stabilizer_code["label"]
        
        lqc_y = LogicalCircuit(**stabilizer_code)
        
        targets = list(range(k))
        lqc_y.y(targets)
        
        lqc_y.measure_all()

def TestZ():
    for stabilizer_code in stabilizer_codes:
        n, k, d = stabilizer_code["label"]
        
        lqc_z = LogicalCircuit(**stabilizer_code)
        
        targets = list(range(k))
        lqc_z.z(targets)
        
        lqc_z.measure_all()

def TestH():
    for stabilizer_code in stabilizer_codes:
        n, k, d = stabilizer_code["label"]
        
        lqc_h = LogicalCircuit(**stabilizer_code)
        
        targets = list(range(k))
        lqc_h.h(targets)
        
        lqc_h.measure_all()

def TestS():
    for stabilizer_code in stabilizer_codes:
        n, k, d = stabilizer_code["label"]
        
        lqc_s = LogicalCircuit(**stabilizer_code)
        
        targets = list(range(k))
        lqc_s.s(targets)
        
        lqc_s.measure_all()

def TestT():
    for stabilizer_code in stabilizer_codes:
        n, k, d = stabilizer_code["label"]
        
        lqc_t = LogicalCircuit(**stabilizer_code)
        
        targets = list(range(k))
        lqc_t.t(targets)
        
        lqc_t.measure_all()

def TestCX():
    for stabilizer_code in stabilizer_codes:
        n, k, d = stabilizer_code["label"]
        
        lqc_cx = LogicalCircuit(stabilizer_tableau["tableau"])
        
        control = 0
        targets = list(range(1, k)) 
        lqc_cx.cx(control, *targets)
        
        lqc_cx.measure_all()  

def TestPauliOperations():
    TestX()
    TestY()
    TestZ()

def TestCliffordGates():
    TestPauliGates()
    TestH()
    TestS()
    TestCX()
