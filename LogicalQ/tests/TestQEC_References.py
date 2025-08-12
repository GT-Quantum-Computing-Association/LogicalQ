from qiskit import QuantumCircuit

def TestEncode_Reference(qecc):
    if qecc["label"] == (5,1,3):
        qc_encode = QuantumCircuit(5)

        qc_encode.h(0)
        qc_encode.h(1)
        qc_encode.h(2)
        qc_encode.h(3)
        qc_encode.z(4)

        qc_encode.cx(0, 4)
        qc_encode.cx(1, 4)
        qc_encode.cx(2, 4)
        qc_encode.cx(3, 4)
        qc_encode.cz(0, 4)
        qc_encode.cz(0, 1)
        qc_encode.cz(2, 3)
        qc_encode.cz(1, 2)
        qc_encode.cz(3, 4)
    elif qecc["label"] == (7,1,3):
        qc_encode = QuantumCircuit(7)
        qc_encode.h(0)
        qc_encode.h(1)
        qc_encode.h(3)
        qc_encode.cx(0, 2)
        qc_encode.cx(3, 5)
        qc_encode.cx(1, 6)
        qc_encode.cx(0, 4)
        qc_encode.cx(3, 6)
        qc_encode.cx(1, 5)
        qc_encode.cx(0, 6)
        qc_encode.cx(1, 2)
        qc_encode.cx(3, 4)
    else:
        print(f"WARNING - Not Implemented: TestEncode has no reference for the {qecc} stabilizer code")

    return qc_encode

