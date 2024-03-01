import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import IfElseOp

__all__ = [
    "prep_zero",
    "qec_cycle",
    "decoder_2d",
    "decoder_flag_update"
]



def prep_zero():
    """
    Prepare the logical |0> state
    """
    qreg = QuantumRegister(8, "q")
    creg = ClassicalRegister(1, "c")

    quantinuum_circuit = QuantumCircuit(qreg, creg)
    quantinuum_circuit.h(qreg[0])
    quantinuum_circuit.h(qreg[4])
    quantinuum_circuit.h(qreg[6])
    quantinuum_circuit.cx(qreg[0], qreg[1])
    quantinuum_circuit.cx(qreg[4], qreg[5])
    quantinuum_circuit.cx(qreg[6], qreg[3])
    quantinuum_circuit.cx(qreg[4], qreg[2])
    quantinuum_circuit.cx(qreg[6], qreg[5])
    quantinuum_circuit.cx(qreg[0], qreg[3])
    quantinuum_circuit.cx(qreg[4], qreg[1])
    quantinuum_circuit.cx(qreg[3], qreg[2])

    # Parity check z1z3z5
    quantinuum_circuit.cx(qreg[1], qreg[7])
    quantinuum_circuit.cx(qreg[3], qreg[7])
    quantinuum_circuit.cx(qreg[5], qreg[7])
    encode0LGate = quantinuum_circuit.to_instruction(label='encoding circuit')
    
    # Repeat multiple times
    qreg = QuantumRegister(8, "q")
    creg = ClassicalRegister(1, "c")

    FTquantinuum_circuit = QuantumCircuit(qreg, creg)
    FTquantinuum_circuit.append(encode0LGate, qreg, creg)
    FTquantinuum_circuit.measure(7,0)

    with FTquantinuum_circuit.if_test((creg, 1)):
        FTquantinuum_circuit.reset(qreg[:])
        FTquantinuum_circuit.append(encode0LGate, qreg, creg)
        FTquantinuum_circuit.measure(7,0)

    with FTquantinuum_circuit.if_test((creg, 1)):
        FTquantinuum_circuit.reset(qreg[:])
        FTquantinuum_circuit.append(encode0LGate, qreg, creg)
        FTquantinuum_circuit.measure(7,0)

    with FTquantinuum_circuit.if_test((creg, 1)):
        FTquantinuum_circuit.reset(qreg[:])
        FTquantinuum_circuit.append(encode0LGate, qreg, creg)
        FTquantinuum_circuit.measure(7,0) 
        
    return  FTquantinuum_circuit
    

def decoder_2d(syndrome_diff):
    """
    A 2D decoder that determines logical corrections. Due to symmetry, can be 
    used to decode stabilizer measurements of any Pauli type.
    
    Args:
        syndrom_diff: The change in syndromes compared to the last time.
    
    Returns:
        A bit representing whether a logical error has occurred.
    """

    bad_syndromes = {[0, 1, 0], [0, 1, 1], [0, 0, 1]}
    
    if syndrome_diff in bad_syndromes:
        logical_error = 1
    else:
        logical_error = 0
        
    return logical_error


def decoder_flag_update(syndrome_diff, flag_diff):
    """
    A lookup decoder used to modify corrections due hook errors indicated by changes 
    in flag and sydrome. 
    Args:
        syndrome diff: The change in syndromes compared to the last time.
        flag_diff: The change in flags compared to the last time.
    Returns:
        A bit representing whether the 2D decoder's correction should be changed due 
        to relationship between flags and syndromes.
    """
    # The following indicate hook faults have occured:
    
    # flag -> syndrome: 0 -> 1
    if flag_diff == [1, 0, 0] and syndrome_diff == [0, 1, 0]: 
        change_correction = 1
        
    # flag -> syndrome: 0 -> 2
    elif flag_diff == [1, 0, 0] and syndrome_diff == [0, 0, 1]: 
        change_correction = 1
        
   # flag -> syndrome: 1,2 -> 2
    elif flag_diff == [0, 1, 1] and syndrome_diff == [0, 0, 1]: 
        change_correction = 1
        
    else:
        change_correction = 0
    
    return change_correction



