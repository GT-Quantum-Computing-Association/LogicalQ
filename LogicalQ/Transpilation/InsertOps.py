
from typing import Literal, get_args

from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit_aer.library.save_instructions import SaveStatevector

_TYPES = Literal["statevector", "density_matrix", "both"]

def insert_before_measurement(logical_circuit, type_: _TYPES = "statevector"):
    """
    Traverses an original DAG, inserts a SaveStatevector instruction before
    a specific 'box' node, and returns a new DAG.

    Args:
        original_dag (DAGCircuit): The DAG to be traversed.

    Returns:
        DAGCircuit: The new DAG with the instruction inserted.
    """
    
    options = get_args(_TYPES)
    if not (type_ in options):
        raise AssertionError(f"'{type_}' is not in {options}")
    
    original_dag = circuit_to_dag(logical_circuit)
    
    new_dag = DAGCircuit()
    for qreg in original_dag.qregs.values():
        new_dag.add_qreg(qreg)
    for creg in original_dag.cregs.values():
        new_dag.add_creg(creg)

    insertion_complete = False

    for node in original_dag.topological_op_nodes():
        op_name = getattr(node.op, 'name', None)
        op_label = getattr(node.op, 'label', None)

        #print(f"Processing node: {op_name}, with label: {op_label}")

        if (not insertion_complete and
            op_name == "box" and
            op_label and op_label.split(":")[0] == "logical.qec.measure"):
            
            qubits = []
            regs = list(new_dag.qregs.values())
            for reg in regs:
                qubits = qubits + [qubit for qubit in reg]
            
            if type_ == "statevector" or type == "both":
                save_inst = SaveStatevector(len(qubits), "statevector")
                new_dag.apply_operation_back(save_inst, qubits)

            if type_ == "density_matrix" or type == "both":
                save_inst = SaveStatevector(len(qubits), "density_matrix")
                new_dag.apply_operation_back(save_inst, qubits)

            insertion_complete = True
        
        new_dag.apply_operation_back(
            node.op,
            qargs=node.qargs,
            cargs=node.cargs
        )
        
    if not insertion_complete:
        raise ValueError(f"No measurements found in circuit with name {logical_circuit.name}; cannot perform insertion.")
    
    new_logical_circuit = dag_to_circuit(new_dag)
    
    return new_logical_circuit, new_dag