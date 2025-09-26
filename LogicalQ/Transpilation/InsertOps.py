
from typing import Literal, get_args

from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit_aer.library.save_instructions import SaveStatevector

_TYPES = Literal["statevector", "density_matrix", "both"]

# @TODO - convert into a formal TransformationPass
def insert_before_measurement(logical_circuit, _type: _TYPES = "statevector"):
    """
    Traverses an original DAG, inserts a SaveStatevector instruction before
    "box" nodes with label "logical.qec.measure", and returns a new DAG.

    Args:
        logical_circuit (DAGCircuit): The DAG to be traversed.

    Returns:
        DAGCircuit: The new DAG with the instruction inserted.
    """
    
    options = get_args(_TYPES)
    if not (_type in options):
        raise AssertionError(f"'{_type}' is not in {options}")
    
    original_dag = circuit_to_dag(logical_circuit)
    
    new_dag = DAGCircuit()
    for qreg in original_dag.qregs.values():
        new_dag.add_qreg(qreg)
    for creg in original_dag.cregs.values():
        new_dag.add_creg(creg)

    insertion_complete = False

    for node in original_dag.topological_op_nodes():
        op_name = getattr(node.op, "name", None)
        op_label = getattr(node.op, "label", None)

        if (not insertion_complete and op_name == "box" and op_label and op_label.split(":")[0] == "logical.qec.measure"):
            qubits = []
            regs = list(new_dag.qregs.values())
            for reg in regs:
                qubits = qubits + [qubit for qubit in reg]
            
            if _type == "statevector" or _type == "both":
                save_inst = SaveStatevector(len(qubits), "statevector")
                new_dag.apply_operation_back(save_inst, qubits)

            if _type == "density_matrix" or _type == "both":
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

