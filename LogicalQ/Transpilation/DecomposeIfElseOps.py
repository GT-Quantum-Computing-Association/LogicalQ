from qiskit import QuantumCircuit
from qiskit.circuit import IfElseOp
from qiskit.circuit.classical.expr import Binary, Unary
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.converters import circuit_to_dag

class DecomposeIfElseOps(TransformationPass):
    """``DecomposeIfElseOps`` transpilation pass decomposes multi-classical bit IfElseOp's into nested single-classical bit IfElseOp's."""

    def __init__(self):
        super().__init__()

    @control_flow.trivial_recurse
    def run(self, dag):
        for if_else_op_node in dag.op_nodes(IfElseOp):
            if_else_op = if_else_op_node.op

            condition = if_else_op.condition

            if isinstance(condition, Binary):
                if_body, else_body = if_else_op.params

                decomposed_circuit = None

                if condition.op.name == "BIT_AND":
                    """
                    Decompose classical AND gate via truth table:
                    |-----------------|
                    | A | B | A AND B |
                    |-----------------|
                    | 0 | 0 |    0    |
                    | 0 | 1 |    0    |
                    | 1 | 0 |    0    |
                    | 1 | 1 |    1    |
                    |-----------------|
                    """

                    # If there is a BIT_NOT gate in a condition, then it will be a unary instead, which we have to handle differently
                    if isinstance(condition.left, Unary):
                        left_var = condition.left.operand.var
                        left_condition = 0 if condition.left.op.name == "BIT_NOT" else 1
                    else:
                        left_var = condition.left.var
                        left_condition = 1
                    if isinstance(condition.right, Unary):
                        right_var = condition.right.operand.var
                        right_condition = 0 if condition.right.op.name == "BIT_NOT" else 1
                    else:
                        right_var = condition.right.var
                        right_condition = 1

                    bits = list(set([*if_body.qubits, *else_body.qubits, *if_body.clbits, *else_body.clbits]))
                    decomposed_circuit = QuantumCircuit(bits, name="DecomposedClassicalXORCircuit")
                    with decomposed_circuit.if_test((left_var, left_condition)) as _else_left:
                        with decomposed_circuit.if_test((right_var, right_condition)) as _else_right:
                            decomposed_circuit.compose(if_body, if_body.qubits, if_body.clbits, inline_captures=True, inplace=True)
                        with _else_right:
                            decomposed_circuit.compose(else_body, else_body.qubits, else_body.clbits, inline_captures=True, inplace=True)
                    with _else_left:
                        decomposed_circuit.compose(else_body, else_body.qubits, else_body.clbits, inline_captures=True, inplace=True)
                elif condition.op.name == "BIT_XOR":
                    """
                    Decompose classical XOR gate via truth table:
                    |-----------------|
                    | A | B | A XOR B |
                    |-----------------|
                    | 0 | 0 |    0    |
                    | 0 | 1 |    1    |
                    | 1 | 0 |    1    |
                    | 1 | 1 |    0    |
                    |-----------------|
                    """

                    bits = list(set([*if_body.qubits, *else_body.qubits, *if_body.clbits, *else_body.clbits]))
                    decomposed_circuit = QuantumCircuit(bits, name="DecomposedClassicalXORCircuit")
                    with decomposed_circuit.if_test((condition.left.var, 0)) as _else_left:
                        with decomposed_circuit.if_test((condition.right.var, 0)) as _else_right:
                            decomposed_circuit.compose(else_body, else_body.qubits, else_body.clbits, inline_captures=True, inplace=True)
                        with _else_right:
                            decomposed_circuit.compose(if_body, if_body.qubits, if_body.clbits, inline_captures=True, inplace=True)
                    with _else_left:
                        with decomposed_circuit.if_test((condition.right.var, 0)) as _else_right:
                            decomposed_circuit.compose(if_body, if_body.qubits, if_body.clbits, inline_captures=True, inplace=True)
                        with _else_right:
                            decomposed_circuit.compose(else_body, else_body.qubits, else_body.clbits, inline_captures=True, inplace=True)
                else:
                    print(f"WARNING - DecomposeIfElseOps encountered IfElseOp with label '{if_else_op.label}' which has condition with name '{condition.op.name}', skipping.")

                if decomposed_circuit is not None:
                    decomposed_dag = circuit_to_dag(decomposed_circuit)
                    dag.substitute_node_with_dag(if_else_op_node, decomposed_dag)
            else:
                print(f"WARNING - DecomposeIfElseOps encountered IfElseOp with label '{if_else_op.label}' which has condition of unrecognized type {type(condition)}, skipping.")

        return dag

