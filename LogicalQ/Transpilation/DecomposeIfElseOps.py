from qiskit import QuantumCircuit
from qiskit.circuit import IfElseOp
from qiskit.circuit.classical.expr import Binary, Unary
from qiskit.transpiler import DoWhileController, TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.converters import circuit_to_dag

class DecomposeIfElseOps(TransformationPass):
    """``DecomposeIfElseOps`` transpilation pass decomposes multi-classical bit IfElseOp's into nested single-classical bit IfElseOp's."""

    def __init__(self, method=None):
        if method == None:
            self.method = "nested"
        else:
            self.method = method

        super().__init__()

    @control_flow.trivial_recurse
    def run(self, dag):
        self.property_set["decompose_if_else_ops_again"] = False

        if len(dag.op_nodes(IfElseOp)) == 0:
            return dag

        for if_else_op_node in dag.op_nodes(IfElseOp):
            if_else_op = if_else_op_node.op

            condition = if_else_op.condition

            if_body, else_body = if_else_op.params

            decomposed_circuit = None

            if isinstance(condition, Unary):
                var = condition.operand.var
                val = 0 if condition.op.name == "BIT_NOT" else 1
                condition = (var, val)

                if else_body is None:
                    bits = list(set([*if_body.qubits, *if_body.clbits]))
                else:
                    bits = list(set([*if_body.qubits, *else_body.qubits, *if_body.clbits, *else_body.clbits]))

                decomposed_circuit = QuantumCircuit(bits, name="DecomposedClassicalUnaryCircuit")
                with decomposed_circuit.if_test(condition) as _else:
                    decomposed_circuit.compose(if_body, if_body.qubits, if_body.clbits, inline_captures=True, inplace=True)
                with _else:
                    if else_body is not None:
                        decomposed_circuit.compose(else_body, else_body.qubits, else_body.clbits, inline_captures=True, inplace=True)

                # @TODO - I don't know how to check, so just forcing it again for safety
                self.property_set["decompose_if_else_ops_again"] = True
            elif isinstance(condition, Binary):
                if condition.op.name == "BIT_AND":
                    # Condition lvalue/rvalue type checking and parsing
                    # If there is a BIT_NOT gate in a condition, then it will be a Unary instead, which we have to handle differently
                    # If there is a nested condition, then it will be a Binary instead, which we have to handle differently
                    # @TODO - handle the other possible types more safely until reaching the final else branch would guarantee an error
                    if isinstance(condition.left, Unary):
                        left_var = condition.left.operand.var
                        left_val = 0 if condition.left.op.name == "BIT_NOT" else 1
                        left_condition = (left_var, left_val)
                    elif isinstance(condition.left, Binary):
                        left_condition = condition.left

                        self.property_set["decompose_if_else_ops_again"] = True
                    else:
                        left_var = condition.left.var
                        left_val = 1
                        left_condition = (left_var, left_val)

                    if isinstance(condition.right, Unary):
                        right_var = condition.right.operand.var
                        right_val = 0 if condition.right.op.name == "BIT_NOT" else 1
                        right_condition = (right_var, right_val)
                    elif isinstance(condition.right, Binary):
                        right_condition = condition.right

                        self.property_set["decompose_if_else_ops_again"] = True
                    else:
                        right_var = condition.right.var
                        right_val = 1
                        right_condition = (right_var, right_val)

                    if else_body is None:
                        bits = list(set([*if_else_op_node.qargs, *if_else_op_node.cargs, *if_body.qubits, *if_body.clbits]))
                    else:
                        bits = list(set([*if_else_op_node.qargs, *if_else_op_node.cargs, *if_body.qubits, *else_body.qubits, *if_body.clbits, *else_body.clbits]))

                    if self.method == "nested":
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

                        decomposed_circuit = QuantumCircuit(bits, name="DecomposedClassicalANDCircuit")
                        with decomposed_circuit.if_test(left_condition) as _else_left:
                            with decomposed_circuit.if_test(right_condition) as _else_right:
                                decomposed_circuit.compose(if_body, if_body.qubits, if_body.clbits, inline_captures=True, inplace=True)
                            with _else_right:
                                if else_body is not None:
                                    decomposed_circuit.compose(else_body, else_body.qubits, else_body.clbits, inline_captures=True, inplace=True)
                        with _else_left:
                            if else_body is not None:
                                decomposed_circuit.compose(else_body, else_body.qubits, else_body.clbits, inline_captures=True, inplace=True)
                    elif self.method == "sequential":
                        """
                        Decompose classical XOR gate via sequential {AND, NOT} decomposition:
                        if (A AND B): ...
                        === becomes ===
                        A -> C
                        B -> C [C now holds A AND B]
                        if (C): ...
                        """

                        creg_setter_qreg = [qubit for qubit in bits if "qsetter" in qubit._register.name]
                        intermediate_state_creg = [clbit for clbit in bits if "cintermediate_state" in clbit._register.name]
                        creg_setter_qreg.sort(key=lambda qubit : qubit._index)
                        intermediate_state_creg.sort(key=lambda clbit : clbit._index)

                        filler_instruction_indices_if_body = []
                        for i, instruction in enumerate(if_body.data):
                            for clbit in instruction.clbits:
                                if clbit in intermediate_state_creg:
                                    filler_instruction_indices_if_body.append(i)
                                    break

                        filler_instruction_indices_else_body = []
                        for i, instruction in enumerate(else_body.data):
                            for clbit in instruction.clbits:
                                if clbit in intermediate_state_creg:
                                    filler_instruction_indices_else_body.append(i)
                                    break

                        # for i in filler_instruction_indices_if_body[::-1]:
                        #     del if_body.data[i]
                        # for i in filler_instruction_indices_else_body[::-1]:
                        #     del else_body.data[i]

                        decomposed_circuit = QuantumCircuit(bits, name="DecomposedClassicalANDCircuit")

                        # @TODO - I might be double-counting NOT's because I sort of try to account for them in the condition handling, so check that

                        # A -> C
                        with decomposed_circuit.if_test(left_condition) as _else_left:
                            decomposed_circuit.measure(creg_setter_qreg[1], intermediate_state_creg[0])
                        with _else_left:
                            decomposed_circuit.measure(creg_setter_qreg[0], intermediate_state_creg[0])

                        # B -> C
                        with decomposed_circuit.if_test(right_condition) as _else_right:
                            decomposed_circuit.measure(creg_setter_qreg[1], intermediate_state_creg[0])
                        with _else_right:
                            decomposed_circuit.measure(creg_setter_qreg[0], intermediate_state_creg[0])

                        # C
                        with decomposed_circuit.if_test((intermediate_state_creg[0], 1)) as _else_right:
                            decomposed_circuit.compose(if_body, if_body.qubits, if_body.clbits, inline_captures=True, inplace=True)
                        with _else_right:
                            decomposed_circuit.compose(else_body, else_body.qubits, else_body.clbits, inline_captures=True, inplace=True)
                elif condition.op.name == "BIT_XOR":
                    if isinstance(condition.left, Unary):
                        left_var = condition.left.operand.var
                        left_val = 1 if condition.left.op.name == "BIT_NOT" else 0
                        left_condition = (left_var, left_val)
                    elif isinstance(condition.left, Binary):
                        left_condition = condition.left

                        self.property_set["decompose_if_else_ops_again"] = True
                    else:
                        left_var = condition.left.var
                        left_val = 0
                        left_condition = (left_var, left_val)

                    if isinstance(condition.right, Unary):
                        right_var = condition.right.operand.var
                        right_val = 1 if condition.right.op.name == "BIT_NOT" else 0
                        right_condition = (right_var, right_val)
                    elif isinstance(condition.right, Binary):
                        right_condition = condition.right

                        self.property_set["decompose_if_else_ops_again"] = True
                    else:
                        right_var = condition.right.var
                        right_val = 0
                        right_condition = (right_var, right_val)

                    if else_body is None:
                        bits = list(set([*if_else_op_node.qargs, *if_else_op_node.cargs, *if_body.qubits, *if_body.clbits]))
                    else:
                        bits = list(set([*if_else_op_node.qargs, *if_else_op_node.cargs, *if_body.qubits, *else_body.qubits, *if_body.clbits, *else_body.clbits]))

                    if self.method == "nested":
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

                        decomposed_circuit = QuantumCircuit(bits, name="DecomposedClassicalXORCircuit")
                        with decomposed_circuit.if_test(left_condition) as _else_left:
                            with decomposed_circuit.if_test(right_condition) as _else_right:
                                if else_body is not None:
                                    decomposed_circuit.compose(else_body, else_body.qubits, else_body.clbits, inline_captures=True, inplace=True)
                            with _else_right:
                                decomposed_circuit.compose(if_body, if_body.qubits, if_body.clbits, inline_captures=True, inplace=True)
                        with _else_left:
                            with decomposed_circuit.if_test(right_condition) as _else_right:
                                decomposed_circuit.compose(if_body, if_body.qubits, if_body.clbits, inline_captures=True, inplace=True)
                            with _else_right:
                                if else_body is not None:
                                    decomposed_circuit.compose(else_body, else_body.qubits, else_body.clbits, inline_captures=True, inplace=True)
                    elif self.method == "sequential":
                        """
                        Decompose classical XOR gate via sequential {AND, NOT} decomposition:
                        if (A XOR B): ...
                        === becomes ===
                        NOT A -> C
                        NOT B -> C
                        NOT C -> D [D now holds NOT(NOT A AND NOT B) = A OR B]
                        A -> C
                        B -> C [C now holds A AND B]
                        NOT C -> D [D now holds NOT(A AND B) AND (A OR B) = A XOR B]
                        if (D): ...
                        """

                        creg_setter_qreg = [qubit for qubit in bits if "qsetter" in qubit._register.name]
                        intermediate_state_creg = [clbit for clbit in bits if "cintermediate_state" in clbit._register.name]
                        creg_setter_qreg.sort(key=lambda qubit : qubit._index)
                        intermediate_state_creg.sort(key=lambda clbit : clbit._index)

                        filler_instruction_indices_if_body = []
                        for i, instruction in enumerate(if_body.data):
                            for clbit in instruction.clbits:
                                if clbit in intermediate_state_creg:
                                    filler_instruction_indices_if_body.append(i)
                                    break

                        filler_instruction_indices_else_body = []
                        for i, instruction in enumerate(else_body.data):
                            for clbit in instruction.clbits:
                                if clbit in intermediate_state_creg:
                                    filler_instruction_indices_else_body.append(i)
                                    break

                        # for i in filler_instruction_indices_if_body[::-1]:
                        #     del if_body.data[i]
                        # for i in filler_instruction_indices_else_body[::-1]:
                        #     del else_body.data[i]

                        decomposed_circuit = QuantumCircuit(bits, name="DecomposedClassicalXORCircuit")

                        # @TODO - I might be double-counting NOT's because I sort of try to account for them in the condition handling, so check that

                        # NOT(A) -> C
                        with decomposed_circuit.if_test(left_condition) as _else_left:
                            decomposed_circuit.measure(creg_setter_qreg[0], intermediate_state_creg[0])
                        with _else_left:
                            decomposed_circuit.measure(creg_setter_qreg[1], intermediate_state_creg[0])

                        # NOT(B) -> C
                        with decomposed_circuit.if_test(right_condition) as _else_right:
                            decomposed_circuit.measure(creg_setter_qreg[0], intermediate_state_creg[0])
                        with _else_right:
                            decomposed_circuit.measure(creg_setter_qreg[1], intermediate_state_creg[0])

                        # NOT(C) -> D
                        with decomposed_circuit.if_test((intermediate_state_creg[0], 1)) as _else:
                            decomposed_circuit.measure(creg_setter_qreg[0], intermediate_state_creg[1])
                        with _else:
                            decomposed_circuit.measure(creg_setter_qreg[1], intermediate_state_creg[1])

                        # A -> C
                        with decomposed_circuit.if_test(left_condition) as _else_left:
                            decomposed_circuit.measure(creg_setter_qreg[1], intermediate_state_creg[0])
                        with _else_left:
                            decomposed_circuit.measure(creg_setter_qreg[0], intermediate_state_creg[0])

                        # B -> C
                        with decomposed_circuit.if_test(right_condition) as _else_right:
                            decomposed_circuit.measure(creg_setter_qreg[1], intermediate_state_creg[0])
                        with _else_right:
                            decomposed_circuit.measure(creg_setter_qreg[0], intermediate_state_creg[0])

                        # D
                        with decomposed_circuit.if_test((intermediate_state_creg[0], 1)) as _else_right:
                            decomposed_circuit.compose(if_body, if_body.qubits, if_body.clbits, inline_captures=True, inplace=True)
                        with _else_right:
                            decomposed_circuit.compose(else_body, else_body.qubits, else_body.clbits, inline_captures=True, inplace=True)
                # else:
                #     print(f"WARNING - DecomposeIfElseOps encountered IfElseOp with label '{if_else_op.label}' which has condition with name '{condition.op.name}', skipping.")
            # else:
            #     print(f"WARNING - DecomposeIfElseOps encountered IfElseOp with label '{if_else_op.label}' which has condition of unrecognized type {type(condition)}, skipping.")
            #     continue

            if decomposed_circuit is not None:
                decomposed_dag = circuit_to_dag(decomposed_circuit)
                dag.substitute_node_with_dag(if_else_op_node, decomposed_dag)

        return dag

def decompose_if_else_ops_condition(property_set):
    # Return True when IfElseOp's contain an instance of Binary as condition lvalue(s)/rvalue(s)
    if "decompose_if_else_ops_again" not in property_set:
        property_set["decompose_if_else_ops"] = True

    return property_set.get("decompose_if_else_ops_again")

def DecomposeIfElseOpsTask():
    return DoWhileController(
        tasks=[DecomposeIfElseOps()],
        do_while=decompose_if_else_ops_condition,
    )

