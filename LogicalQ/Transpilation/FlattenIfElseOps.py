import copy

from qiskit.circuit import IfElseOp
from qiskit.circuit.classical import expr, types

from qiskit.transpiler import TransformationPass, DoWhileController
from qiskit.transpiler.passes.utils import control_flow
from qiskit.converters import dag_to_circuit, circuit_to_dag

class FlattenIfElseOps(TransformationPass):
    """``FlattenIfElseOps`` transpilation pass decomposes nested IfElseOp's into flat, sequential IfElseOp's."""

    def __init__(self, qreg_setter):
        self.qreg_setter = qreg_setter

        super().__init__()

    @control_flow.trivial_recurse
    def run(self, dag):
        self.property_set["flatten_if_else_ops_again"] = False

        if len(dag.op_nodes(IfElseOp)) == 0:
            return dag

        new_dag = dag.copy_empty_like()

        for node in dag.op_nodes():
            if node.op.name == "if_else":
                if_else_op = node.op
                condition = if_else_op.condition

                _if_body, _else_body = copy.deepcopy(if_else_op.params)

                for _branch, prefix in zip([_if_body, _else_body], [0, 1]):
                    if _branch is None:
                        continue

                    branch_dag = circuit_to_dag(_branch)
                    sub_dag = new_dag.copy_empty_like()
                    running_dag = new_dag.copy_empty_like()

                    outer_condition = parse_condition(prefix, condition)

                    for sub_node in branch_dag.op_nodes():
                        if sub_node.op.name == "if_else":
                            sub_branch_op = IfElseOp(
                                condition=outer_condition,
                                true_body=dag_to_circuit(running_dag),
                            )
                            sub_dag.apply_operation_back(sub_branch_op, qargs=new_dag.qubits, cargs=new_dag.clbits)

                            inner_condition = parse_condition(1, sub_node.op.condition)

                            sub_branch_op = IfElseOp(
                                condition=expr.Binary(expr.Binary.Op.BIT_AND, outer_condition, inner_condition, types.Bool()),
                                true_body=sub_node.op.params[0],
                            )
                            sub_dag.apply_operation_back(sub_branch_op, qargs=node.qargs, cargs=node.cargs)

                            running_dag = new_dag.copy_empty_like()
                            if sub_node.op.params[1] is not None:
                                running_dag.compose(circuit_to_dag(sub_node.op.params[1]), qubits=list(sub_node.qargs), clbits=list(sub_node.cargs), inplace=True)

                            # @TODO - I don't know the best way to check, so just forcing it again for safety
                            self.property_set["flatten_if_else_ops_again"] = True
                        else:
                            running_dag.apply_operation_back(sub_node.op, qargs=sub_node.qargs, cargs=sub_node.cargs)

                    if running_dag.num_ops() > 0:
                        sub_branch_op = IfElseOp(
                            condition=outer_condition,
                            true_body=dag_to_circuit(running_dag),
                        )
                        sub_dag.apply_operation_back(sub_branch_op, qargs=new_dag.qubits, cargs=new_dag.clbits)

                    new_dag.compose(sub_dag, qubits=new_dag.qubits, clbits=new_dag.clbits, inplace=True)
            else:
                new_dag.apply_operation_back(node.op, qargs=node.qargs, cargs=node.cargs)

        return new_dag

# Parse simpler condition object into type accepted by Qiskit
def parse_condition(prefix, condition):
    if isinstance(condition, tuple) and isinstance(condition[1], int):
        """
        Case 1:
            ```
            if (A == 0): do X
            else: do Y
            ```
        For the if branch:
            -> prefix = 0
            -> condition[1] = 0
            -> prefix ^ condition[1] = 0
            -> condition = (condition[0] == 0)
        For the else branch:
            -> prefix = 1
            -> condition[1] = 0
            -> prefix ^ condition[1] = 1
            -> condition = (condition[0] == 1)

        Case 2:
            ```
            if (A == 1): do X
            else: do Y
            ```
        For the if branch:
            -> prefix = 0
            -> condition[1] = 1
            -> prefix ^ condition[1] = 1
            -> condition = (condition[0] == 1)
        For the else branch:
            -> prefix = 1
            -> condition[1] = 1
            -> prefix ^ condition[1] = 0
            -> condition = (condition[0] == 0)
        """

        if prefix ^ condition[1] == 1:
            return expr.lift(condition[0])
        else:
            return expr.Unary(expr.Unary.Op.BIT_NOT, expr.lift(condition[0]), types.Bool())
    elif isinstance(condition, int):
        if prefix == 1:
            return expr.lift(condition)
        elif prefix == 0:
            return expr.Unary(expr.Unary.Op.BIT_NOT, expr.lift(condition), types.Bool())
        else:
            raise ValueError(f"Prefix must be 0 or 1, not {prefix}.")
    elif isinstance(condition, (expr.Var, expr.Unary, expr.Binary)):
        return condition
    else:
        # @TODO - account for other types
        raise TypeError(f"Invalid type for condition input: {type(condition)}.")

def flatten_if_else_ops_condition(property_set):
    # Return True when IfElseOp branch contains another IfElseOp
    if "flatten_if_else_ops_again" not in property_set:
        property_set["flatten_if_else_ops"] = True

    return property_set.get("flatten_if_else_ops_again")

def FlattenIfElseOpsTask(qreg_setter):
    return DoWhileController(
        tasks=[FlattenIfElseOps(qreg_setter)],
        do_while=flatten_if_else_ops_condition,
    )

