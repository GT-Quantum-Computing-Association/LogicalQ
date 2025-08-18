from qiskit.circuit import BoxOp
from qiskit.transpiler import ConditionalController, DoWhileController
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.converters import circuit_to_dag

class UnBox(TransformationPass):
    """``UnBox`` transpilation pass unboxes BoxOp's."""

    def __init__(self):
        super().__init__()

    @control_flow.trivial_recurse
    def run(self, dag):
        self.property_set["unbox_agan"] = False

        for box_op in dag.op_nodes(BoxOp):
            assert len(box_op.op.params) == 1, f"BoxOp has more than one param: {box_op.op.params}"
            body = box_op.op.params[0]
            unboxed_dag = circuit_to_dag(body)
            dag.substitute_node_with_dag(box_op, unboxed_dag)

            if "box" in body.count_ops():
                self.property_set["unbox_again"] = True

        return dag

def unbox_condition(property_set):
    # Return True when circuit contains a BoxOp
    if "unbox_again" not in property_set:
        property_set["unbox_again"] = True

    return property_set.get("unbox_again", True)

def UnBoxTask():
    return DoWhileController(
        tasks=[UnBox()],
        do_while=unbox_condition,
    )

