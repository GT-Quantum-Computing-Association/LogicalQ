from qiskit.circuit import BoxOp
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.converters import circuit_to_dag

class ClearQEC(TransformationPass):
    """``ClearQEC`` transpilation pass removes inessential QEC operations, excluding encoding but including encoding verification."""

    def __init__(self):
        super().__init__()

    @control_flow.trivial_recurse
    def run(self, dag):
        for box_op in dag.op_nodes(BoxOp):
            if (
                box_op.label is not None and
                box_op.label.startswith("logical.qec") and
                not box_op.label.startswith("logical.qec.encode")
            ):
                dag.remove_op_node(box_op)

        return dag

