from qiskit.circuit import BoxOp
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.converters import circuit_to_dag

class UnBox(TransformationPass):
    """``UnBox`` transpilation pass unboxes BoxOps."""

    def __init__(self):
        super().__init__()

    @control_flow.trivial_recurse
    def run(self, dag):
        for box_op in dag.op_nodes(BoxOp):
            assert len(box_op.op.params) == 1, f"BoxOp has more than one param: {box_op.op.params}"
            body = box_op.op.params[0]
            unboxed_dag = circuit_to_dag(body)
            dag.substitute_node_with_dag(box_op, unboxed_dag)

        return dag

