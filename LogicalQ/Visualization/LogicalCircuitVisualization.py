# This code is part of LogicalQ, altered from code from Qiskit.
# 
# ORIGINAL LICENSE:
# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from __future__ import annotations

import logging
import typing
from warnings import warn

from qiskit import user_config
from qiskit.circuit import ControlFlowOp, Measure

from qiskit.visualization.exceptions import VisualizationError
from .LogicalMatplotlibDrawer import _logical_matplotlib_circuit_drawer

if typing.TYPE_CHECKING:
    from typing import Any
    from qiskit.circuit import QuantumCircuit

logger = logging.getLogger(__name__)

def logical_circuit_drawer(
    circuit: QuantumCircuit,
    scale: float | None = None,
    filename: str | None = None,
    style: dict | str | None = None,
    output: str | None = None,
    interactive: bool = False,
    plot_barriers: bool = True,
    reverse_bits: bool | None = None,
    justify: str | None = None,
    vertical_compression: str | None = "medium",
    idle_wires: bool | str | None = None,
    with_layout: bool = True,
    fold: int | None = None,
    # The type of ax is matplotlib.axes.Axes, but this is not a fixed dependency, so cannot be
    # safely forward-referenced.
    ax: Any | None = None,
    initial_state: bool = False,
    cregbundle: bool | None = None,
    wire_order: list[int] | None = None,
    expr_len: int = 30,
    fold_qec=True,
    fold_qed=True,
    fold_logicalop=True
):
    image = None
    expr_len = max(expr_len, 0)
    config = user_config.get_config()
    
    # Get default from config file else use text
    default_output = "mpl"
    default_reverse_bits = False
    default_idle_wires = config.get("circuit_idle_wires", "auto")
    if config:
        default_output = config.get("circuit_drawer", "mpl")
        if wire_order is None:
            default_reverse_bits = config.get("circuit_reverse_bits", False)
    if output is None:
        output = default_output

    if reverse_bits is None:
        reverse_bits = default_reverse_bits

    if idle_wires is None:
        idle_wires = default_idle_wires
    if isinstance(idle_wires, str):
        if idle_wires == "auto":
            idle_wires = hasattr(circuit, "_layout") and circuit._layout is None
        else:
            raise VisualizationError(f"Parameter idle_wires={idle_wires} unrecognized.")

    if wire_order is not None and reverse_bits:
        raise VisualizationError(
            "The wire_order option cannot be set when the reverse_bits option is True."
        )

    complete_wire_order = wire_order
    if wire_order is not None:
        wire_order_len = len(wire_order)
        total_wire_len = circuit.num_qubits + circuit.num_clbits
        if wire_order_len not in [circuit.num_qubits, total_wire_len]:
            raise VisualizationError(
                f"The wire_order list (length {wire_order_len}) should as long as "
                f"the number of qubits ({circuit.num_qubits}) or the "
                f"total numbers of qubits and classical bits {total_wire_len}."
            )

        if len(set(wire_order)) != len(wire_order):
            raise VisualizationError("The wire_order list should not have repeated elements.")

        if wire_order_len == circuit.num_qubits:
            complete_wire_order = wire_order + list(range(circuit.num_qubits, total_wire_len))

    if (
        circuit.clbits
        and (reverse_bits or wire_order is not None)
        and not set(wire_order or []).issubset(set(range(circuit.num_qubits)))
    ):
        if cregbundle:
            warn(
                "cregbundle set to False since either reverse_bits or wire_order "
                "(over classical bit) has been set.",
                RuntimeWarning,
                2,
            )
        cregbundle = False

    def check_clbit_in_inst(circuit, cregbundle):
        if cregbundle is False:
            return False
        for inst in circuit.data:
            if isinstance(inst.operation, ControlFlowOp):
                for block in inst.operation.blocks:
                    if check_clbit_in_inst(block, cregbundle) is False:
                        return False
            elif inst.clbits and not isinstance(inst.operation, Measure):
                if cregbundle is not False:
                    warn(
                        "Cregbundle set to False since an instruction needs to refer"
                        " to individual classical wire",
                        RuntimeWarning,
                        3,
                    )
                return False

        return True

    cregbundle = check_clbit_in_inst(circuit, cregbundle)

    if output == "mpl":
        image = _logical_matplotlib_circuit_drawer(
            circuit,
            scale=scale,
            filename=filename,
            style=style,
            plot_barriers=plot_barriers,
            reverse_bits=reverse_bits,
            justify=justify,
            idle_wires=idle_wires,
            with_layout=with_layout,
            fold=fold,
            ax=ax,
            initial_state=initial_state,
            cregbundle=cregbundle,
            wire_order=complete_wire_order,
            expr_len=expr_len,
            fold_qec=fold_qec,
            fold_qed=fold_qed,
            fold_logicalop=fold_logicalop,
        )
    else:
        raise VisualizationError(
            f"Invalid output type {output} selected. The only valid choice is mpl"
        )
    if image and interactive:
        image.show()
    return image

