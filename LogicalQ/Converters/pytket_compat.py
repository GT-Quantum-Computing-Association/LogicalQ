"""
LogicalQ-pytket compatibility module.

Provides classical condition simplification and fidelity verification
for LogicalCircuit to pytket conversion.

Includes monkey-patches for pytket-qiskit to support multi-bit conditions.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    QISKIT = True
except ImportError:
    QISKIT = False

try:
    from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
    PYTKET = True
except ImportError:
    PYTKET = False


# Apply monkey-patch for multi-bit condition support
def _apply_pytket_patch():
    """Patch pytket-qiskit to support Binary/Unary conditions in IfElseOp."""
    if not PYTKET:
        return False
    
    try:
        from pytket.extensions.qiskit import qiskit_convert
        from pytket.circuit import Bit, CircBox
        from qiskit.circuit.classical.expr import Unary, Binary, Var
        from qiskit.circuit import Clbit, ClassicalRegister, IfElseOp
        from pytket.circuit.logic_exp import reg_eq, reg_neq
        
        def flatten_condition(_condition, bits):
            """Flatten qiskit Binary/Unary conditions to pytket ClExpr."""
            if isinstance(_condition, Var):
                for bit in bits:
                    if bit.reg_name == _condition.var._register.name and bit.index[0] == _condition.var._index:
                        return bit
                raise ValueError(f"Var condition bit not found: {_condition.var}")
            elif isinstance(_condition, Unary):
                val = 0 if _condition.op.name == "BIT_NOT" else 1
                for bit in bits:
                    if bit.reg_name == _condition.operand.var._register.name and bit.index[0] == _condition.operand.var._index:
                        return bit ^ val
                raise ValueError(f"Unary condition bit not found")
            elif isinstance(_condition, Binary):
                if _condition.op.name == "BIT_AND":
                    return flatten_condition(_condition.left, bits) & flatten_condition(_condition.right, bits)
                elif _condition.op.name == "BIT_OR":
                    return flatten_condition(_condition.left, bits) | flatten_condition(_condition.right, bits)
                elif _condition.op.name == "BIT_XOR":
                    return flatten_condition(_condition.left, bits) ^ flatten_condition(_condition.right, bits)
                else:
                    raise ValueError(f"Unsupported Binary op: {_condition.op.name}")
            raise TypeError(f"Cannot flatten condition of type {type(_condition)}")
        
        def _patched_append_if_else(if_else_op, outer_builder, bits, qargs, cargs):
            """Patched version supporting Binary/Unary conditions."""
            if_circ, else_circ = qiskit_convert._pytket_circuits_from_ifelseop(
                if_else_op, outer_builder, qargs, cargs
            )
            
            # Use bits from parameter (passed by caller) or fall back to outer circuit
            if not bits:
                bits = list(outer_builder.tkc.bits)
            
            # Handle Binary/Unary conditions
            if isinstance(if_else_op.condition, (Unary, Binary)):
                condition_flattened = flatten_condition(if_else_op.condition, bits)
                
                outer_builder.tkc.add_circbox(
                    circbox=CircBox(if_circ),
                    args=if_circ.qubits + if_circ.bits,
                    condition=condition_flattened,
                )
                if else_circ is not None:
                    outer_builder.tkc.add_circbox(
                        circbox=CircBox(else_circ),
                        args=else_circ.qubits + else_circ.bits,
                        condition=1 ^ condition_flattened,
                    )
            elif isinstance(if_else_op.condition, Var):
                # Single bit Var condition
                condition_bits = []
                for bit in bits:
                    if bit.reg_name == if_else_op.condition.var._register.name and bit.index[0] == if_else_op.condition.var._index:
                        condition_bits.append(bit)
                
                if len(condition_bits) == 0:
                    raise ValueError("Failed to find pytket Bit matching Qiskit Clbit")
                
                outer_builder.tkc.add_circbox(
                    circbox=CircBox(if_circ),
                    args=if_circ.qubits + if_circ.bits,
                    condition_bits=condition_bits,
                    condition_value=1,
                )
                if else_circ is not None:
                    outer_builder.tkc.add_circbox(
                        circbox=CircBox(else_circ),
                        args=else_circ.qubits + else_circ.bits,
                        condition_bits=condition_bits,
                        condition_value=0,
                    )
            # Handle tuple (Clbit, value)
            elif hasattr(if_else_op.condition, "__getitem__") and isinstance(if_else_op.condition[0], Clbit):
                condition_bits = []
                for bit in bits:
                    if bit.reg_name == if_else_op.condition[0]._register.name and bit.index[0] == if_else_op.condition[0]._index:
                        condition_bits.append(bit)
                
                if len(condition_bits) == 0:
                    raise ValueError("Failed to find pytket Bit matching Qiskit Clbit")
                
                outer_builder.tkc.add_circbox(
                    circbox=CircBox(if_circ),
                    args=if_circ.qubits + if_circ.bits,
                    condition_bits=condition_bits,
                    condition_value=if_else_op.condition[1],
                )
                if else_circ is not None:
                    outer_builder.tkc.add_circbox(
                        circbox=CircBox(else_circ),
                        args=else_circ.qubits + else_circ.bits,
                        condition_bits=condition_bits,
                        condition_value=1 ^ if_else_op.condition[1],
                    )
            # Handle tuple (ClassicalRegister, value)
            elif hasattr(if_else_op.condition, "__getitem__") and isinstance(if_else_op.condition[0], ClassicalRegister):
                pytket_bit_reg = outer_builder.tkc.get_c_register(if_else_op.condition[0].name)
                
                outer_builder.tkc.add_circbox(
                    circbox=CircBox(if_circ),
                    args=if_circ.qubits + if_circ.bits,
                    condition=reg_eq(pytket_bit_reg, if_else_op.condition[1]),
                )
                if else_circ is not None:
                    outer_builder.tkc.add_circbox(
                        circbox=CircBox(else_circ),
                        args=else_circ.qubits + else_circ.bits,
                        condition=reg_neq(pytket_bit_reg, if_else_op.condition[1]),
                    )
            else:
                raise TypeError(
                    f"Unrecognized condition type: {type(if_else_op.condition)}"
                )
        
        # Replace the function
        qiskit_convert._append_if_else_circuit = _patched_append_if_else
        return True
        
    except Exception as e:
        print(f"Warning: Could not apply pytket patch: {e}")
        return False

# Apply patch on import
_PATCH_APPLIED = _apply_pytket_patch()


# Expression AST for classical condition simplification

class Expr:
    """Base class for boolean expressions."""
    def simplify(self): return self
    def eval(self, env: Dict[str, bool]) -> bool: raise NotImplementedError

class Const(Expr):
    def __init__(self, v: bool): self.v = v
    def __repr__(self): return '1' if self.v else '0'
    def eval(self, env): return self.v

class Var(Expr):
    def __init__(self, name: str): self.name = name
    def __repr__(self): return self.name
    def eval(self, env): return env.get(self.name, False)

class Not(Expr):
    def __init__(self, op: Expr): self.op = op
    def __repr__(self): return f'~{self.op}' if isinstance(self.op, (Const, Var)) else f'~({self.op})'
    def eval(self, env): return not self.op.eval(env)
    def simplify(self):
        s = self.op.simplify()
        if isinstance(s, Const): return Const(not s.v)
        if isinstance(s, Not): return s.op.simplify()
        if isinstance(s, And) and isinstance(s.l, Not) and isinstance(s.r, Not):
            return Or(s.l.op.simplify(), s.r.op.simplify())
        if isinstance(s, Or) and isinstance(s.l, Not) and isinstance(s.r, Not):
            return And(s.l.op.simplify(), s.r.op.simplify())
        return Not(s)

class And(Expr):
    def __init__(self, l: Expr, r: Expr): self.l, self.r = l, r
    def __repr__(self): return f'({self.l} & {self.r})'
    def eval(self, env): return self.l.eval(env) and self.r.eval(env)
    def simplify(self):
        l, r = self.l.simplify(), self.r.simplify()
        if isinstance(l, Const): return r if l.v else Const(False)
        if isinstance(r, Const): return l if r.v else Const(False)
        return And(l, r)

class Or(Expr):
    def __init__(self, l: Expr, r: Expr): self.l, self.r = l, r
    def __repr__(self): return f'({self.l} | {self.r})'
    def eval(self, env): return self.l.eval(env) or self.r.eval(env)
    def simplify(self):
        l, r = self.l.simplify(), self.r.simplify()
        if isinstance(l, Const): return Const(True) if l.v else r
        if isinstance(r, Const): return Const(True) if r.v else l
        return Or(l, r)

class Xor(Expr):
    def __init__(self, l: Expr, r: Expr): self.l, self.r = l, r
    def __repr__(self): return f'({self.l} ^ {self.r})'
    def eval(self, env): return self.l.eval(env) != self.r.eval(env)
    def simplify(self):
        l, r = self.l.simplify(), self.r.simplify()
        if isinstance(l, Const) and not l.v: return r
        if isinstance(r, Const) and not r.v: return l
        if isinstance(l, Const) and l.v: return Not(r).simplify()
        if isinstance(r, Const) and r.v: return Not(l).simplify()
        if repr(l) == repr(r): return Const(False)
        return Xor(l, r)


def parse(s: str) -> Expr:
    """Parse a boolean expression string into an AST."""
    s = s.strip()
    
    if s.startswith('(') and s.endswith(')'):
        depth, balanced = 0, True
        for i, c in enumerate(s):
            depth += (c == '(') - (c == ')')
            if depth == 0 and i < len(s) - 1: balanced = False; break
        if balanced: s = s[1:-1].strip()
    
    depth, pos, op = 0, -1, None
    for prec, ops in enumerate([('|',), ('&',), ('^',)]):
        for i, c in enumerate(s):
            depth += (c == '(') - (c == ')')
            if depth == 0 and c in ops: pos, op = i, c
        if pos >= 0: break
    
    if pos >= 0:
        l, r = s[:pos].strip(), s[pos+1:].strip()
        if op == '|': return Or(parse(l), parse(r))
        if op == '&': return And(parse(l), parse(r))
        if op == '^': return Xor(parse(l), parse(r))
    
    if s.startswith('~'): return Not(parse(s[1:]))
    if s.startswith('!'): return Not(parse(s[1:]))
    
    if s in ('0', 'False', 'false'): return Const(False)
    if s in ('1', 'True', 'true'): return Const(True)
    return Var(s)


def simplify(expr) -> str:
    """Simplify a boolean expression."""
    if isinstance(expr, str): expr = parse(expr)
    return repr(expr.simplify())


@dataclass
class FidelityReport:
    """Report from fidelity verification."""
    fidelity: float
    passed: bool
    gates: tuple
    depth: tuple


def verify_conversion(circuit, threshold: float = 1e-10, decompose: bool = True) -> FidelityReport:
    """Verify conversion preserves fidelity via statevector comparison."""
    if not QISKIT or not PYTKET:
        raise ImportError("qiskit and pytket-qiskit required")
    
    qc = circuit.to_qiskit() if hasattr(circuit, 'to_qiskit') else circuit
    
    if decompose:
        from qiskit.transpiler import PassManager
        from qiskit.transpiler.passes import Decompose
        from LogicalQ.Transpilation.UnBox import UnBoxTask
        from LogicalQ.Transpilation.DecomposeIfElseOps import DecomposeIfElseOpsTask
        from LogicalQ.Transpilation.FlattenIfElseOps import FlattenIfElseOpsTask
        
        def qreg_setter(circuit):
            return circuit.qubits
        
        pm = PassManager([
            UnBoxTask(),
            DecomposeIfElseOpsTask(),
            FlattenIfElseOpsTask(qreg_setter),
            Decompose()
        ])
        qc = pm.run(qc)
    
    orig_qubit_names = [str(q) for q in qc.qubits]
    
    qc_clean = qc.copy_empty_like()
    for instr in qc.data:
        if instr.operation.name not in ('measure', 'barrier'):
            qc_clean.append(instr.operation, instr.qubits, instr.clbits)
    
    tket = qiskit_to_tk(qc)
    back = tk_to_qiskit(tket)
    
    back_qubit_names = [str(q) for q in back.qubits]
    perm = []
    for orig_name in orig_qubit_names:
        orig_clean = orig_name.replace('Ancilla', '')
        for back_idx, back_name in enumerate(back_qubit_names):
            back_clean = back_name.replace('Ancilla', '')
            if orig_clean == back_clean:
                perm.append(back_idx)
                break
    
    back_clean = back.copy_empty_like()
    for instr in back.data:
        if instr.operation.name not in ('measure', 'barrier'):
            back_clean.append(instr.operation, instr.qubits, instr.clbits)
    
    sv1 = Statevector(qc_clean).data
    sv2 = Statevector(back_clean).data
    
    n_qubits = len(perm)
    inv_perm = [0] * n_qubits
    for orig_idx, back_idx in enumerate(perm):
        inv_perm[back_idx] = orig_idx
    
    sv2_permuted = np.zeros_like(sv2)
    for idx in range(len(sv2)):
        new_idx = 0
        for bit_pos in range(n_qubits):
            if idx & (1 << bit_pos):
                new_idx |= (1 << inv_perm[bit_pos])
        sv2_permuted[new_idx] = sv2[idx]
    
    fid = float(np.abs(np.vdot(sv1, sv2_permuted)) ** 2)
    return FidelityReport(fid, fid >= 1 - threshold, (len(qc.data), len(back.data)), (qc.depth(), back.depth()))


def to_pytket(circuit, verify: bool = True, threshold: float = 1e-10, for_quantinuum: bool = False):
    """Convert LogicalCircuit/QuantumCircuit to pytket circuit.
    
    Args:
        circuit: LogicalCircuit or QuantumCircuit
        verify: Run fidelity verification
        threshold: Fidelity threshold
        for_quantinuum: Apply Quantinuum-specific passes (breaks round-trip to qiskit)
    """
    if not PYTKET:
        raise ImportError("pytket-qiskit required")
    
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import Decompose
    from LogicalQ.Transpilation.UnBox import UnBoxTask
    from LogicalQ.Transpilation.DecomposeIfElseOps import DecomposeIfElseOpsTask
    from LogicalQ.Transpilation.FlattenIfElseOps import FlattenIfElseOpsTask
    
    qc = circuit.to_qiskit() if hasattr(circuit, 'to_qiskit') else circuit
    
    # FlattenIfElseOps needs a qreg_setter function
    def qreg_setter(circuit):
        return circuit.qubits
    
    pm = PassManager([
        UnBoxTask(),
        DecomposeIfElseOpsTask(),
        FlattenIfElseOpsTask(qreg_setter),  # Flatten nested conditionals
        Decompose()
    ])
    qc = pm.run(qc)
    
    tket = qiskit_to_tk(qc)
    
    # Apply Quantinuum-specific passes (these create scratch bits that break round-trip)
    if for_quantinuum:
        from pytket.passes import DecomposeClassicalExp, RemoveRedundancies
        DecomposeClassicalExp().apply(tket)
        RemoveRedundancies().apply(tket)
    
    report = verify_conversion(qc, threshold, decompose=False) if verify else None
    
    return tket, report
