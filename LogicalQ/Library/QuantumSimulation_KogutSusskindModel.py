import numpy as np

from LogicalQ.Logical import LogicalCircuit

from qiskit import QuantumCircuit
from qiskit.transpiler.passes import SolovayKitaev
from qiskit.synthesis import generate_basic_approximations
from qiskit.circuit.library import RXGate, RYGate, RZGate, RXXGate, RYYGate, RZZGate

def append_rot_gate(qc, axis = "z", theta = 0.0, qubit_indices = [0], label = None, return_subcircuit = False, depth = 10, recursion_degree = 1):
    gates = {"x": (RXGate, 1), "y": (RYGate, 1), "z": (RZGate, 1), "xx": (RXXGate, 2), "yy": (RYYGate, 2), "zz": (RZZGate, 2)}
    gate_base, num_target_qubits = gates[axis]
    gate = gate_base(theta)

    sub_qc = QuantumCircuit(num_target_qubits)

    def apply_Rzz(sub_qc):
        sub_qc.cx(0, 1)
        sub_qc.rz(theta, 1)
        sub_qc.cx(0, 1)

    match axis:
        case "xx":
            sub_qc.h([0, 1])
            apply_Rzz(sub_qc)
            sub_qc.h([0, 1])
        case "yy":
            sub_qc.rx(np.pi / 2, [0, 1])
            apply_Rzz(sub_qc)
            sub_qc.rx(-np.pi / 2, [0, 1])
        case "zz":
            apply_Rzz(sub_qc)
        case _:        
            sub_qc.append(gate, qargs = list(range(num_target_qubits)))

    basis = ["s", "sdg", "h", "x", "y", "z", "cz"]
    approx = generate_basic_approximations(basis, depth=depth)
    skd = SolovayKitaev(recursion_degree=recursion_degree, basic_approximations=approx)

    discretized_sub_qc = skd(sub_qc)
    box_label = fr"S-K: R$_\text{{{axis}}}$({np.round(theta / np.pi, 2)}$\pi$)" if label == None else label
    if isinstance(qc, LogicalCircuit):
        # with qc.box(label="logical.logicalop.R:{box_label}"):
        for i in range(len(sub_qc.data)):
            circuit_instruction = discretized_sub_qc.data[i]
            qc.append(circuit_instruction)
    elif isinstance(qc, QuantumCircuit):
        qc = discretized_sub_qc
        # sub_gate = discretized_sub_qc.to_gate(label=box_label)
        # qc.append(sub_gate, qubit_indices)
    else:
        raise TypeError("qc is not a valid QuantumCircuit or LogicalCircuit.")

    if return_subcircuit:
        return qc, discretized_sub_qc
    else:
        return qc

qc = QuantumCircuit(1)
qc = append_rot_gate(qc, "x", np.pi, [0])

def append_sx_gate(qc, qubits, **kwargs):
    append_rot_gate(qc, "x", np.pi/2, qubits, **kwargs)
    return

def append_sxdg_gate(qc, qubits, **kwargs):
    append_rot_gate(qc, "x", -np.pi/2, qubits, **kwargs)
    return qc

# ## Kogut-Susskind Model

def construct_Usp():
    qc = QuantumCircuit(3)

    qc.h(0)
    qc.x(1)
    qc.x(2)

    qc.cx(0, 1)
    qc.cx(1, 2)

    qc.h(1)

    return qc

def construct_U0(N_s, epsilon, **kwargs):
    qc = QuantumCircuit(3)

    m = N_s//2 - 1
    append_rot_gate(qc, "z", -m * epsilon, [0], **kwargs)
    append_rot_gate(qc, "x", epsilon, [0], **kwargs)
    append_rot_gate(qc, "z", m * epsilon, [0], **kwargs)

    return qc

def construct_Ufh1(epsilon, eta, **kwargs):
    qc = QuantumCircuit(3)

    qc.h(1)
    qc.cx(1, 0)
    qc.cx(1, 2)

    qc.h(0)
    append_rot_gate(qc, "x", -epsilon * eta/2, [0], **kwargs)
    qc.h(2)

    qc.cx(1, 0)

    qc.h(0)
    qc.z(1)

    qc.s(0)
    qc.cx(1, 2)

    append_rot_gate(qc, "x", -epsilon * eta/2, [0], **kwargs)
    qc.h(2)

    qc.s(2)
    qc.cx(1, 2)
    qc.cx(1, 0)

    qc.sdg(0)
    qc.h(1)
    qc.sdg(2)

    return qc

def construct_Ufh2(epsilon, eta, **kwargs):
    qc = QuantumCircuit(3)

    append_sxdg_gate(qc, [1], **kwargs)
    qc.s(1)
    qc.cx(0, 1)
    append_sx_gate(qc, [0], **kwargs)
    qc.cx(0, 2)

    append_rot_gate(qc, "x", -epsilon * eta/2, [0], **kwargs)
    append_rot_gate(qc, "y", -epsilon * eta/2, [2], **kwargs)

    qc.cx(0, 2)
    append_sxdg_gate(qc, [0], **kwargs)
    qc.cx(0, 1)
    qc.sdg(1)
    append_sx_gate(qc, [1], **kwargs)

    return qc

def construct_KogutSusskind_timestep(N_s, epsilon, eta, **kwargs):
    qc_U0 = construct_U0(N_s, epsilon, **kwargs)
    qc_Ufh1 = construct_Ufh1(epsilon, eta, **kwargs)
    qc_Ufh2 = construct_Ufh2(epsilon, eta, **kwargs)

    qc = QuantumCircuit(3)
    qc.compose(qc_U0, inplace=True)
    # qc.barrier()
    qc.compose(qc_Ufh1, inplace=True)
    # qc.barrier()
    qc.compose(qc_Ufh2, inplace=True)
    # qc.barrier()
    # qc.draw("mpl")

    return qc

def KogutSusskindHamiltonian(N_s, T, N_timesteps, eta, **kwargs):
    qc = construct_Usp()

    for ts in range(N_timesteps):
        epsilon = T/N_timesteps
        qc_KS_timestep = construct_KogutSusskind_timestep(N_s, epsilon, eta, **kwargs)
        qc.compose(qc_KS_timestep, inplace=True)

    return qc

if __name__ == "__main__":
    N_s = 2
    T = 1
    N_timesteps = 2
    eta = 1
    qc = KogutSusskindHamiltonian(N_s, T, N_timesteps, eta)
    print(qc.draw("text"))

