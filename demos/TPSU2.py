#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[52]:


import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity

from LogicalQ.Logical import LogicalCircuit, LogicalStatevector, logical_state_fidelity
from LogicalQ.Library.QECCs import steane_code
from LogicalQ.Library.HardwareModels import hardware_models_Quantinuum
from LogicalQ.Execution import execute_circuits


# # Two Plaquette SU(2) Hamiltonian Model

# In[53]:


def TwoPlaquetteSU2TrotterStep(qc, x, t):
    qc.cx(1, 0)

    qc.ry(-1/2 * x * t, 0)
    qc.rz(-3/8 * t, 0)

    qc.cx(1, 0)

    qc.ry(-3/2 * x * t, 0)
    qc.rz(-9/8 * t, 1)

    qc.rz(-9/4 * t, 0)
    qc.ry(-3 * x * t, 1)

    qc.cx(0, 1)

    qc.ry(-1 * x * t, 1)

    qc.cx(0, 1)

    qc.ry(-3/2 * x * t, 0)
    qc.rz(-9/8 * t, 1)

    qc.cx(1, 0)

    qc.rz(-3/8 * t, 0)
    qc.ry(-1/2 * x * t, 0)

    qc.cx(1, 0)

def TwoPlaquetteSU2(x, dt, Nt, measure=True, mitigation=False):
    qc = QuantumCircuit(2)

    # Physical pass
    for _ in range(Nt):
        TwoPlaquetteSU2TrotterStep(qc, x, dt)

    if measure:
        qc.measure_all()

    # Mitigation passes
    if mitigation:
        # Forward-time half-pass
        for _ in range(Nt//2):
            TwoPlaquetteSU2TrotterStep(qc, x, dt)
        # Backward-time half-pass
        for _ in range(Nt//2):
            TwoPlaquetteSU2TrotterStep(qc, x, -dt)

        if measure:
            qc.measure_all()

    return qc


# # In[54]:


# qc = TwoPlaquetteSU2(2, 0.1, 1, measure=True, mitigation=False)
# qc.draw("mpl")


# # In[55]:


# qc_no_meas = TwoPlaquetteSU2(2, 0.1, 1, measure=False, mitigation=False)


# # # Circuit Splitting

# # In[56]:


# from qiskit_addon_utils.slicing import slice_by_depth

# def SplitCircuit(qc):
#     qc_split_list = [QuantumCircuit(1) for _ in range(qc.num_qubits)]

#     slices = slice_by_depth(qc, 1)
#     for s, qc_slice in enumerate(slices):
#         for qubit in qc.qubits:
#             qubit_used = False

#             for instruction in qc_slice.data:
#                 if qubit in instruction.qubits:
#                     if len(instruction.qubits) == 1:
#                         qc_split_list[qubit._index].append(instruction.operation, [0])
#                     else:
#                         qc_split_list[qubit._index].t(0)

#                     qubit_used = True

#             # if not qubit_used:
#                 # qc_split_list[qubit._index].id(0)

#     return qc_split_list

# qc_split_list = SplitCircuit(qc_no_meas)


# # In[57]:


# qc_split_list[0].draw("mpl")


# # In[58]:


# qc_split_list[1].draw("mpl")


# # In[59]:


# qc_no_meas = qc_split_list[0]


# # # Exact Simulation

# # In[60]:


# sv_exact = Statevector(qc_no_meas)
# sv_exact.draw("latex")


# # In[79]:


# print(np.abs(sv_exact))
# print(np.linalg.norm(np.abs(sv_exact)))


# # # Physical QuantumCircuit Simulation

# # In[66]:


# qc = qc_no_meas.copy()
# qc.measure_all()


# # In[67]:


# result_physical = execute_circuits(
#     qc,
#     backend="aer_simulator", method="statevector",
#     hardware_model=hardware_models_Quantinuum["H2-2"], coupling_map=None,
#     shots=int(1E6)
# )[0]
# result_physical


# # In[68]:


# counts = result_physical.get_counts()
# total_counts = sum(list(counts.values()))
# counts, total_counts


# # In[69]:


# amplitudes = []
# for o in range(2):
#     outcome_bin = bin(o)[2:]

#     amplitudes.append((counts.get(outcome_bin, 0.0)/total_counts)**0.5)

# sv = Statevector(amplitudes)
# print(sv)
# print(sv/np.linalg.norm(sv))


# # In[70]:


# fidelity_physical = state_fidelity(sv_exact, sv)
# fidelity_physical


# # # LogicalCircuit Simulation

# # In[80]:


# lqc = LogicalCircuit.from_physical_circuit(qc_no_meas, **steane_code, max_iterations=3)
# lqc.append_qec_cycle()
# lqc.measure_all(with_error_correction=False)
# lqc.draw("mpl", expr_len=100, fold_qec=True)


# # In[81]:


# result_logical = execute_circuits(
#     lqc,
#     backend="aer_simulator",
#     method="statevector",
#     # method="matrix_product_state",
#     # hardware_model=hardware_models_Quantinuum["H2-2"], coupling_map=None,
#     shots=int(1E4)
# )[0]
# result_logical


# # In[82]:


# counts = result_logical.get_counts()
# counts


# # In[83]:


# logical_counts = lqc.get_logical_counts(counts)
# logical_counts


# # In[84]:


# lsv = LogicalStatevector.from_counts(counts, lqc.n_logical_qubits, **steane_code)
# lsv.draw("latex")


# # In[85]:


# fidelity_logical = logical_state_fidelity(sv_exact, lsv)
# fidelity_logical


# # # QEC Cycle Optimization Experiment

# # In[88]:


# # Insert varying number of QEC cycles and see how fidelity behaves
# constraint_scan_keys = ["effective_threshold", "circuit_depth_logical_qubit"]
# constraint_scan_val_lists = [[1e-3], list(range(1,4+1))]

# # Concrete noise model (base Quantinuum H1-1)
# from qiskit_aer.noise import NoiseModel
# base_noise_model = NoiseModel()


# # In[ ]:


# from LogicalQ.Experiments import qec_cycle_efficiency_experiment

# qec_eff_data_serial = qec_cycle_efficiency_experiment(
#     circuit_input=qc,
#     qecc=steane_code,
#     constraint_scan_keys=constraint_scan_keys,
#     constraint_scan_val_lists=constraint_scan_val_lists,
#     backend="aer_simulator", method="statevector",
#     shots=int(1E4),
#     with_mp=False
# )


# # In[ ]:


# qec_eff_data_serial


# # In[ ]:


# from LogicalQ.Analysis import qec_cycle_efficiency_scatter
# qec_cycle_efficiency_scatter(qec_eff_data_serial)


# # # Quantinuum Hardware Execution

# # In[ ]:


# # result_hardware = execute_circuits(
# #     lqc_su2,
# #     backend="quantinuum_H2-2E",
# #     shots=int(1E3)
# # )
# # result_hardware


# # In[ ]:




