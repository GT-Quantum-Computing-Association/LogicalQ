import numpy as py
from qiskit import QuantumCircuit

def cpmg_sequence(qc, qubit, tau):
    qc.delay(tau // 2, qubit)  
    qc.x(qubit_index)                
    qc.delay(tau, qubit)       
    qc.x(qubit_index)                
    qc.delay(tau // 2, qubit)  
    return qc

def xy4_sequence(qc, qubit, tau):
    qc.delay(tau // 2, qubit)
    qc.x(qubit_index)
    qc.delay(tau, qubit)
    qc.y(qubit_index)
    qc.delay(tau, qubit)
    qc.x(qubit_index)
    qc.delay(tau, qubit)
    qc.y(qubit)
    qc.delay(tau // 2, qubit)
    return qc

def inverse_xy4_sequence(qc, qubit, tau):
    qc.delay(tau // 2, qubit_index)
    qc.y(qubit_index)
    qc.delay(tau, qubit_index)
    qc.x(qubit_index)
    qc.delay(tau, qubit_index)
    qc.y(qubit_index)
    qc.delay(tau, qubit_index)
    qc.x(qubit_index)
    qc.delay(tau // 2, qubit_index)
    return qc

def xy8_sequence(qc, qubit, tau):
    xy4_sequence(qc, qubit, tau)
    inverse_xy4_sequence(qc, qubit, tau)
    return qc

def knill_block(qc, qubit, phase):
    def pulse(phi):
        qc.rz(phi, qubit)
        qc.x(qubit)
        qc.rz(-phi, qubit)
    pulse(np.pi/6 + phase)
    pulse(phase)
    pulse(np.pi/2 + phase)
    pulse(phase)
    pulse(np.pi/6 + phase)

def knill_sequence(qc, qubit, tau):
    phases = [0, np.pi/2, np.pi, 3*np.pi/2]
    for phi in phases:
        knill_block(qc, qubit, phi)
        if phi != phases[-1]:
            qc.delay(tau, qubit)
    qc.delay(tau // 2, qubit_index) 
    return qc

def uhrig_sequence(qc, qubit, npulses, total_time):
    pulse_times = [(total_time * (np.sin((j * np.pi) / (2 * npulses + 2)))**2) for j in range(1, npulses + 1)]
    qc.delay(int(pulse_times[0]), qubit)
    for i in range(npulses):
        qc.x(qubit)
        if i < npulses - 1:
            delay_duration = int(pulse_times[i+1] - pulse_times[i])
        else:
            delay_duration = int(total_time - pulse_times[-1])
        qc.delay(delay_duration, qubit)
    return qc