import numpy as np

from qiskit.circuit.library import UGate as IBMUGate
from qiskit.circuit.library import RZZGate

class UGate(IBMUGate):
    def __init__(self, theta, phi, label=None):
        """Create new U gate dependent on theta and phi following Quantinuum standard."""
        super().__init__(theta=theta, phi=phi-np.pi/2, lam=np.pi/2-phi, label=label)

class ZZGate(RZZGate):
    _standard_gate = None

    def __init__(self, label=None):
        """Create new ZZ gate."""
        super().__init__(theta=np.pi/2, label=label)
        self.name = "zz"
        
        assert (len(self.params) == 1) and (np.isclose(self.params[0], np.pi/2))

    def __eq__(self, other):
        if isinstance(other, ZZGate):
            return self._compare_parameters(other)
        return False
