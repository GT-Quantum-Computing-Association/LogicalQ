
import numpy as np

from typing import List, Iterator
import contextlib

from qiskit import AncillaRegister
from qiskit.circuit import AncillaQubit

class AncillaReservoir:
    
    def __init__(
        self,
        num_ancillas,
        label = "reservoir_qreg",
        algorithm = "cyclic",
        backend = None
    ):
        """_summary_
        
        I think AncillaReservoir can support multiple allocation strategies. Below are 3 that I came up with.
        
        The first, "manual", is best in funky scenarios when there is an analytically optimal approach to ancilla allocation or the heuristic method fails. NOTE: probably don't need to add this as option to algorithm since request() is always available. Only allocate() needs to inherit that behavior. Maybe make it so that users will use:
        
        with reservoir.allocate(num_qubits):
            code that needs ancillas
        
        so that the deallocation is automatic as well.
        
        
        "cyclic" algorithm allocates ancillas cyclically, attempting to maximize the number of parallel gates which can be executed at once. "coupling_depdendent" algorithm accepts a coupling map and generates a heuristic algorithm to allocate based on factors such as proximity between the ancillas / target qubits.

        Args:
            num_ancillas (_type_): _description_
            label (str, optional): _description_. Defaults to "reservoir_qreg".
            algorithm (str, optional): _description_. Defaults to "cyclic".
        """
        
        # Saving kwargs from __init__
        self._num_ancillas = num_ancillas
        self._algorithm = algorithm
        self._reservoir = AncillaRegister(self.num_ancillas, name=label)
        self._coupling_map = None if (backend is None) else backend.configuration().coupling_map
        
        # Metadata
        self.status = np.array(self.num_ancillas * ["free"])
        self._allocation_history = []
        
        # Cyclic algorithm
        self._flag = 0 # Track which index is currently first free
        
    @contextlib.contextmanager
    def allocate(self, num_ancillas) -> Iterator[List[AncillaQubit]]:
        """Context manager to safely allocate and free ancilla qubits according to automatic algorithm."""
        
        alloc_indices = []
        alloc_qubits = []
        
        if self.algorithm == "cyclic":
            
            alloc_indices = list(range(self.flag, self.flag + num_ancillas))
            alloc_qubits = [self._reservoir[idx % self.num_ancillas] for idx in alloc_indices]
            
            self.flag = (self.flag + num_ancillas) % self.num_ancillas
            #return self.reservoir[*alloc_qubits]
        
        if self.algorithm == "coupling_based":
            # @TODO develop some sort of heuristic based on coupling map
            
            pass
        
        try:
            for i in alloc_indices:
                self._status[i] = "busy"
            yield alloc_qubits
        finally:
            for i in alloc_indices:
                self._status[i] = "free"
        
    def request(self, indices) -> List[AncillaQubit]:
        """
        Manual request of ancillas by index. Supports integers or lists. Not recommended, users should use allocate() when possible.
        """
        if isinstance(indices, int):
            indices = [indices]
        
        # Check that user is not requesting qubit that has already been allocated
        is_requesting_busy_qubit = np.isin(self._status[indices], "busy")
        if is_requesting_busy_qubit:
            raise ValueError(f"Failure to return requested qubits, as one or more requested qubits reports status 'busy'.")
            
        for i in indices:
            self._status[i] = "busy"
        
        return self.reservoir[indices]
    
    def free(self, indices):
        """
        Manually free ancillas by index. Supports integers or lists. Not recommended, users should use allocate() when possible. Using free() incorrectly can result in ancillas which are unintentionally accessed by multiple subcircuits simultaneously.
        """
        if isinstance(indices, int):
            indices = [indices]
        
        # Check that user is not trying to free an already free qubit
        is_requesting_free_qubit = np.isin(self._status[indices], "free")
        if is_requesting_free_qubit:
            print("WARNING: Attempting to free qubits which are already free. This will not throw an error, but perhaps check that you are freeing the correct index?")
        
        for i in indices:
            self._status[i] = "free"
        
    def next(self) -> AncillaQubit:
        """
        Retrieves next available ancilla (counting in ascending order of index).
        """

        free_indices = np.nonzero(self._status == "free")[0]

        if free_indices == []:
            raise ValueError("Not enough free ancillas available.")
        else:
            next_free = free_indices[0]
            self._status[next_free] = "busy"
            return self._reservoir[next_free]