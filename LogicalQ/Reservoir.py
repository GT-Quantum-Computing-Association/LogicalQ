
import numpy as np

from typing import List, Iterator, Optional
import contextlib

from qiskit import AncillaRegister
from qiskit.circuit import AncillaQubit

import rustworkx as rx

class AncillaReservoir:
    
    def __init__(
        self,
        num_ancillas: int,
        name: str = "reservoir_qreg",
        algorithm: str = "cyclic",
        backend = None
    ):
        """
        Wrapper for AncillaRegister that allows for intelligent qubit allocation. Currently supports manual allocation, as well as two automatic algorithms:
            - "cyclic": allocates ancillas cyclically, attempting to maximize the number of parallel gates which can be executed at once
            - "min_path": when a coupling map is specified, we calculate the ancillas that are the minimum distance to the targets to in an attempt to minimize the necessary number of SWAP operations during compilation

        Args:
            num_ancillas (int): Number of ancillas in reservoir.
            label (str, optional): Name of the internal AncillaRegister object. Defaults to "reservoir_qreg".
            algorithm (str, optional): Algorithm used by allocate(). Defaults to "cyclic".
        """
        
        # Saving kwargs from __init__
        self._num_ancillas = num_ancillas
        self._algorithm = algorithm
        self._reservoir = AncillaRegister(self._num_ancillas, name=name)
        self._coupling_map = None if (backend is None) else backend.configuration().coupling_map
        
        # Metadata
        self.status = np.array(self._num_ancillas * ["free"])
        self._allocation_history = []
        
        # Cyclic algorithm
        self._flag = 0 # Track which index is currently first free
        
        # Coupling-map heuristic
        self._distance_matrix = None
        if backend:
            coupling_list = backend.configuration().coupling_map
            self._coupling_map = rx.PyGraph()
            self._coupling_map.add_nodes_from(range(backend.configuration().num_qubits))
            self._coupling_map.add_edges_from_no_data([(c[0], c[1]) for c in coupling_list])
            
            # pre-calculate all-pairs shortest paths (i think this scales poorly? may remove)
            self._distance_matrix = rx.graph_all_pairs_shortest_path_lengths(self._coupling_map)
        
    @contextlib.contextmanager
    def allocate(self, num_qubits, targets: Optional[List[int]] = None) -> Iterator[List[AncillaQubit]]:
        """
        Context manager to safely allocate and free ancilla qubits according to automatic algorithm.
        
        Args:
            num_qubits (int): Number of qubits to allocate.
            targets (Optional[List[int]]): If working with a choice of algorithm that is context-dependent (e.g. 'min_path'), provide a list of targets that informs the algorithm.
        
        Example:
        | reservoir = AncillaReservoir(10)
        | with reservoir.allocate(2) as ancillas:
        |     ancilla_A, ancilla_B = ancillas
        |     # Do whatever with your ancilla qubits...
        |     # Ancillas are automatically deallocated when out of scope.
        """
         
        allocated_indices = []
         
        if self._algorithm == "cyclic" or targets == None:
            # Get all free indices
            free_indices = np.where(self.status == "free")[0]
            
            if len(free_indices) < num_qubits:
                raise ValueError(f"Cannot allocate {num_qubits} qubits, only {len(free_indices)} are free.")

            # Find first free after flag
            start_offset = np.searchsorted(free_indices, self._flag)
            
            # Select indices to allocate
            num_free = len(free_indices)
            indices_to_take = [(start_offset + i) % num_free for i in range(num_qubits)]
            
            allocated_indices = free_indices[indices_to_take]
            
            # Move flag
            last_allocated_index = allocated_indices[-1]
            self._flag = (last_allocated_index + 1) % self._num_ancillas
        
        elif self._algorithm == "min_path":
            if self._distance_matrix is None:
                raise ValueError("Cannot use 'min_path' algorithm without a backend.")
            if targets is None:
                raise ValueError("'min_path' algorithm requires target data qubit indices.")
            
            # Score each free ancilla based on its total distance to all targets
            costs = []
            for ancilla_idx in free_indices:
                # Assuming reservoir indices map directly to the backend's physical qubit indices?
                # @TODO: check above is true
                total_distance = sum(self._distance_matrix[ancilla_idx][target_idx] for target_idx in targets)
                costs.append((ancilla_idx, total_distance))
            
            # Sort by cost (lower is better) and take the top N
            costs.sort(key=lambda x: x[1])
            allocated_indices = [idx for idx, cost in costs[:num_qubits]]
            
        else:
            raise NotImplementedError(f"Allocation algorithm '{self._algorithm}' is not implemented.")
        
        # Handle allocation
        try:
            self.status[allocated_indices] = "busy"
            yield [self._reservoir[i] for i in allocated_indices]
        finally:
            self.status[allocated_indices] = "free"
            
    def allocate_all(self):
        """
        Shortcut to allocate all ancillas. Assumes use of "cyclic" algorithm.
        """
        if self._algorithm != "cyclic":
            raise NotImplementedError("allocate_all() requires cyclic algorithm.")
        
        self.allocate(self._num_ancillas)
        
    def request(self, indices) -> List[AncillaQubit]:
        """
        Manual request of ancillas by index. Supports integers or lists. Not recommended, users should use allocate() when possible.
        """
        if isinstance(indices, int):
            indices = [indices]
        
        # Check that user is not requesting qubit that has already been allocated
        if np.any(self.status[indices] == "busy"):
            raise ValueError(f"Failure to return requested qubits, as one or more requested qubits reports status 'busy'.")
            
        for i in indices:
            self.status[i] = "busy"
        
        return self.reservoir[indices]
    
    def free(self, indices):
        """
        Manually free ancillas by index. Supports integers or lists. Not recommended, users should use allocate() when possible. Using free() incorrectly may result in a race condition.
        """
        if isinstance(indices, int):
            indices = [indices]
        
        # Check that user is not trying to free an already free qubit
        if np.any(self.status[indices] == "free"):
            print("WARNING: Attempting to free qubits which are already free. This will not throw an error, but perhaps check that you are freeing the correct index?")
        
        for i in indices:
            self.status[i] = "free"
        
    def next(self) -> AncillaQubit:
        """
        Retrieves next available ancilla (counting in ascending order of index).
        """

        free_indices = np.nonzero(self.status == "free")[0]

        if free_indices.size == 0:
            raise ValueError("Not enough free ancillas available.")
        else:
            next_free = free_indices[0]
            self.status[next_free] = "busy"
            return self._reservoir[next_free]
        
    def get_free_indices(self):
        """Returns a list of indices that have not yet been allocated."""
        
        return np.where(self.status == "free")[0]