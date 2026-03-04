
from dataclasses import dataclass
from qiskit.transpiler import CouplingMap

@dataclass(init=False)
class HardwareModel:
    data: dict
    valid: bool
    
    def __init__(self, data: dict):
        self.data = data
        self.valid = self.validate()
        if not self.valid:
            raise ValueError("Data used to instantiate hardware model is not valid.")
        
    def validate(self) -> bool:
        num_qubits = None
        
        # Define utility function which checks if ls1 is a sublist of ls2
        def sublist(ls1, ls2):
            def get_all_in(one, another):
                for element in one:
                    if element in another:
                        yield element
            for x1, x2 in zip(get_all_in(ls1, ls2), get_all_in(ls2, ls1)):
                if x1 != x2:
                    return False
            return True
        
        # Verify both "device_info" and "noise_params" are included in dictionary
        if not sublist(list(self.data.keys()), ["device_info", "noise_params"]):
            raise ValueError("Top scope of HardwareModel can only contain the keys 'device_info' and 'noise_params'.")
        
        # Validate hardware parameters
        if "device_info" in self.data.keys():
            
            # Check structure of device_info is correct
            if not isinstance(self.data['device_info'], dict):
                raise TypeError("Value of 'device_info' must be a dictionary.")
            if not sublist(list(self.data['device_info'].keys()), ["n_qubits", "coupling_"]):
                raise ValueError("Key under dictionary associated to 'device_info' is not valid.")
            
            # Check that all fields contain roughly the right sort of information
            if 'num_qubits' in self.data['device_info'].keys():
                if not isinstance(self.data['device_info']['num_qubits'], int):
                    raise TypeError("Value of 'num_qubits' must be an integer (of type int).")
                num_qubits = self.data['device_info']['num_qubits']
                
            if 'coupling_map' in self.data['device_info'].keys():
                map = self.data['device_info']['coupling_map']
                # @TODO update valid strs list
                valid_coupling_strs = ["fully_coupled"]
                if not (isinstance(map, CouplingMap) or map in valid_coupling_strs):
                    raise TypeError("Value of 'num_qubits' must be an integer (of type int).")
            
            if 'basis_gates' in self.data['device_info'].keys():
                if not isinstance(self.data['device_info']['basis_gates'], dict):
                    raise TypeError("Value of 'basis_gates'" + ' must be of type dict, e.g. dict(zip(["r", "rz", "rzz"], [RGate, RZGate, RZZGate]))')
                
        # Validate noise_params
        # First layer -> qubit targets (which qubits does this information apply to?)
        # Second layer -> error type (e.g. t1, t2, dephasing_error, etc.)
        # Third layer -> Num qubits in gate (e.g. 1 for single-qubit gates, 2 for two-qubit gates, etc.)
        # Fourth layers -> values, which must be floats or integers

        # Define valid keys for certain fields
        valid_qubit_keys = ["all_qubit"] # + all ints (should be less than num_qubits if said field is populated). We will check under 'check second layer keys' whether num_qubits is defined, to determine which condition should be used for validation.
        valid_error_types = ["t1", "t2", "gate_time", "readout_error", "depolarizing_error", "amplitude_damping_error", "dephasing_error"]

        if 'noise_params' in self.data.keys():
            
            # Check first layer structure
            if not isinstance(self.data['noise_params'], dict):
                raise TypeError("Value of 'noise_params' must be a dictionary.")
            
            # Check first layer keys are str
            for key1 in self.data['noise_params'].keys():
                # Check key is str
                if not isinstance(key1, str):
                    raise TypeError(f"{key1} under 'noise_params' dict must be of type str.")
            
            for key1 in self.data['noise_params'].keys():    
                # Check whether first layer keys have valid values
                
                if num_qubits is None: # If num_qubits is None, then there is no integer limit
                    # Check that key is either in valid str list or is an integer
                    if not (key1 in valid_qubit_keys or key1.isdigit()):
                        raise ValueError(f"Value of {key1} under 'noise_params' dict is not valid.")
                else:
                    valid_qubit_keys = valid_qubit_keys + [str(i) for i in range(num_qubits)]
                    # Check that key is in valid str list
                    if not key1 in valid_qubit_keys:
                        raise ValueError(f"Value of {key1} under 'noise_params' dict is not valid.")

                # Check first layer structure
                val1 = self.data['noise_params'][key1]
                if not isinstance(val1, dict):
                    raise TypeError(f"Value of '{key1}' must be a dictionary.")
                
                # Check second layer keys
                if not sublist(list(val1.keys()), valid_error_types):
                    raise ValueError(f"Key under dictionary associated to '{key1}' is not valid.")
                
                print("Checks for implementation of dictionary associated to each error not yet implemented.")
        return True