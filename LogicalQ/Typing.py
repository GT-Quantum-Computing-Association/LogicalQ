import numpy as np

from qiskit_aer import AerSimulator
from qiskit.providers import Backend
from pytket.extensions.quantinuum import QuantinuumBackend
from qbraid.runtime.native.device import QbraidDevice

from .Logical import LogicalStatevector, LogicalDensityMatrix

from qiskit.quantum_info import Statevector

from typing import TYPE_CHECKING
from typing import TypeAlias, Iterable

class Sentinel: pass

DEFAULT = Sentinel()

Number: TypeAlias = int | float | complex | np.number

PhysicalQuantumState: TypeAlias = Iterable[Number] | Statevector

LogicalQuantumState: TypeAlias = Iterable[Number] | LogicalStatevector

QuantumState: TypeAlias = PhysicalQuantumState | LogicalQuantumState

QuantumBackend: TypeAlias = str | AerSimulator | Backend | QuantinuumBackend | QbraidDevice

