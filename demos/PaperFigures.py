#!/usr/bin/env python
"""Standalone, parallel version of ``demos/PaperFigures.ipynb``.

Every figure in the notebook is broken into independent tasks, which run on one process pool shared by all
figures:

- gadget-cost chunks (the constraint-model precompute),
- QEC schedule computations (``optimize_qec_cycle_indices``),
- physical-circuit runs,
- shot chunks of logical-circuit runs, whose logical counts are pooled afterwards.

Each figure is assembled, saved, and plotted as soon as its own tasks finish, so a long run produces figures
incrementally. Every completed task is checkpointed, so an interrupted run (walltime, OOM, Ctrl-C) resumes
where it stopped when rerun with the same ``--output-dir``.

Usage, from any directory, with LogicalQ installed or importable from this checkout:

    python demos/PaperFigures.py --fast              # quick end-to-end check (the notebook's FAST_MODE)
    python demos/PaperFigures.py                     # full-size figures, proxy headline circuit
    python demos/PaperFigures.py --heavy             # full-size figures, Kogut-Susskind headline circuit
    python demos/PaperFigures.py --figures 3,5 --workers 16
    python demos/PaperFigures.py --output-dir DIR    # resume an interrupted run
    python demos/PaperFigures.py --output-dir DIR --plot-only

Example Slurm job:

    #SBATCH --cpus-per-task=16
    #SBATCH --mem-per-cpu=8G
    python demos/PaperFigures.py --output-dir "$SLURM_SUBMIT_DIR/paper_figures"

Outputs under ``--output-dir`` (default ``./paper_figures_<timestamp>``):

    plots/            one PNG per figure
    data/             the assembled data behind each figure, for re-plotting with --plot-only
    logs/             stdout/stderr of each worker process (library output, warnings, tracebacks)
    run.log           progress log of the main process
    summary.json      per-figure status, failures, CPU time, and the slowest / most memory-hungry tasks
    run_config.json   every parameter of the run; a resume refuses to mix results from different configs
    checkpoint.pkls   completed task results

Differences from the notebook, all deliberate:

- Simulation method defaults to ``auto``: statevector for circuits of at most 24 encoded qubits and
  matrix_product_state above that. On the 14-qubit Steane proxy circuit statevector measured about 56 ms/shot
  against about 485 ms/shot for MPS, and both are exact, so results are unchanged up to shot noise. Pass
  ``--method matrix_product_state`` to reproduce the notebook exactly.
- Figure 7 evaluates the fixed-rate schedules themselves. The notebook cell was a scaffold that re-ran the
  no-QEC baseline for every interval.
- Figure 1 runs each physical circuit once and shares it across threshold scales, and plots the logical
  results through the same schedule and run primitives as the other figures.
- The gadget-cost cache is keyed by the gadget-library parameters, so a fast-mode cache is never silently
  reused for a full-mode run.
- The mirror benchmark circuit is seeded, so it is identical across resumed runs.
"""

import os
import sys
import argparse


def _apply_thread_limits():
    """Limit OpenMP/BLAS threads per process before numpy, Qiskit, or Aer are imported.

    Aer uses every core by default; with one worker per core that would oversubscribe the node. Workers
    inherit these variables, and setting them before the imports below also covers the ``fork`` start method.
    """

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--threads-per-worker", type=int, default=1)
    known_args, _ = pre_parser.parse_known_args()

    n_threads = str(max(1, known_args.threads_per_worker))
    for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[variable] = n_threads


if __name__ == "__main__":
    _apply_thread_limits()

# Qiskit's transpiler starts its own process pool whenever it is given several circuits, which inside our
# workers would nest pools (workers x cores processes) and re-import this script in each of them
os.environ["QISKIT_IN_PARALLEL"] = "TRUE"

import copy
import json
import math
import time
import heapq
import pickle
import shutil
import signal
import platform
import itertools
import traceback
import contextlib
import collections
import multiprocessing
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from concurrent.futures.process import BrokenProcessPool

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.ticker import MaxNLocator

try:
    import LogicalQ  # noqa: F401
except ImportError:
    # Allow running from a source checkout without installing the package
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import qiskit
import qiskit_aer
from qiskit import QuantumCircuit
from qiskit_addon_utils.slicing import slice_by_depth

from LogicalQ.Logical import LogicalCircuit
from LogicalQ.Execution import execute_circuits
from LogicalQ.Benchmarks import mirror_benchmarking, n_qubit_ghz_generation
from LogicalQ.Experiments import comparison_schedule_types, create_qec_schedules, compute_circuit_duration
from LogicalQ.Experiments import _exact_logical_state
from LogicalQ.Analysis import (
    qec_cycle_schedule_comparison_bar,
    qec_cycle_schedule_noise_scaling_scatter,
    noise_parameter_comparison_bar,
    qecc_comparison_bar,
    gadget_cost_differential_scatter,
    compute_schedule_fidelity,
    calculate_state_probability,
)
from LogicalQ.Estimators import build_gadgets_library, compute_gadget_costs, compute_constraint_model
from LogicalQ.Library.QECCs import steane_code, five_qubit_code
from LogicalQ.Library.HardwareModels import hardware_model_depolarizing, hardware_models_Quantinuum
from LogicalQ.Library.QuantumSimulation_KogutSusskindModel import KogutSusskindHamiltonian


##############################################
##### Configuration
##############################################

QECCS = {
    "steane": steane_code,
    "five_qubit": five_qubit_code,
}

# Display names used in figure legends, mapped to keys of QECCS
QECC_DISPLAY_NAMES = {
    "Steane [[7,1,3]]": "steane",
    "Five-qubit [[5,1,3]]": "five_qubit",
}

HARDWARE_MODELS = {
    "Depolarizing": hardware_model_depolarizing,
    "Quantinuum H1-1": hardware_models_Quantinuum["H1-1"],
    "Quantinuum H2-1": hardware_models_Quantinuum["H2-1"],
}

# With --method auto, circuits up to this many encoded qubits use statevector and larger ones use MPS
AUTO_STATEVECTOR_MAX_QUBITS = 24

# Gadgets per precompute task; fixed rather than derived from the worker count so that checkpoint keys do not
# depend on how many workers a run happened to use
GADGET_CHUNK_SIZE = 10

# A task whose worker process dies (e.g., OOM-killed) is retried this many times in total
MAX_ATTEMPTS = 3

STATUS_INTERVAL_SECONDS = 300

PRIORITY_PRECOMPUTE = 3
PRIORITY_SCHEDULE = 2
PRIORITY_RUN = 1


def build_config(fast, heavy, shots, method, fidelity_method, seed):
    """All hardcoded values from the notebook, keyed by the notebook's names where it had them."""

    return {
        "FAST_MODE": fast,
        "HEAVY_MODE": heavy,
        "SEED": seed,

        # Simulation settings shared by every figure
        "SHOTS": shots if shots is not None else (int(1E3) if fast else int(1E5)),
        "METHOD": method,
        # "logical_counts" is the Z-basis probability of the all-zero logical outcome and is cheap;
        # "density_matrix" is a phase-sensitive fidelity against the noiseless logical circuit
        "FIDELITY_METHOD": fidelity_method,

        "QECC": "steane",
        "HARDWARE_MODEL": "Quantinuum H2-1",

        # Constraint model
        "EFFECTIVE_THRESHOLD": 1E-3,
        "GADGET_MIN_DEPTH": 2,
        "GADGET_MAX_DEPTH": 2+1 if fast else 4+1,
        "GADGET_MIN_N_QUBITS": 1,
        "GADGET_MAX_N_QUBITS": 2+1,

        # Benchmark circuits
        "KS_N_S": 2,
        "KS_T": 1,
        "KS_N_TIMESTEPS": 2 if fast else 8,
        "KS_ETA": 1,
        "PROXY_N_REPEATS": 4 if fast else 12,

        # Figure 1
        "FIG1_MIN_CIRCUIT_LENGTH": 1,
        "FIG1_MAX_CIRCUIT_LENGTH": 4 if fast else 12,
        "FIG1_THRESHOLD_SCALES": [0.5, 1.0, 2.0],

        # Figure 2
        "FIG2_LOW_1Q": 1E-5,
        "FIG2_LOW_2Q": 1E-4,
        "FIG2_HIGH_1Q": 1E-3,
        "FIG2_HIGH_2Q": 1E-2,

        # Figure 3 (repeat counts are reused by Figures 4 and 6, as in the notebook)
        "FIG3_N_RANDOM_REPEATS": 2 if fast else 16,
        "FIG3_N_REPEATS": 1 if fast else 5,
        "FIG3_MIRROR_CIRCUIT_LENGTH": 4,

        # Figure 4
        "FIG4_HARDWARE_MODELS": ["Depolarizing", "Quantinuum H1-1", "Quantinuum H2-1"],

        # Figure 5
        "FIG5_NOISE_SCAN_KEY": ["noise_params", "all_qubit", "depolarizing_error", 2, "all"],
        "FIG5_NOISE_SCAN_VALUES": [1E-4, 5E-4, 1E-3] if fast else [1E-5, 5E-5, 1E-4, 5E-4, 1E-3, 5E-3],
        "FIG5_SCHEDULE_TYPES": ["no_qec", "fixed", "optimized"],

        # Figure 6
        "FIG6_QECCS": ["Steane [[7,1,3]]", "Five-qubit [[5,1,3]]"],
        "FIG6_WISHLIST": ["Repetition", "Bacon-Shor", "Reed-Muller"],

        # Figure 7
        "FIG7_INTERVALS": [1, 2, 3, 5, 10, 20],
        "FIG7_N_REPEATS": 1,

        # Figure 8
        "FIG8_DEPTH": 2,
        "FIG8_N_QUBITS": 1,
    }


##############################################
##### Circuits
##############################################

def proxy_circuit(n_repeats=4, measure=True):
    """Cheap stand-in with the same repetitive structure as a Trotterized simulation circuit."""

    qc = QuantumCircuit(1)
    for _ in range(n_repeats):
        qc.h(0)
        qc.t(0)

    if measure:
        qc.measure_all()

    return qc


def build_circuits(cfg):
    """Build every circuit used by the figures, keyed by a stable label used in checkpoint keys."""

    circuits = {}

    proxy_lengths = {cfg["PROXY_N_REPEATS"]} | set(range(cfg["FIG1_MIN_CIRCUIT_LENGTH"], cfg["FIG1_MAX_CIRCUIT_LENGTH"]+1))
    for n_repeats in sorted(proxy_lengths):
        circuits[f"proxy{n_repeats}"] = proxy_circuit(n_repeats=n_repeats)

    circuits["mirror"] = mirror_benchmarking(n_qubits=1, circuit_length=cfg["FIG3_MIRROR_CIRCUIT_LENGTH"], measure=True, seed=cfg["SEED"])
    circuits["ghz"] = n_qubit_ghz_generation(n_qubits=1)

    # KS is only built when it is the headline workload, since Steane-encoding it takes 38 qubits
    if cfg["HEAVY_MODE"]:
        circuits["ks"] = KogutSusskindHamiltonian(cfg["KS_N_S"], cfg["KS_T"], cfg["KS_N_TIMESTEPS"], cfg["KS_ETA"])

    return circuits


def headline_circuits(cfg):
    """The headline workload for Figures 2, 4, 5 and 9, as {display name: circuit label}."""

    if cfg["HEAVY_MODE"]:
        return {"Kogut-Susskind": "ks"}

    return {"Proxy": f"proxy{cfg['PROXY_N_REPEATS']}"}


def noise_parameter_groups(error_1q, error_2q):
    return {
        "Ideal": {"all_qubit": {}},
        "1q depolarizing": {"all_qubit": {"depolarizing_error": {1: {"all": error_1q}}}},
        "2q depolarizing": {"all_qubit": {"depolarizing_error": {2: {"all": error_2q}}}},
        "1q amp. damping": {"all_qubit": {"amplitude_damping_error": {1: {"all": error_1q}}}},
        "2q amp. damping": {"all_qubit": {"amplitude_damping_error": {2: {"all": error_2q}}}},
    }


def perturb_hardware_model(hardware_model, key_path, value):
    """Copy of a hardware model with the parameter at key_path set to value (as in the noise-scaling experiment)."""

    hardware_model_scan = copy.deepcopy(hardware_model)
    target = hardware_model_scan
    for key in list(key_path)[:-1]:
        target = target[key]
    target[list(key_path)[-1]] = value

    return hardware_model_scan


def resolve_method(method, n_qubits):
    if method == "auto":
        return "statevector" if n_qubits <= AUTO_STATEVECTOR_MAX_QUBITS else "matrix_product_state"

    return method


def shot_chunks(total_shots, shots_per_task):
    """Split total_shots into balanced chunks of at most shots_per_task."""

    n_chunks = max(1, math.ceil(total_shots/shots_per_task))
    base, extra = divmod(total_shots, n_chunks)

    return [base + (1 if i < extra else 0) for i in range(n_chunks)]


##############################################
##### Worker tasks
##############################################
# These run in worker processes. Each takes and returns only small, picklable objects.

class TaskError(Exception):
    """Carries a worker-side traceback back to the main process as plain text."""


def task_gadget_chunk(gadgets_library, hardware_model):
    _, gadget_infidelities, gadget_costs = compute_gadget_costs(
        gadgets_library=gadgets_library, backend="aer_simulator", hardware_model=hardware_model
    )

    return {"gadget_infidelities": gadget_infidelities, "gadget_costs": gadget_costs}


def task_schedule(circuit, qecc, constraint_model, compute_exact_state):
    """Optimized QEC schedule of a circuit, plus the matched cycle count every baseline method is held to."""

    circuit_no_meas = circuit.remove_final_measurements(inplace=False)

    lqc = LogicalCircuit.from_physical_circuit(circuit_no_meas, **qecc)
    qec_cycle_indices = lqc.optimize_qec_cycle_indices(constraint_model=constraint_model)

    exact_state = None
    if compute_exact_state:
        exact_state = _exact_logical_state(LogicalCircuit.from_physical_circuit(circuit_no_meas, **qecc))

    return {
        "qec_cycle_indices": qec_cycle_indices,
        # As in qec_cycle_schedule_comparison_experiment, the matched count is taken from logical qubit 0
        "n_qec_cycles": len(qec_cycle_indices.get(0, [])),
        "circuit_depth": len(slice_by_depth(circuit_no_meas, 1)),
        "n_encoded_qubits": lqc.num_qubits,
        "exact_state": exact_state,
    }


def task_physical(circuit, hardware_model, shots, method):
    circuit = copy.deepcopy(circuit)
    if "measure" not in circuit.count_ops():
        circuit.measure_all()

    method = resolve_method(method, circuit.num_qubits)
    result = execute_circuits(circuit, hardware_model=hardware_model, method=method, shots=shots)[0]

    return {"counts": dict(result.get_counts()), "shots": shots, "num_qubits": circuit.num_qubits, "method": method}


def task_logical(circuit, qecc, qec_cycle_indices, hardware_model, fidelity_method, exact_state, shots, method):
    """One shot chunk of a scheduled logical circuit, mirroring one draw of qec_cycle_schedule_comparison_experiment."""

    circuit_no_meas = circuit.remove_final_measurements(inplace=False)

    lqc = LogicalCircuit.from_physical_circuit(circuit_no_meas, **qecc)
    lqc.insert_qec_cycles(qec_cycle_indices=copy.deepcopy(qec_cycle_indices))
    lqc.measure_all()

    duration = compute_circuit_duration(lqc, hardware_model=hardware_model)

    method = resolve_method(method, lqc.num_qubits)
    result = execute_circuits(lqc, hardware_model=hardware_model, method=method, shots=shots, save_density_matrix=(fidelity_method == "density_matrix"))[0]

    record = {
        "duration": duration,
        "num_qec": len(qec_cycle_indices.get(0, [])),
        "shots": shots,
        "method": method,
        "n_encoded_qubits": lqc.num_qubits,
    }

    if fidelity_method == "logical_counts":
        # Logical counts are integer sums of physical counts, so chunks pool exactly
        record["logical_counts"] = lqc.get_logical_counts(result.get_counts())
        record["n_logical_qubits"] = lqc.n_logical_qubits
    else:
        record["fidelity"] = compute_schedule_fidelity(result, lqc, exact_state=exact_state, fidelity_method=fidelity_method)

    return record


def _reset_peak_rss():
    # Linux only: resets VmHWM so that the peak measured after a task belongs to that task alone
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
    except OSError:
        pass


def _peak_rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])/1024
    except OSError:
        pass

    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak/2**20 if sys.platform == "darwin" else peak/1024
    except ImportError:
        return float("nan")


def run_task(fn, kwargs, key, logs_dir):
    """Worker-side wrapper: sends library output to a per-worker log and reports time and peak memory."""

    _reset_peak_rss()
    log_path = os.path.join(logs_dir, f"worker-{os.getpid()}.log")

    start = time.perf_counter()
    with open(log_path, "a", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        print(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} start {key}")
        try:
            result = fn(**kwargs)
        except Exception as e:
            tb = traceback.format_exc()
            print(tb)
            raise TaskError(f"{type(e).__name__}: {e}\n\n{tb}\n(worker log: {log_path})") from None

        elapsed = time.perf_counter() - start
        print(f"===== done {key} in {elapsed:.1f}s")

    return result, {"elapsed": elapsed, "peak_rss_mb": _peak_rss_mb(), "pid": os.getpid()}


##############################################
##### Task runner
##############################################

class Group:
    """A figure, or the precompute stage. Collects task results and finalizes once nothing is outstanding."""

    def __init__(self, name, finalize):
        self.name = name
        self.finalize = finalize
        self.results = {}
        self.failures = []
        self.outstanding = 0
        self.sealed = False
        self.finished = False
        self.status = "running"
        self.outputs = []
        self.task_seconds = 0.0
        self.n_tasks_run = 0
        self.started = time.time()
        self.ended = None


class Task:
    def __init__(self, key, fn, kwargs, priority, subscribers):
        self.key = key
        self.fn = fn
        self.kwargs = kwargs
        self.priority = priority
        self.subscribers = subscribers
        self.attempts = 0


class Runner:
    """Priority-ordered process pool with deduplication, checkpointing, and recovery from dead workers.

    Tasks are requested by key. Requests for a key that is already queued or running share its result, and
    keys already in the checkpoint are not re-run. A continuation (``then``) can request further tasks, which
    is how schedule computations fan out into the runs that depend on them.
    """

    def __init__(self, n_workers, mp_context, out_dir, log, on_group_finished=None):
        self.n_workers = n_workers
        self.mp_context = mp_context
        self.log = log
        self.on_group_finished = on_group_finished
        self.logs_dir = out_dir / "logs"
        self.checkpoint_path = out_dir / "checkpoint.pkls"

        self.completed = self._load_checkpoint()
        self.n_checkpointed = len(self.completed)
        self.failed = {}
        self.pending = {}
        self.heap = []
        self.counter = itertools.count()
        self.inflight = {}
        self.ready = collections.deque()
        self.groups = []
        self.task_stats = []
        self.n_run = 0
        self.n_reused = 0
        self.pool = None
        self.pool_broken = False
        self.rebuilds_without_progress = 0
        self.started = time.time()

    # --- checkpoint ---

    def _load_checkpoint(self):
        completed = {}
        if not self.checkpoint_path.exists():
            return completed

        last_good = 0
        with open(self.checkpoint_path, "rb") as f:
            while True:
                try:
                    key, result = pickle.load(f)
                except EOFError:
                    break
                except Exception:
                    self.log(f"WARNING - checkpoint is truncated after {len(completed)} records (interrupted write); discarding the tail")
                    break
                completed[key] = result
                last_good = f.tell()

        # Drop a partial trailing record so that later appends stay readable
        if self.checkpoint_path.stat().st_size > last_good:
            with open(self.checkpoint_path, "r+b") as f:
                f.truncate(last_good)

        return completed

    def _save_checkpoint(self, key, result):
        with open(self.checkpoint_path, "ab") as f:
            pickle.dump((key, result), f, protocol=5)

    # --- requests ---

    def request(self, group, slot, key, fn, priority=PRIORITY_RUN, then=None, **kwargs):
        """Request the result of fn(**kwargs) under key, delivered into group.results[slot]."""

        group.outstanding += 1
        subscriber = (group, slot, then)

        if key in self.completed:
            self.n_reused += 1
            self.ready.append(([subscriber], key, True, self.completed[key], None))
        elif key in self.failed:
            self.ready.append(([subscriber], key, False, self.failed[key], None))
        elif key in self.pending:
            self.pending[key].subscribers.append(subscriber)
        else:
            task = Task(key, fn, kwargs, priority, [subscriber])
            self.pending[key] = task
            heapq.heappush(self.heap, (-priority, next(self.counter), task))

    def start_group(self, group, submit):
        self.groups.append(group)
        self.log(f"Starting {group.name}")

        try:
            submit(group)
        except Exception:
            group.failures.append({"key": None, "error": "submission failed:\n" + traceback.format_exc()})
            self.log(f"ERROR - submitting {group.name} failed:\n{traceback.format_exc()}")

        group.sealed = True
        self._maybe_finish(group)

    # --- delivery ---

    def _deliver(self, subscribers, key, ok, payload, stats):
        for group, slot, then in subscribers:
            if stats is not None:
                group.task_seconds += stats["elapsed"]
                group.n_tasks_run += 1

            if ok:
                group.results[slot] = payload
                if then is not None:
                    try:
                        then(payload)
                    except Exception:
                        group.failures.append({"key": key, "error": "continuation failed:\n" + traceback.format_exc()})
                        self.log(f"ERROR - continuation of {key} failed:\n{traceback.format_exc()}")
            else:
                group.failures.append({"key": key, "error": payload})

            group.outstanding -= 1
            self._maybe_finish(group)

    def _maybe_finish(self, group):
        if not group.sealed or group.finished or group.outstanding > 0:
            return

        group.finished = True
        group.ended = time.time()
        try:
            group.finalize(group)
            if group.status == "running":
                group.status = "complete" if len(group.failures) == 0 else "partial"
        except Exception:
            group.status = "failed"
            group.failures.append({"key": None, "error": "finalize failed:\n" + traceback.format_exc()})
            self.log(f"ERROR - finalizing {group.name} failed:\n{traceback.format_exc()}")

        self.log(f"Finished {group.name}: {group.status}, {len(group.failures)} failure(s), {group.task_seconds/3600:.2f} CPU-hours of tasks" + (f", wrote {', '.join(Path(p).name for p in group.outputs)}" if group.outputs else ""))

        if self.on_group_finished is not None:
            self.on_group_finished()

    # --- pool ---

    def _new_pool(self):
        return ProcessPoolExecutor(max_workers=self.n_workers, mp_context=self.mp_context)

    def _fill(self):
        while self.heap and len(self.inflight) < self.n_workers:
            _, _, task = heapq.heappop(self.heap)
            try:
                future = self.pool.submit(run_task, task.fn, task.kwargs, task.key, str(self.logs_dir))
            except BrokenProcessPool:
                heapq.heappush(self.heap, (-task.priority, next(self.counter), task))
                self.pool_broken = True
                return
            self.inflight[future] = task

    def _collect(self, future):
        task = self.inflight.pop(future)

        try:
            result, stats = future.result()
        except BrokenProcessPool:
            # Every in-flight task of a broken pool lands here, not only the one whose worker died
            self.pool_broken = True
            task.attempts += 1
            if task.attempts < MAX_ATTEMPTS:
                heapq.heappush(self.heap, (-task.priority, next(self.counter), task))
            else:
                self._fail(task, f"worker process died on {task.attempts} attempts (most likely killed for running out of memory)")
            return
        except Exception as e:
            self._fail(task, str(e))
            return

        self.pending.pop(task.key, None)
        self.completed[task.key] = result
        self._save_checkpoint(task.key, result)
        self.n_run += 1
        self.task_stats.append({"key": task.key, **stats})

        self.rebuilds_without_progress = 0
        # "left" grows while schedules finish and fan out into the runs that depend on them
        self.log(f"[{len(self.task_stats)} done, {len(self.pending)} left] {task.key} ({stats['elapsed']:.1f}s, peak {stats['peak_rss_mb']:.0f} MB)")

        self._deliver(task.subscribers, task.key, True, result, stats)

    def _fail(self, task, error):
        self.pending.pop(task.key, None)
        self.failed[task.key] = error
        self.log(f"FAILED {task.key}: {error.splitlines()[0] if error else ''}")
        self._deliver(task.subscribers, task.key, False, error, None)

    def _rebuild_pool(self):
        # Drain the rest of the broken pool's futures, which all fail together, then start a fresh pool
        if self.inflight:
            wait(list(self.inflight))
            for future in list(self.inflight):
                self._collect(future)

        self.rebuilds_without_progress += 1
        if self.rebuilds_without_progress > 2*MAX_ATTEMPTS:
            raise RuntimeError("The process pool keeps failing without completing any task; check logs/ and free memory")

        self.log("WARNING - a worker process died (most likely out of memory); restarting the pool and retrying in-flight tasks")
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.pool = self._new_pool()
        self.pool_broken = False

    def _status(self):
        elapsed = time.time() - self.started
        line = f"status: {len(self.task_stats)} run + {self.n_checkpointed} from checkpoint, {len(self.inflight)} running, {len(self.heap)} queued, {len(self.failed)} failed, elapsed {elapsed/3600:.2f} h"
        if self.task_stats and self.heap:
            mean = np.mean([stats["elapsed"] for stats in self.task_stats])
            line += f"; queued work at the current mean task time is ~{mean*len(self.heap)/self.n_workers/3600:.2f} h (more tasks appear as schedules finish)"
        self.log(line)

    def run(self):
        self.pool = self._new_pool()
        last_status = time.time()

        try:
            while self.heap or self.inflight or self.ready:
                while self.ready:
                    self._deliver(*self.ready.popleft())

                self._fill()
                if self.pool_broken:
                    self._rebuild_pool()
                    continue

                if not self.inflight:
                    continue

                done, _ = wait(list(self.inflight), timeout=STATUS_INTERVAL_SECONDS, return_when=FIRST_COMPLETED)
                for future in done:
                    self._collect(future)

                if self.pool_broken:
                    self._rebuild_pool()

                if time.time() - last_status > STATUS_INTERVAL_SECONDS:
                    self._status()
                    last_status = time.time()
        finally:
            self.shutdown()

    def shutdown(self):
        if self.pool is None:
            return

        if self.inflight:
            # Interrupted: stop running tasks rather than waiting for them
            self.pool.shutdown(wait=False, cancel_futures=True)
            if hasattr(self.pool, "terminate_workers"):
                self.pool.terminate_workers()
            else:
                for process in list(getattr(self.pool, "_processes", {}).values()):
                    process.terminate()
        else:
            self.pool.shutdown(wait=True)

        self.pool = None


##############################################
##### Shared figure machinery
##############################################

class Context:
    def __init__(self, cfg, out_dir, cache_dir, shots_per_task):
        self.cfg = cfg
        self.out_dir = out_dir
        self.plots_dir = out_dir / "plots"
        self.data_dir = out_dir / "data"
        self.cache_dir = cache_dir
        self.shots_per_task = shots_per_task

        self.qecc_name = cfg["QECC"]
        self.qecc = QECCS[self.qecc_name]
        self.hardware_model = HARDWARE_MODELS[cfg["HARDWARE_MODEL"]]

        self.circuits = build_circuits(cfg)
        self.headline = headline_circuits(cfg)

        self.gadget_costs = None
        self.constraint_models = {}

    def gadget_cache_path(self):
        cfg = self.cfg
        tag = "_".join([
            cfg["HARDWARE_MODEL"].replace(" ", "-"),
            f"depth{cfg['GADGET_MIN_DEPTH']}-{cfg['GADGET_MAX_DEPTH']}",
            f"nq{cfg['GADGET_MIN_N_QUBITS']}-{cfg['GADGET_MAX_N_QUBITS']}",
            f"seed{cfg['SEED']}",
        ])
        return self.cache_dir / f"gadget_costs_{tag}.pkl"


def request_schedule(runner, group, ctx, slot, circuit_label, qecc_name, cm_label, then, compute_exact_state=None):
    if compute_exact_state is None:
        compute_exact_state = ctx.cfg["FIDELITY_METHOD"] == "density_matrix"

    runner.request(
        group, slot, f"schedule/{qecc_name}/{circuit_label}/{cm_label}" + ("/exact" if compute_exact_state else ""),
        task_schedule, priority=PRIORITY_SCHEDULE, then=then,
        circuit=ctx.circuits[circuit_label], qecc=QECCS[qecc_name],
        constraint_model=ctx.constraint_models[cm_label], compute_exact_state=compute_exact_state,
    )


def request_logical_run(runner, group, ctx, slot, key, circuit_label, qecc, qec_cycle_indices, hardware_model, exact_state, fidelity_method=None):
    """Request one logical run, split into shot chunks; chunk results land at slot + (chunk, n_chunks)."""

    if fidelity_method is None:
        fidelity_method = ctx.cfg["FIDELITY_METHOD"]

    # A saved density matrix is not linear in the reported fidelity, so density-matrix runs are not split
    if fidelity_method == "density_matrix":
        chunks = [ctx.cfg["SHOTS"]]
    else:
        chunks = shot_chunks(ctx.cfg["SHOTS"], ctx.shots_per_task)

    for c, shots in enumerate(chunks):
        runner.request(
            group, tuple(slot) + (c, len(chunks)), f"{key}/chunk{c:03d}of{len(chunks):03d}",
            task_logical, priority=PRIORITY_RUN,
            circuit=ctx.circuits[circuit_label], qecc=qecc, qec_cycle_indices=qec_cycle_indices,
            hardware_model=hardware_model, fidelity_method=fidelity_method, exact_state=exact_state,
            shots=shots, method=ctx.cfg["METHOD"],
        )


def pooled_runs(group, prefix):
    """Pool the shot chunks of every run whose slot starts with prefix.

    Returns {run id: record}, where run id is the slot element after the prefix (e.g., the draw index). Runs
    with a missing chunk are left out and noted in group.failures, so that no point silently has fewer shots.
    """

    prefix = tuple(prefix)
    chunks_by_run = collections.defaultdict(dict)
    n_chunks_by_run = {}
    for slot, result in group.results.items():
        if len(slot) == len(prefix) + 3 and slot[:len(prefix)] == prefix:
            run_id, chunk, n_chunks = slot[len(prefix):]
            chunks_by_run[run_id][chunk] = result
            n_chunks_by_run[run_id] = n_chunks

    runs = {}
    for run_id, chunks in chunks_by_run.items():
        if len(chunks) != n_chunks_by_run[run_id]:
            group.failures.append({"key": None, "error": f"run {prefix + (run_id,)} dropped: only {len(chunks)} of {n_chunks_by_run[run_id]} shot chunks succeeded"})
            continue

        records = [chunks[c] for c in sorted(chunks)]
        if "fidelity" in records[0]:
            fidelity = records[0]["fidelity"]
        else:
            counts = collections.Counter()
            for record in records:
                counts.update(record["logical_counts"])
            total = sum(counts.values())
            fidelity = counts.get("0"*records[0]["n_logical_qubits"], 0)/total if total > 0 else 0.0

        runs[run_id] = {
            "fidelity": float(fidelity),
            "duration": records[0]["duration"],
            "num_qec": records[0]["num_qec"],
            "shots": sum(record["shots"] for record in records),
            "method": records[0]["method"],
        }

    return runs


def draw_schedules(circuit_no_meas, schedule, schedule_types, n_random_repeats, n_repeats, seed):
    """QEC cycle indices for every draw of every method, as qec_cycle_schedule_comparison_experiment builds them."""

    rng = np.random.default_rng(seed)

    draws = {}
    for schedule_type in schedule_types:
        n_draws = n_random_repeats if schedule_type == "random" else n_repeats
        draws[schedule_type] = []
        for _ in range(n_draws):
            if schedule_type == "optimized":
                qec_cycle_indices = copy.deepcopy(schedule["qec_cycle_indices"])
            else:
                qec_cycle_indices = create_qec_schedules(circuit_no_meas, schedule_type, n_qec_cycles=schedule["n_qec_cycles"], rng=rng)
            draws[schedule_type].append(qec_cycle_indices)

    return draws


def submit_comparison(runner, group, ctx, sub, circuits, qecc_name, cm_label, schedule_types, n_random_repeats, n_repeats, variants):
    """Task-level equivalent of qec_cycle_schedule_comparison_experiment.

    variants maps a label to the hardware model to execute under; the schedule is computed once from the
    constraint model and shared by every variant, which is what Figure 5 needs.
    """

    for circuit_name, circuit_label in circuits.items():
        def then(schedule, circuit_name=circuit_name, circuit_label=circuit_label):
            circuit_no_meas = ctx.circuits[circuit_label].remove_final_measurements(inplace=False)
            draws = draw_schedules(circuit_no_meas, schedule, schedule_types, n_random_repeats, n_repeats, ctx.cfg["SEED"])

            for variant, hardware_model in variants.items():
                for schedule_type, schedule_draws in draws.items():
                    for d, qec_cycle_indices in enumerate(schedule_draws):
                        request_logical_run(
                            runner, group, ctx,
                            slot=("run", sub, variant, circuit_name, schedule_type, d),
                            key=f"{group.name}/{sub}/{variant}/{circuit_name}/{schedule_type}/draw{d:03d}",
                            circuit_label=circuit_label, qecc=QECCS[qecc_name], qec_cycle_indices=qec_cycle_indices,
                            hardware_model=hardware_model, exact_state=schedule["exact_state"],
                        )

        request_schedule(runner, group, ctx, ("schedule", sub, circuit_name), circuit_label, qecc_name, cm_label, then)


def assemble_comparison(group, sub, variant, circuit_names, schedule_types):
    """Results in the format returned by qec_cycle_schedule_comparison_experiment."""

    all_data = {"circuits": {}, "summary": {}}

    for circuit_name in circuit_names:
        schedule = group.results.get(("schedule", sub, circuit_name))
        if schedule is None:
            continue

        methods = {}
        for schedule_type in schedule_types:
            runs = pooled_runs(group, ("run", sub, variant, circuit_name, schedule_type))
            if len(runs) == 0:
                continue

            records = [runs[d] for d in sorted(runs)]
            fidelities = [record["fidelity"] for record in records]
            methods[schedule_type] = {
                "fidelity": float(np.mean(fidelities)),
                "fidelity_std": float(np.std(fidelities)),
                "fidelities": fidelities,
                "n_repeats": len(records),
                "num_qec": int(np.mean([record["num_qec"] for record in records])),
                "duration": float(np.mean([record["duration"] for record in records])),
                "shots": records[0]["shots"],
            }

        # Report every method against the no-QEC baseline, which establishes whether the optimum is interior
        if "no_qec" in methods:
            fidelity_no_qec = methods["no_qec"]["fidelity"]
            if fidelity_no_qec and fidelity_no_qec > 0:
                for method_results in methods.values():
                    method_results["vs_no_qec"] = method_results["fidelity"]/fidelity_no_qec

        all_data["circuits"][circuit_name] = {
            "circuit_depth": schedule["circuit_depth"],
            "n_qec_cycles_matched": schedule["n_qec_cycles"],
            "n_encoded_qubits": schedule["n_encoded_qubits"],
            "exact_state": schedule["exact_state"],
            "methods": methods,
        }

    return all_data


def _save_dir(plots_dir):
    # The library plotters build paths as f"{save_dir}{filename}"
    return str(plots_dir) + os.sep


def _close_and_list(plots_dir, filename):
    plt.close("all")
    return [str(Path(plots_dir) / filename)]


##############################################
##### Precompute: gadget costs and constraint model
##############################################

def submit_precompute(runner, group, ctx):
    cfg = ctx.cfg
    cache_path = ctx.gadget_cache_path()

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            ctx.gadget_costs = pickle.load(f)["gadget_costs"]
        runner.log(f"Loaded cached gadget costs from {cache_path}")
        return

    # build_gadgets_library draws 2-qubit target orderings from the global numpy RNG
    np.random.seed(cfg["SEED"])
    gadgets_library = build_gadgets_library(
        min_depth=cfg["GADGET_MIN_DEPTH"], max_depth=cfg["GADGET_MAX_DEPTH"], step_depth=1,
        min_n_qubits=cfg["GADGET_MIN_N_QUBITS"], max_n_qubits=cfg["GADGET_MAX_N_QUBITS"], step_n_qubits=1,
    )

    n_gadgets = sum(len(gadgets) for depth_gadgets in gadgets_library.values() for gadgets in depth_gadgets.values())
    runner.log(f"Computing gadget costs for {n_gadgets} gadgets in chunks of {GADGET_CHUNK_SIZE} (cache: {cache_path})")

    tag = cache_path.stem
    for depth, depth_gadgets in gadgets_library.items():
        for n_qubits, gadgets in depth_gadgets.items():
            for start in range(0, len(gadgets), GADGET_CHUNK_SIZE):
                runner.request(
                    group, ("gadgets", depth, n_qubits, start), f"{tag}/depth{depth}/nq{n_qubits}/{start:05d}",
                    task_gadget_chunk, priority=PRIORITY_PRECOMPUTE,
                    gadgets_library={depth: {n_qubits: gadgets[start:start+GADGET_CHUNK_SIZE]}},
                    hardware_model=ctx.hardware_model,
                )


def finalize_precompute(group, ctx, runner, start_figures):
    cfg = ctx.cfg

    if ctx.gadget_costs is None:
        if len(group.failures) > 0:
            raise RuntimeError("Gadget-cost precompute had failed chunks, so no constraint model can be built and no figures will run; see the failures in summary.json")

        gadget_costs = {}
        gadget_infidelities = {}
        for slot in sorted(group.results):
            _, depth, n_qubits, _ = slot
            chunk = group.results[slot]
            gadget_costs.setdefault(depth, {}).setdefault(n_qubits, {}).update(chunk["gadget_costs"][depth][n_qubits])
            gadget_infidelities.setdefault(depth, {}).setdefault(n_qubits, {}).update(chunk["gadget_infidelities"][depth][n_qubits])

        ctx.gadget_costs = gadget_costs

        cache_path = ctx.gadget_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(".tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump({"gadget_costs": gadget_costs, "gadget_infidelities": gadget_infidelities}, f, protocol=5)
        os.replace(tmp_path, cache_path)
        runner.log(f"Cached gadget costs to {cache_path}")

    constraint_model = compute_constraint_model(
        ctx.hardware_model, **ctx.qecc,
        effective_threshold=cfg["EFFECTIVE_THRESHOLD"],
        gadget_costs=ctx.gadget_costs,
    )
    ctx.constraint_models["base"] = constraint_model

    n_gadget_keys = len([key for key in constraint_model if key.startswith("cost_") and "-" in key])
    runner.log(f"Constraint model: {len(constraint_model)} keys ({len(constraint_model) - n_gadget_keys - 1} single-component, {n_gadget_keys} gadget)")

    start_figures()


##############################################
##### Figure 1 - Noise scales with circuit size, and smart QEC beats it down
##############################################
# Infidelity against circuit length on the proxy circuit, for several scalings of the effective threshold.

def submit_fig1(runner, group, ctx):
    cfg = ctx.cfg

    for threshold_scale in cfg["FIG1_THRESHOLD_SCALES"]:
        constraint_model_scaled = dict(ctx.constraint_models["base"])
        constraint_model_scaled["effective_threshold"] = cfg["EFFECTIVE_THRESHOLD"] * threshold_scale
        ctx.constraint_models[f"threshold-x{threshold_scale:g}"] = constraint_model_scaled

    for circuit_length in range(cfg["FIG1_MIN_CIRCUIT_LENGTH"], cfg["FIG1_MAX_CIRCUIT_LENGTH"]+1):
        circuit_label = f"proxy{circuit_length}"

        runner.request(
            group, ("physical", circuit_length), f"fig1/physical/length{circuit_length:03d}",
            task_physical, circuit=ctx.circuits[circuit_label], hardware_model=ctx.hardware_model,
            shots=cfg["SHOTS"], method=cfg["METHOD"],
        )

        for threshold_scale in cfg["FIG1_THRESHOLD_SCALES"]:
            def then(schedule, circuit_length=circuit_length, circuit_label=circuit_label, threshold_scale=threshold_scale):
                request_logical_run(
                    runner, group, ctx,
                    slot=("run", threshold_scale, circuit_length),
                    key=f"fig1/threshold-x{threshold_scale:g}/length{circuit_length:03d}",
                    circuit_label=circuit_label, qecc=ctx.qecc, qec_cycle_indices=schedule["qec_cycle_indices"],
                    hardware_model=ctx.hardware_model, exact_state=None, fidelity_method="logical_counts",
                )

            request_schedule(runner, group, ctx, ("schedule", threshold_scale, circuit_length), circuit_label, ctx.qecc_name, f"threshold-x{threshold_scale:g}", then, compute_exact_state=False)


def assemble_fig1(group, ctx):
    cfg = ctx.cfg
    circuit_lengths = list(range(cfg["FIG1_MIN_CIRCUIT_LENGTH"], cfg["FIG1_MAX_CIRCUIT_LENGTH"]+1))

    infidelity_physical = {}
    for circuit_length in circuit_lengths:
        physical = group.results.get(("physical", circuit_length))
        if physical is not None:
            infidelity_physical[circuit_length] = 1.0 - calculate_state_probability("0"*physical["num_qubits"], physical["counts"])

    infidelity_logical = {}
    num_qec = {}
    for threshold_scale in cfg["FIG1_THRESHOLD_SCALES"]:
        runs = pooled_runs(group, ("run", threshold_scale))
        infidelity_logical[threshold_scale] = {circuit_length: 1.0 - run["fidelity"] for circuit_length, run in runs.items()}
        num_qec[threshold_scale] = {circuit_length: run["num_qec"] for circuit_length, run in runs.items()}

    return {
        "circuit_lengths": circuit_lengths,
        "infidelity_physical": infidelity_physical,
        "infidelity_logical": infidelity_logical,
        "num_qec": num_qec,
    }


def plot_fig1(data, plots_dir):
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")

    for threshold_scale, by_length in data["infidelity_logical"].items():
        circuit_lengths = sorted(by_length)
        ax.plot(circuit_lengths, [by_length[length] for length in circuit_lengths], marker="o", label=f"With QEC ($T \\times {threshold_scale}$)")

    circuit_lengths = sorted(data["infidelity_physical"])
    ax.plot(circuit_lengths, [data["infidelity_physical"][length] for length in circuit_lengths], marker="x", linestyle="--", color="black", label="Without QEC")

    ax.set_xlabel("Circuit length")
    ax.set_ylabel("Infidelity")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.savefig(Path(plots_dir) / "fig1_circuit_scaling.png", dpi=256)
    return _close_and_list(plots_dir, "fig1_circuit_scaling.png")


##############################################
##### Figure 2 - Sensitivity to individual noise channels
##############################################
# Each group applies one noise channel in isolation; (a) low error rates, (b) high error rates. With
# logical_counts scoring the "Ideal" group sits at 1.0 only if the circuit's noiseless output is |0>_L.

def submit_fig2(runner, group, ctx, error_1q, error_2q):
    groups = noise_parameter_groups(error_1q, error_2q)

    def group_hardware_model(noise_params):
        return {
            "device_info": copy.deepcopy(ctx.hardware_model["device_info"]),
            "noise_params": copy.deepcopy(noise_params),
        }

    for circuit_name, circuit_label in ctx.headline.items():
        for group_name, noise_params in groups.items():
            runner.request(
                group, ("physical", circuit_name, group_name), f"{group.name}/physical/{circuit_name}/{group_name}",
                task_physical, circuit=ctx.circuits[circuit_label], hardware_model=group_hardware_model(noise_params),
                shots=ctx.cfg["SHOTS"], method=ctx.cfg["METHOD"],
            )

        def then(schedule, circuit_name=circuit_name, circuit_label=circuit_label):
            for group_name, noise_params in groups.items():
                request_logical_run(
                    runner, group, ctx,
                    slot=("run", circuit_name, group_name),
                    key=f"{group.name}/logical/{circuit_name}/{group_name}",
                    circuit_label=circuit_label, qecc=ctx.qecc, qec_cycle_indices=schedule["qec_cycle_indices"],
                    hardware_model=group_hardware_model(noise_params), exact_state=schedule["exact_state"],
                )

        request_schedule(runner, group, ctx, ("schedule", circuit_name), circuit_label, ctx.qecc_name, "base", then)


def assemble_fig2(group, ctx, error_1q, error_2q, title):
    """Results in the format returned by noise_parameter_comparison_experiment."""

    group_names = list(noise_parameter_groups(error_1q, error_2q).keys())

    all_data = {"circuits": {}, "title": title}
    for circuit_name in ctx.headline:
        logical_runs = pooled_runs(group, ("run", circuit_name))

        groups = {}
        for group_name in group_names:
            group_results = {}

            physical = group.results.get(("physical", circuit_name, group_name))
            if physical is not None:
                group_results["fidelity_physical"] = calculate_state_probability("0"*physical["num_qubits"], physical["counts"])

            if group_name in logical_runs:
                group_results["fidelity_logical"] = logical_runs[group_name]["fidelity"]
                group_results["num_qec"] = logical_runs[group_name]["num_qec"]

            groups[group_name] = group_results

        all_data["circuits"][circuit_name] = {"groups": groups}

    return all_data


def plot_fig2(data, plots_dir, filename):
    noise_parameter_comparison_bar(data, title=data.get("title"), save=True, filename=filename, save_dir=_save_dir(plots_dir), show=False)
    return _close_and_list(plots_dir, filename)


##############################################
##### Figure 3 - The optimizer performs well across circuit types
##############################################
# Every method is held to the cycle count of the optimized schedule, so differences come from placement; the
# no-QEC baseline is the exception and establishes whether the optimum is interior at all.

def fig3_benchmarks(ctx):
    return {
        "Proxy": f"proxy{ctx.cfg['PROXY_N_REPEATS']}",
        "Mirror": "mirror",
        "GHZ": "ghz",
    }


def submit_fig3(runner, group, ctx):
    submit_comparison(
        runner, group, ctx, sub="main", circuits=fig3_benchmarks(ctx), qecc_name=ctx.qecc_name, cm_label="base",
        schedule_types=comparison_schedule_types,
        n_random_repeats=ctx.cfg["FIG3_N_RANDOM_REPEATS"], n_repeats=ctx.cfg["FIG3_N_REPEATS"],
        variants={"base": ctx.hardware_model},
    )


def assemble_fig3(group, ctx):
    return assemble_comparison(group, "main", "base", list(fig3_benchmarks(ctx)), comparison_schedule_types)


def plot_fig3(data, plots_dir):
    qec_cycle_schedule_comparison_bar(data, save=True, filename="fig3_circuit_types.png", save_dir=_save_dir(plots_dir), show=False)
    return _close_and_list(plots_dir, "fig3_circuit_types.png")


##############################################
##### Figure 4 - The optimizer performs well across real hardware backends
##############################################
# The constraint model is recomputed per backend.
# @TODO - recompute gadget costs per backend as well; currently only the single-component costs, which
#         compute_constraint_model derives directly from the hardware model, vary across backends

def submit_fig4(runner, group, ctx):
    cfg = ctx.cfg

    for hardware_name in cfg["FIG4_HARDWARE_MODELS"]:
        hardware_model = HARDWARE_MODELS[hardware_name]

        cm_label = f"hardware-{hardware_name}"
        ctx.constraint_models[cm_label] = compute_constraint_model(
            hardware_model, **ctx.qecc,
            effective_threshold=cfg["EFFECTIVE_THRESHOLD"],
            gadget_costs=ctx.gadget_costs,
        )

        submit_comparison(
            runner, group, ctx, sub=hardware_name, circuits=ctx.headline, qecc_name=ctx.qecc_name, cm_label=cm_label,
            schedule_types=comparison_schedule_types,
            n_random_repeats=cfg["FIG3_N_RANDOM_REPEATS"], n_repeats=1,
            variants={"base": hardware_model},
        )


def assemble_fig4(group, ctx):
    return {
        hardware_name: assemble_comparison(group, hardware_name, "base", list(ctx.headline), comparison_schedule_types)
        for hardware_name in ctx.cfg["FIG4_HARDWARE_MODELS"]
    }


def plot_fig4(data, plots_dir):
    # Reuse the code-comparison plotter, which groups by the outer key - here backends rather than codes
    fig = qecc_comparison_bar(data, save=False, show=False)
    fig.axes[0].set_xlabel("Hardware backend")
    fig.savefig(Path(plots_dir) / "fig4_hardware_backends.png", dpi=256)
    return _close_and_list(plots_dir, "fig4_hardware_backends.png")


##############################################
##### Figure 5 - Optimized QEC handles noise scaling better than naive approaches
##############################################
# The constraint model and schedule stay fixed at the unperturbed hardware model while the noise parameter is
# scanned, so the figure measures whether a schedule fitted at one operating point holds up away from it.

def submit_fig5(runner, group, ctx):
    cfg = ctx.cfg

    variants = {
        value: perturb_hardware_model(ctx.hardware_model, cfg["FIG5_NOISE_SCAN_KEY"], value)
        for value in cfg["FIG5_NOISE_SCAN_VALUES"]
    }

    submit_comparison(
        runner, group, ctx, sub="scan", circuits=ctx.headline, qecc_name=ctx.qecc_name, cm_label="base",
        schedule_types=cfg["FIG5_SCHEDULE_TYPES"], n_random_repeats=8, n_repeats=1, variants=variants,
    )


def assemble_fig5(group, ctx):
    cfg = ctx.cfg
    return {
        value: assemble_comparison(group, "scan", value, list(ctx.headline), cfg["FIG5_SCHEDULE_TYPES"])
        for value in cfg["FIG5_NOISE_SCAN_VALUES"]
    }


def plot_fig5(data, plots_dir):
    qec_cycle_schedule_noise_scaling_scatter(data, xlabel="Two-qubit depolarizing error rate", save=True, filename="fig5_noise_scaling.png", save_dir=_save_dir(plots_dir), show=False)
    return _close_and_list(plots_dir, "fig5_noise_scaling.png")


##############################################
##### Figure 6 - Code comparison
##############################################
# One group per QEC code, one bar per scheduling method. Codes whose runs fail are omitted, as in the notebook.

def submit_fig6(runner, group, ctx):
    cfg = ctx.cfg
    circuits = {"Proxy": f"proxy{cfg['PROXY_N_REPEATS']}"}

    for qecc_display_name in cfg["FIG6_QECCS"]:
        qecc_name = QECC_DISPLAY_NAMES[qecc_display_name]

        cm_label = f"qecc-{qecc_name}"
        try:
            ctx.constraint_models[cm_label] = compute_constraint_model(
                ctx.hardware_model, **QECCS[qecc_name],
                effective_threshold=cfg["EFFECTIVE_THRESHOLD"],
                gadget_costs=ctx.gadget_costs,
            )
        except Exception:
            group.failures.append({"key": cm_label, "error": traceback.format_exc()})
            runner.log(f"WARNING - {qecc_display_name} constraint model failed; omitting it from Figure 6")
            continue

        submit_comparison(
            runner, group, ctx, sub=qecc_display_name, circuits=circuits, qecc_name=qecc_name, cm_label=cm_label,
            schedule_types=comparison_schedule_types,
            n_random_repeats=cfg["FIG3_N_RANDOM_REPEATS"], n_repeats=1,
            variants={"base": ctx.hardware_model},
        )


def assemble_fig6(group, ctx):
    data = {}
    for qecc_display_name in ctx.cfg["FIG6_QECCS"]:
        all_data = assemble_comparison(group, qecc_display_name, "base", ["Proxy"], comparison_schedule_types)
        if any(len(circuit_data["methods"]) > 0 for circuit_data in all_data["circuits"].values()):
            data[qecc_display_name] = all_data

    missing = [name for name in ctx.cfg["FIG6_QECCS"] if name not in data]
    if missing:
        group.failures.append({"key": None, "error": f"omitted from Figure 6 because their runs failed: {missing}"})

    return data


def plot_fig6(data, plots_dir):
    qecc_comparison_bar(data, save=True, filename="fig6_qecc_comparison.png", save_dir=_save_dir(plots_dir), show=False)
    return _close_and_list(plots_dir, "fig6_qecc_comparison.png")


##############################################
##### Figure 7 - Is the optimum interior?
##############################################
# Infidelity against the number of QEC cycles for the fixed-rate family (from zero cycles upward), with the
# optimized schedule overlaid at its own cycle count.

def submit_fig7(runner, group, ctx):
    cfg = ctx.cfg
    circuit_label = f"proxy{cfg['PROXY_N_REPEATS']}"
    circuit_no_meas = ctx.circuits[circuit_label].remove_final_measurements(inplace=False)

    # Several intervals can produce the same schedule (e.g., every interval longer than the circuit gives zero
    # cycles), so the family is deduplicated by its cycle indices; zero cycles is always included
    family = {(): []}
    for interval in cfg["FIG7_INTERVALS"]:
        qec_cycle_indices = create_qec_schedules(circuit_no_meas, "fixed", interval=interval)
        family.setdefault(tuple(qec_cycle_indices.get(0, [])), []).append(interval)
    group.results[("family",)] = {indices: intervals for indices, intervals in family.items()}

    def then(schedule):
        for indices in family:
            for r in range(cfg["FIG7_N_REPEATS"]):
                request_logical_run(
                    runner, group, ctx,
                    slot=("run", "fixed", indices, r),
                    key=f"fig7/fixed/cycles{'-'.join(str(i) for i in indices) or 'none'}/draw{r:03d}",
                    circuit_label=circuit_label, qecc=ctx.qecc, qec_cycle_indices={0: list(indices)},
                    hardware_model=ctx.hardware_model, exact_state=schedule["exact_state"],
                )

        for r in range(cfg["FIG7_N_REPEATS"]):
            request_logical_run(
                runner, group, ctx,
                slot=("run", "optimized", "optimized", r),
                key=f"fig7/optimized/draw{r:03d}",
                circuit_label=circuit_label, qecc=ctx.qecc, qec_cycle_indices=schedule["qec_cycle_indices"],
                hardware_model=ctx.hardware_model, exact_state=schedule["exact_state"],
            )

    request_schedule(runner, group, ctx, ("schedule",), circuit_label, ctx.qecc_name, "base", then)


def _summarize_runs(runs):
    if len(runs) == 0:
        return None

    fidelities = [run["fidelity"] for run in runs.values()]
    return {
        "fidelity": float(np.mean(fidelities)),
        "fidelity_std": float(np.std(fidelities)),
        "fidelities": fidelities,
        "num_qec": int(np.mean([run["num_qec"] for run in runs.values()])),
    }


def assemble_fig7(group, ctx):
    family = group.results[("family",)]

    fixed = []
    for indices, intervals in family.items():
        summary = _summarize_runs(pooled_runs(group, ("run", "fixed", indices)))
        if summary is not None:
            fixed.append({"n_cycles": len(indices), "qec_cycle_indices": list(indices), "intervals": intervals, **summary})
    fixed.sort(key=lambda point: point["n_cycles"])

    schedule = group.results.get(("schedule",))
    optimized = _summarize_runs(pooled_runs(group, ("run", "optimized", "optimized")))
    if optimized is not None and schedule is not None:
        optimized["qec_cycle_indices"] = schedule["qec_cycle_indices"]

    return {"fixed": fixed, "optimized": optimized}


def plot_fig7(data, plots_dir):
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")

    fixed = data["fixed"]
    ax.errorbar(
        [point["n_cycles"] for point in fixed], [1.0 - point["fidelity"] for point in fixed],
        yerr=[point["fidelity_std"] for point in fixed], marker="o", capsize=3, color="#D32F2F", label="Fixed-rate family",
    )

    optimized = data["optimized"]
    if optimized is not None:
        ax.errorbar([optimized["num_qec"]], [1.0 - optimized["fidelity"]], yerr=[optimized["fidelity_std"]], fmt="none", ecolor="#2E7D32", capsize=3)
        ax.scatter([optimized["num_qec"]], [1.0 - optimized["fidelity"]], marker="*", s=260, color="#2E7D32", zorder=5, label="Optimized")

    ax.set_xlabel("Number of QEC cycles")
    ax.set_ylabel("Infidelity")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.savefig(Path(plots_dir) / "fig7_interior_optimum.png", dpi=256)
    return _close_and_list(plots_dir, "fig7_interior_optimum.png")


##############################################
##### Figure 8 - Gadget cost differentials, colored by sign
##############################################

def submit_fig8(runner, group, ctx):
    # Needs only the precomputed gadget costs
    pass


def assemble_fig8(group, ctx):
    return {"gadget_costs": ctx.gadget_costs, "depth": ctx.cfg["FIG8_DEPTH"], "n_qubits": ctx.cfg["FIG8_N_QUBITS"]}


def plot_fig8(data, plots_dir):
    gadget_cost_differential_scatter(data["gadget_costs"], depth=data["depth"], n_qubits=data["n_qubits"], save=True, filename="fig8_gadget_cost_differential.png", save_dir=_save_dir(plots_dir), show=False)
    return _close_and_list(plots_dir, "fig8_gadget_cost_differential.png")


##############################################
##### Figure 9 - Does the gadget stage actually earn its cost?
##############################################
# Optimized schedule from additive per-gate costs only, against the gadget-augmented constraint model.

FIG9_CONSTRAINT_MODELS = {"Additive costs only": "additive", "Gadget costs": "base"}
FIG9_SCHEDULE_TYPES = ["no_qec", "optimized"]


def submit_fig9(runner, group, ctx):
    # Additive-only model: strip every multi-component gadget key, keeping single-component and arity costs
    ctx.constraint_models["additive"] = {
        key: value for key, value in ctx.constraint_models["base"].items()
        if not (key.startswith("cost_") and "-" in key)
    }
    runner.log(f"Figure 9: gadget-augmented model has {len(ctx.constraint_models['base'])} keys, additive-only has {len(ctx.constraint_models['additive'])}")

    for label, cm_label in FIG9_CONSTRAINT_MODELS.items():
        submit_comparison(
            runner, group, ctx, sub=label, circuits=ctx.headline, qecc_name=ctx.qecc_name, cm_label=cm_label,
            schedule_types=FIG9_SCHEDULE_TYPES, n_random_repeats=8, n_repeats=1,
            variants={"base": ctx.hardware_model},
        )


def assemble_fig9(group, ctx):
    return {
        label: assemble_comparison(group, label, "base", list(ctx.headline), FIG9_SCHEDULE_TYPES)
        for label in FIG9_CONSTRAINT_MODELS
    }


def plot_fig9(data, plots_dir):
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")

    circuit_names = []
    for all_data in data.values():
        for circuit_name in all_data["circuits"]:
            if circuit_name not in circuit_names:
                circuit_names.append(circuit_name)

    x = np.arange(len(circuit_names))
    width = 0.35

    for i, (label, all_data) in enumerate(data.items()):
        records = [all_data["circuits"].get(name, {}).get("methods", {}).get("optimized", {}) for name in circuit_names]
        infidelities = [1.0 - record.get("fidelity", np.nan) for record in records]
        n_cycles = [record.get("num_qec", "?") for record in records]

        offset = (i - len(data)/2 + 0.5) * width
        bars = ax.bar(x + offset, infidelities, width, label=label, color=colormaps["Set1"](i), alpha=0.85)

        for bar, n in zip(bars, n_cycles):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f"{n} cycles", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Benchmark circuit")
    ax.set_ylabel("Infidelity")
    ax.set_xticks(x)
    ax.set_xticklabels(circuit_names, rotation=20, ha="right")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    fig.savefig(Path(plots_dir) / "fig9_gadget_ablation.png", dpi=256)
    return _close_and_list(plots_dir, "fig9_gadget_ablation.png")


##############################################
##### Figure registry
##############################################

class Figure:
    def __init__(self, fig_id, submit, assemble, plot):
        self.id = fig_id
        self.name = f"fig{fig_id}"
        self.submit = submit
        self.assemble = assemble
        self.plot = plot


def _fig2(which):
    keys = {"low": ("FIG2_LOW_1Q", "FIG2_LOW_2Q", "Low error rates", "fig2a_noise_parameters_low.png"),
            "high": ("FIG2_HIGH_1Q", "FIG2_HIGH_2Q", "High error rates", "fig2b_noise_parameters_high.png")}[which]
    key_1q, key_2q, title, filename = keys

    return (
        lambda runner, group, ctx: submit_fig2(runner, group, ctx, ctx.cfg[key_1q], ctx.cfg[key_2q]),
        lambda group, ctx: assemble_fig2(group, ctx, ctx.cfg[key_1q], ctx.cfg[key_2q], title),
        lambda data, plots_dir: plot_fig2(data, plots_dir, filename),
    )


FIGURES = [
    Figure("1", submit_fig1, assemble_fig1, plot_fig1),
    Figure("2a", *_fig2("low")),
    Figure("2b", *_fig2("high")),
    Figure("3", submit_fig3, assemble_fig3, plot_fig3),
    Figure("4", submit_fig4, assemble_fig4, plot_fig4),
    Figure("5", submit_fig5, assemble_fig5, plot_fig5),
    Figure("6", submit_fig6, assemble_fig6, plot_fig6),
    Figure("7", submit_fig7, assemble_fig7, plot_fig7),
    Figure("8", submit_fig8, assemble_fig8, plot_fig8),
    Figure("9", submit_fig9, assemble_fig9, plot_fig9),
]


def select_figures(spec):
    if spec in [None, "", "all"]:
        return list(FIGURES)

    requested = [item.strip().lower().removeprefix("fig") for item in spec.split(",") if item.strip()]
    selected = [figure for figure in FIGURES if any(figure.id == item or figure.id.rstrip("ab") == item for item in requested)]

    unknown = [item for item in requested if not any(figure.id == item or figure.id.rstrip("ab") == item for figure in FIGURES)]
    if unknown:
        raise SystemExit(f"Unknown figure(s): {unknown}; choose from {[figure.id for figure in FIGURES]}")

    return selected


##############################################
##### Plotting, summary, and main
##############################################

def latex_available():
    return shutil.which("latex") is not None and (shutil.which("dvipng") is not None or shutil.which("gs") is not None)


def configure_matplotlib(use_tex):
    plt.rcParams.update({
        "text.usetex": bool(use_tex),
        "font.size": 16,
        "font.family": "serif",
    })


def plot_with_fallback(figure, data, plots_dir, log):
    """Plot a figure; if LaTeX rendering fails, retry once without it rather than losing the figure."""

    try:
        return figure.plot(data, plots_dir)
    except Exception:
        plt.close("all")
        if not plt.rcParams["text.usetex"]:
            raise

        log(f"WARNING - LaTeX rendering failed for {figure.name}; retrying without usetex:\n{traceback.format_exc()}")
        configure_matplotlib(False)
        try:
            return figure.plot(data, plots_dir)
        finally:
            configure_matplotlib(True)


def finalize_figure(group, figure, ctx, log):
    data = figure.assemble(group, ctx)

    data_path = ctx.data_dir / f"{figure.name}.pkl"
    with open(data_path, "wb") as f:
        pickle.dump(data, f, protocol=5)

    group.outputs = plot_with_fallback(figure, data, ctx.plots_dir, log)


def detect_cpus():
    for variable in ["SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"]:
        value = os.environ.get(variable)
        if value and value.isdigit() and int(value) > 0:
            return int(value), variable

    if hasattr(os, "process_cpu_count") and os.process_cpu_count():
        return os.process_cpu_count(), "os.process_cpu_count()"

    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0)), "sched_getaffinity"

    return os.cpu_count() or 1, "os.cpu_count()"


def make_logger(log_path):
    def log(message):
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    return log


def write_summary(path, runner, cfg, run_info):
    groups = []
    for group in runner.groups:
        groups.append({
            "name": group.name,
            "status": group.status,
            "outputs": group.outputs,
            "task_cpu_hours": group.task_seconds/3600,
            "tasks_run_this_session": group.n_tasks_run,
            "wall_hours": ((group.ended or time.time()) - group.started)/3600,
            "failures": group.failures,
        })

    stats = runner.task_stats
    summary = {
        "run": run_info,
        "config": cfg,
        "groups": groups,
        "tasks": {
            "run_this_session": len(stats),
            "from_checkpoint": runner.n_checkpointed,
            "failed": len(runner.failed),
            "still_pending": len(runner.pending),
            "total_task_cpu_hours": sum(s["elapsed"] for s in stats)/3600,
            "max_peak_rss_mb": max([s["peak_rss_mb"] for s in stats], default=None),
            "slowest": sorted(stats, key=lambda s: -s["elapsed"])[:15],
            "largest_peak_rss": sorted(stats, key=lambda s: -s["peak_rss_mb"])[:15],
        },
    }

    tmp_path = Path(str(path) + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    os.replace(tmp_path, path)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate the paper figures from demos/PaperFigures.ipynb in parallel.")

    parser.add_argument("--fast", action="store_true", help="Shrink every expensive parameter, as FAST_MODE in the notebook")
    parser.add_argument("--heavy", action="store_true", help="Use the Kogut-Susskind circuit (38 encoded qubits with Steane) as the headline workload, as HEAVY_MODE in the notebook")
    parser.add_argument("--figures", default="all", help="Comma-separated figure ids, e.g. '1,3,7'; '2' selects 2a and 2b (default: all)")

    parser.add_argument("--workers", type=int, default=None, help="Worker processes (default: allocated CPUs // --threads-per-worker)")
    parser.add_argument("--threads-per-worker", type=int, default=1, help="OpenMP threads per worker, used by Aer for parallel shots (default: 1)")
    parser.add_argument("--shots", type=int, default=None, help="Shots per circuit (default: 1e3 with --fast, else 1e5)")
    parser.add_argument("--shots-per-task", type=int, default=5000, help="Logical runs are split into tasks of at most this many shots, then pooled (default: 5000)")
    parser.add_argument("--method", default="auto", choices=["auto", "statevector", "matrix_product_state", "density_matrix"], help=f"Aer simulation method (default: auto, i.e. statevector up to {AUTO_STATEVECTOR_MAX_QUBITS} encoded qubits and MPS above)")
    parser.add_argument("--fidelity-method", default="logical_counts", choices=["logical_counts", "density_matrix"], help="Scoring, as FIDELITY_METHOD in the notebook (default: logical_counts)")
    parser.add_argument("--seed", type=int, default=1234, help="Seed for random schedules, the gadget library, and the mirror circuit (default: 1234)")

    parser.add_argument("--output-dir", default=None, help="Output directory (default: ./paper_figures_<timestamp>); pass an existing one to resume")
    parser.add_argument("--cache-dir", default=str(Path(__file__).resolve().parent / "data"), help="Where gadget costs are cached across runs (default: demos/data)")
    parser.add_argument("--start-method", default=None, choices=["forkserver", "spawn", "fork"], help="Multiprocessing start method (default: forkserver on Linux, else spawn)")
    parser.add_argument("--plot-only", action="store_true", help="Re-plot from the data/ directory of --output-dir without running anything")

    return parser.parse_args()


def plot_only(args):
    if args.output_dir is None:
        raise SystemExit("--plot-only needs --output-dir")

    out_dir = Path(args.output_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(out_dir / "run.log")

    configure_matplotlib(latex_available())
    for figure in select_figures(args.figures):
        data_path = out_dir / "data" / f"{figure.name}.pkl"
        if not data_path.exists():
            log(f"{figure.name}: no data at {data_path}, skipping")
            continue

        with open(data_path, "rb") as f:
            data = pickle.load(f)

        try:
            outputs = plot_with_fallback(figure, data, plots_dir, log)
            log(f"{figure.name}: wrote {', '.join(outputs)}")
        except Exception:
            log(f"ERROR - plotting {figure.name} failed:\n{traceback.format_exc()}")


def main():
    args = parse_args()

    if args.plot_only:
        plot_only(args)
        return

    out_dir = Path(args.output_dir or f"paper_figures_{datetime.now():%Y-%m-%d_%H-%M-%S}").resolve()
    for sub_dir in ["plots", "data", "logs"]:
        (out_dir / sub_dir).mkdir(parents=True, exist_ok=True)

    log = make_logger(out_dir / "run.log")

    cfg = build_config(fast=args.fast, heavy=args.heavy, shots=args.shots, method=args.method, fidelity_method=args.fidelity_method, seed=args.seed)
    # Normalize through JSON so that it compares equal to a stored run_config.json
    cfg = json.loads(json.dumps(cfg))

    config_path = out_dir / "run_config.json"
    if config_path.exists():
        with open(config_path) as f:
            previous_cfg = json.load(f)
        if previous_cfg != cfg:
            changed = sorted(key for key in set(cfg) | set(previous_cfg) if cfg.get(key) != previous_cfg.get(key))
            raise SystemExit(f"{out_dir} holds results from a different configuration (differs in {changed}); use a new --output-dir, or rerun with the original options to resume")
        log(f"Resuming run in {out_dir}")
    else:
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)

    n_cpus, cpu_source = detect_cpus()
    threads_per_worker = max(1, args.threads_per_worker)
    n_workers = args.workers or max(1, n_cpus // threads_per_worker)

    start_method = args.start_method or ("forkserver" if sys.platform.startswith("linux") else "spawn")
    mp_context = multiprocessing.get_context(start_method)

    use_tex = latex_available()
    configure_matplotlib(use_tex)

    run_info = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "qiskit": qiskit.__version__,
        "qiskit_aer": qiskit_aer.__version__,
        "cpus": n_cpus,
        "cpu_source": cpu_source,
        "workers": n_workers,
        "threads_per_worker": threads_per_worker,
        "start_method": start_method,
        "shots_per_task": args.shots_per_task,
        "usetex": use_tex,
        "figures": [figure.id for figure in select_figures(args.figures)],
    }

    log(f"Output directory: {out_dir}")
    log(f"{n_cpus} CPUs (from {cpu_source}) -> {n_workers} workers x {threads_per_worker} thread(s), start method {start_method}")
    log(f"Mode: {'fast' if cfg['FAST_MODE'] else 'full'}, {'heavy (Kogut-Susskind)' if cfg['HEAVY_MODE'] else 'proxy'} headline, {cfg['SHOTS']} shots in tasks of <= {args.shots_per_task}, method {cfg['METHOD']}, scoring {cfg['FIDELITY_METHOD']}")
    if not use_tex:
        log("LaTeX not found; plotting without usetex")

    ctx = Context(cfg, out_dir, Path(args.cache_dir), args.shots_per_task)

    # Report the encoded size of the headline workload before launching anything, so nothing silently OOMs
    for name, label in list(ctx.headline.items()) + [("Kogut-Susskind (reference)", None)]:
        n_physical = ctx.circuits[label].num_qubits if label is not None else 3
        n_encoded = LogicalCircuit(n_physical, **ctx.qecc).num_qubits
        log(f"{name}: {n_physical} physical -> {n_encoded} encoded qubits | statevector {2**n_encoded * 16 / 2**30:.4g} GB | method {resolve_method(cfg['METHOD'], n_encoded)}")

    summary_path = out_dir / "summary.json"
    runner = None

    def on_group_finished():
        write_summary(summary_path, runner, cfg, run_info)

    runner = Runner(n_workers, mp_context, out_dir, log, on_group_finished=on_group_finished)
    if runner.n_checkpointed:
        log(f"Loaded {runner.n_checkpointed} completed tasks from the checkpoint")

    # SIGTERM (e.g., a scheduler's walltime limit) is handled like Ctrl-C so that the summary is still written
    def handle_sigterm(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, handle_sigterm)

    def start_figures():
        for figure in select_figures(args.figures):
            group = Group(figure.name, finalize=lambda group, figure=figure: finalize_figure(group, figure, ctx, log))
            runner.start_group(group, submit=lambda group, figure=figure: figure.submit(runner, group, ctx))

    precompute = Group("precompute", finalize=lambda group: finalize_precompute(group, ctx, runner, start_figures))
    runner.start_group(precompute, submit=lambda group: submit_precompute(runner, group, ctx))

    interrupted = False
    try:
        runner.run()
    except KeyboardInterrupt:
        interrupted = True
        log(f"Interrupted. Completed tasks are checkpointed; rerun with --output-dir {out_dir} and the same options to resume")
    finally:
        run_info["ended"] = datetime.now().isoformat(timespec="seconds")
        run_info["interrupted"] = interrupted
        write_summary(summary_path, runner, cfg, run_info)

    statuses = {group.name: group.status for group in runner.groups}
    log(f"Done. Figure status: {statuses}")
    log(f"Plots: {ctx.plots_dir}")
    log(f"Summary: {summary_path}")

    if interrupted:
        sys.exit(130)
    if any(status != "complete" for status in statuses.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
