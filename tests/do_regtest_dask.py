#!/usr/bin/env python3

# Dask-based CP2K regression test runner.
#
# Standalone alternative to do_regtest.py: instead of an asyncio Semaphore
# worker pool in one process, batches (test directories) are distributed as
# Dask tasks across a local or externally-managed Dask cluster. Each batch
# runs its tests with plain blocking subprocess calls -- there is no
# keepalive/persistent-shell mode, so no async code is needed at all.
#
# Rough pipeline, see main() at the bottom for the exact sequence:
#   1. Parse command line arguments into a Config object.
#   2. Ask the cp2k binary for its version and feature flags.
#   3. Build one Batch per UNIT_TESTS line and per TEST_DIRS directory.
#   4. Start a Dask cluster and submit each Batch as one task (run_batch).
#   5. Collect results as they complete, print them, and write a summary.

from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, TextIO, Tuple, Union
import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import time

from distributed import Client, Future, LocalCluster, as_completed
from matchers import run_matcher
from regtest_requirements import requirements_satisfied

# Try importing toml from various places.
try:
    import tomllib  # not available before Python 3.11
except ImportError:
    try:
        import pip._vendor.tomli as tomllib  # type: ignore
    except ImportError:
        try:
            import pip._vendor.toml as tomllib  # type: ignore
        except ImportError:
            import toml as tomllib  # type: ignore

# A test's status is always one of: "OK", "WRONG RESULT", "RUNTIME FAIL",
# "TIMED OUT", "HUGE OUTPUT", "N/A".
TestStatus = str


# ======================================================================================
# Small standalone helpers, used by Config and Batch below.
# ======================================================================================
def _is_intel_mpi(mpiexec_cmd: str = "mpiexec") -> bool:
    """Check if the given mpiexec command belongs to Intel MPI."""
    try:
        result = subprocess.run(
            [mpiexec_cmd, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return "Intel" in result.stdout or "Intel" in result.stderr
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _run_shell_capture_stdout(shell_command: str) -> bytes:
    """Run a shell command and return whatever it printed on stdout."""
    result = subprocess.run(
        shell_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    return result.stdout


def _detect_num_gpus() -> int:
    nvidia_command = "nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l"
    num_nvidia_gpus = int(_run_shell_capture_stdout(nvidia_command))
    amd_command = "rocm-smi --showid --csv | grep card | wc -l"
    num_amd_gpus = int(_run_shell_capture_stdout(amd_command))
    return num_nvidia_gpus + num_amd_gpus


def _find_cp2k_data_dir(cp2kdatadir: Optional[Path], cp2k_root: Path) -> Path:
    if cp2kdatadir is not None:
        return cp2kdatadir.resolve()
    default_data_dir = str(cp2k_root / "data")
    return Path(os.getenv("CP2K_DATA_DIR", default_data_dir)).resolve()


def _build_mpiexec_template(mpiexec: str) -> str:
    """Make sure the template contains a "{N}" placeholder for the rank count."""
    if "{N}" in mpiexec:
        return mpiexec
    # Old command lines omit the placeholder. Insert "-n {N}" right after the
    # launcher command, e.g. "mpiexec --bind-to none" -> "mpiexec -n {N} --bind-to none".
    words = mpiexec.split(" ", 1)
    launcher_command = words[0]
    remaining_arguments = words[1] if len(words) > 1 else ""
    return f"{launcher_command} -n {{N}} {remaining_arguments}".strip()


def _name_matches_any_pattern(name: str, patterns: List[str]) -> bool:
    for pattern in patterns:
        if re.fullmatch(pattern, name):
            return True
    return False


# ======================================================================================
class Config:
    def __init__(self, args: argparse.Namespace):
        self.timeout = args.timeout
        self.use_mpi = args.version.startswith("p")
        default_ompthreads = 2 if "smp" in args.version else 1
        self.ompthreads = args.ompthreads if args.ompthreads else default_ompthreads
        self.mpiranks = args.mpiranks if self.use_mpi else 1
        self.num_workers = int(args.maxtasks / self.ompthreads / self.mpiranks) or 1
        self.cp2k_root = Path(__file__).resolve().parent.parent
        self.mpiexec = _build_mpiexec_template(args.mpiexec)
        self.intel_mpi = _is_intel_mpi(self.mpiexec.split()[0])
        if self.intel_mpi and "--bind-to" in self.mpiexec:
            self.mpiexec = self.mpiexec.replace(" --bind-to none", "")
        self.smoketest = args.smoketest
        self.valgrind = args.valgrind
        self.flag_slow = args.flagslow
        self.binary_dir = args.binary_dir.resolve()
        self.version = args.version
        self.debug = args.debug
        self.max_errors = args.maxerrors
        self.restrictdirs = args.restrictdir if args.restrictdir else [".*"]
        self.skipdirs = args.skipdir if args.skipdir else []
        self.skip_unittests = args.skip_unittests
        self.skip_regtests = args.skip_regtests
        self.scheduler_file = args.scheduler_file
        datestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.work_base_dir = args.workbasedir.resolve() / f"TEST-{datestamp}"
        self.error_summary = self.work_base_dir / "error_summary"
        self.cp2k_data_dir = _find_cp2k_data_dir(args.cp2kdatadir, self.cp2k_root)

        # Parse suppression files.
        slow_supps_fn = self.cp2k_root / "tests" / "SLOW_TESTS_SUPPRESSIONS"
        self.slow_suppressions = slow_supps_fn.read_text(encoding="utf8").split("\n")
        huge_supps_fn = self.cp2k_root / "tests" / "HUGE_TESTS_SUPPRESSIONS"
        self.huge_suppressions = huge_supps_fn.read_text(encoding="utf8").split("\n")

        # Detect number of GPU devices, if not specified by the user.
        if args.num_gpus > 0:
            self.num_gpus = args.num_gpus
        else:
            self.num_gpus = _detect_num_gpus()

    def build_command(
        self, exe_stem: str, *args: str, cwd: Optional[Path] = None, gpu_offset: int = 0
    ) -> Tuple[List[str], Dict[str, str]]:
        env = os.environ.copy()
        if self.num_gpus > self.mpiranks:
            visible_gpu_devices = []
            for rank in range(self.mpiranks):
                device_index = (gpu_offset + rank) % self.num_gpus
                visible_gpu_devices.append(str(device_index))
            env["CUDA_VISIBLE_DEVICES"] = ",".join(visible_gpu_devices)
            env["HIP_VISIBLE_DEVICES"] = ",".join(visible_gpu_devices)
        env["OMP_NUM_THREADS"] = str(self.ompthreads)
        if self.intel_mpi:
            env["I_MPI_PIN"] = "0"
        env["CP2K_DATA_DIR"] = str(self.cp2k_data_dir)
        env["PIKA_COMMANDLINE_OPTIONS"] = (
            f"--pika:bind=none --pika:threads={self.ompthreads}"
        )
        if cwd is not None and cwd.parent.name == "MIMIC":
            env["MCL_COMM_MODE"] = "TEST_STUB"
            env["MCL_PROGRAM"] = "1"
            env["MCL_TEST_DATA"] = "MCL_LOG_1"

        exe_name = f"{exe_stem}.{self.version}"
        cmd = [str(self.binary_dir / exe_name)]
        if self.valgrind:
            cmd = ["valgrind", "--error-exitcode=42", "--exit-on-first-error=yes"] + cmd
        if self.use_mpi:
            cmd = self.mpiexec.format(N=self.mpiranks).split() + cmd
        cmd = cmd + list(args)

        if self.debug:
            print(f"Running subprocess: {cmd}")
        return cmd, env


# ======================================================================================
class Unittest:
    """A unit test, ie. a standalone binary that matches '*_unittest.{cfg.version}'."""

    def __init__(self, name: str, workdir: Path):
        self.name = name
        self.matcher_specs: List[Any] = []
        self.out_path = workdir / (self.name + ".out")


# ======================================================================================
class Regtest:
    """A single input file to test, ie. a line in a TEST_FILES file."""

    def __init__(self, inp_fn: str, matcher_specs: List[Any], workdir: Path):
        self.inp_fn = inp_fn
        self.name = self.inp_fn
        self.matcher_specs = matcher_specs
        self.out_path = workdir / (self.name + ".out")


# ======================================================================================
class Batch:
    """A directory of tests, ie. a line in the TEST_DIRS file."""

    def __init__(self, line: str, cfg: Config, index: int):
        parts = line.split()
        self.name = parts[0]
        self.requirements = parts[1:]
        self.unittests: List[Unittest] = []
        self.regtests: List[Regtest] = []
        self.src_dir = cfg.cp2k_root / "tests" / self.name
        self.workdir = cfg.work_base_dir / self.name
        self.huge_suppressions = cfg.huge_suppressions
        # Deterministic per-batch GPU pinning, computed once up front so no
        # mutable state needs to be shared across Dask worker processes.
        if cfg.num_gpus > 0:
            self.gpu_offset = index % cfg.num_gpus
        else:
            self.gpu_offset = 0

    def requirements_satisfied(self, flags: List[str], mpiranks: int) -> bool:
        return requirements_satisfied(self.requirements, flags, mpiranks)


# ======================================================================================
class TestResult:
    def __init__(
        self,
        batch: Batch,
        test: Union[Regtest, Unittest],
        spec: Optional[Dict[str, Any]],
        duration: float,
        status: TestStatus,
        error: Optional[str] = None,
        value: Optional[float] = None,
    ):
        self.batch = batch
        self.test = test
        self.spec = spec
        self.duration = duration
        self.status = status
        self.error = error
        self.value = value
        self.fullname = f"{batch.name}/{test.name}"

    def __str__(self) -> str:
        display_name = self.test.name
        if self.spec and len(self.test.matcher_specs) > 1:
            display_name += f":{self.spec.get('matcher', '???')}"
        if self.value:
            value_text = f"{self.value:.10g}"
        else:
            value_text = "-"
        return (
            f"    {display_name :<80s} {value_text :>17} "
            f"{self.status :>12s} ( {self.duration:6.2f} sec)"
        )


# ======================================================================================
class BatchResult:
    def __init__(self, batch: Batch, results: List[TestResult]):
        self.batch = batch
        self.results = results
        self.duration = sum(float(r.duration) for r in results)


# ======================================================================================
# Functions that actually run tests. These execute on Dask workers.
# ======================================================================================
def run_child_process(
    cmd: List[str], env: Dict[str, str], cwd: Optional[Path], timeout: int
) -> Tuple[bytes, int, bool]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return proc.stdout, proc.returncode, False
    except subprocess.TimeoutExpired as exc:
        return (exc.output or b""), -9, True


def run_batch(batch: Batch, cfg: Config) -> BatchResult:
    """Run every unittest and regtest of one Batch. This is the Dask task function."""
    results = []
    if not cfg.skip_unittests:
        results += run_unittests(batch, cfg)
    if not cfg.skip_regtests:
        results += run_regtests(batch, cfg)
    return BatchResult(batch, results)


def run_unittests(batch: Batch, cfg: Config) -> List[TestResult]:
    results: List[TestResult] = []
    for test in batch.unittests:
        start_time = time.perf_counter()
        cmd, env = cfg.build_command(
            test.name, str(cfg.cp2k_root), cwd=batch.workdir, gpu_offset=batch.gpu_offset
        )
        output, returncode, timed_out = run_child_process(
            cmd, env, batch.workdir, cfg.timeout
        )
        duration = time.perf_counter() - start_time
        test.out_path.write_bytes(output)
        output_lines = output.decode("utf8", errors="replace").split("\n")
        output_tail = "\n".join(output_lines[-100:])
        error = "x" * 100 + f"\n{test.out_path}\n{output_tail}\n\n"
        if timed_out:
            error += f"Timed out after {duration} seconds."
            results.append(TestResult(batch, test, None, duration, "TIMED OUT", error))
        elif returncode != 0:
            error += f"Runtime failure with code {returncode}."
            results.append(TestResult(batch, test, None, duration, "RUNTIME FAIL", error))
        else:
            results.append(TestResult(batch, test, None, duration, "OK"))

    return results


def run_regtests(batch: Batch, cfg: Config) -> List[TestResult]:
    results: List[TestResult] = []
    for test in batch.regtests:
        start_time = time.perf_counter()
        start_dirsize = dirsize(batch.workdir)
        cmd, env = cfg.build_command(
            "cp2k", test.inp_fn, cwd=batch.workdir, gpu_offset=batch.gpu_offset
        )
        output, returncode, timed_out = run_child_process(
            cmd, env, batch.workdir, cfg.timeout
        )
        test.out_path.write_bytes(output)
        duration = time.perf_counter() - start_time
        output_size = dirsize(batch.workdir) - start_dirsize
        results += eval_regtest(
            batch, test, duration, output_size, returncode, timed_out
        )

    return results


def dirsize(folder: Path) -> int:
    total = 0
    for path in folder.rglob("*"):
        total += path.stat().st_size
    return total


def eval_regtest(
    batch: Batch,
    test: Regtest,
    duration: float,
    output_size: int,
    returncode: int,
    timed_out: bool,
) -> List[TestResult]:
    is_huge_suppressed = f"{batch.name}/{test.name}" in batch.huge_suppressions
    if test.out_path.exists():
        output_bytes = test.out_path.read_bytes()
    else:
        output_bytes = b""
    output = output_bytes.decode("utf8", errors="replace")
    output_tail = "\n".join(output.split("\n")[-100:])
    error = "x" * 100 + f"\n{test.out_path}\n"

    # check for timeout
    if timed_out:
        error += f"{output_tail}\n\nTimed out after {duration} seconds."
        return [TestResult(batch, test, None, duration, "TIMED OUT", error)]

    # check for crash
    if returncode != 0:
        error += f"{output_tail}\n\nRuntime failure with code {returncode}."
        return [TestResult(batch, test, None, duration, "RUNTIME FAIL", error)]

    # check for huge output
    if output_size > 2 * 1024 * 1024 and not is_huge_suppressed:  # 2 MiB limit
        error += f"Test produced {output_size / 1024 / 1024:.2f} MiB of output."
        return [TestResult(batch, test, None, duration, "HUGE OUTPUT", error)]

    # happy end if there are no matchers
    if not test.matcher_specs:
        return [TestResult(batch, test, None, duration, "OK")]

    # run the matchers
    results: List[TestResult] = []
    for spec in test.matcher_specs:
        spec = dict(spec)  # copy so popping "file" below does not change the original
        alt_file = spec.pop("file", None)
        if alt_file:
            alt_path = test.out_path.parent / alt_file
            if not alt_path.exists():
                message = (
                    f"{error}Spec: {spec}\nExpected output file not found: {alt_path}"
                )
                results.append(
                    TestResult(batch, test, spec, duration, "WRONG RESULT", message)
                )
                continue
            match_output = alt_path.read_bytes().decode("utf8", errors="replace")
        else:
            match_output = output
        matcher_result = run_matcher(match_output, **spec)
        if matcher_result.error:
            matcher_result.error = f"{error}Spec: {spec}\n{matcher_result.error}"
        results.append(
            TestResult(
                batch,
                test,
                spec,
                duration,
                matcher_result.status,
                matcher_result.error,
                matcher_result.value,
            )
        )

    return results


def percentile(values: List[float], percent: float) -> float:
    k = (len(values) - 1) * percent
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    d0 = values[int(f)] * (c - k)
    d1 = values[int(c)] * (k - f)
    return d0 + d1


def is_relative_to(p: Path, u: Path) -> bool:  # not in pathlib before Python 3.9
    return u == p or u in p.parents


# ======================================================================================
# main() and the steps it goes through, in order.
# ======================================================================================
def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Runs CP2K regression test suite via a Dask cluster."
    )
    parser.add_argument("--mpiranks", type=int, default=2)
    parser.add_argument("--ompthreads", type=int)
    parser.add_argument("--maxtasks", type=int, default=os.cpu_count())
    parser.add_argument("--num_gpus", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=400)
    parser.add_argument("--maxerrors", type=int, default=50)
    help_text = "Template for launching MPI jobs, {N} is replaced by number of processors."
    parser.add_argument(
        "--mpiexec", default="mpiexec -n {N} --bind-to none", help=help_text
    )
    help_text = "Runs only the first test of each directory."
    parser.add_argument("--smoketest", action="store_true", help=help_text)
    help_text = "Runs tests under Valgrind memcheck."
    parser.add_argument("--valgrind", action="store_true", help=help_text)
    help_text = "Flag slow tests in the final summary and status report."
    parser.add_argument("--flagslow", action="store_true", help=help_text)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--restrictdir", action="append")
    parser.add_argument("--skipdir", action="append")
    parser.add_argument("--workbasedir", type=Path, default=Path.cwd() / "regtesting")
    parser.add_argument("--cp2kdatadir", type=Path)
    parser.add_argument("--skip_unittests", action="store_true")
    parser.add_argument("--skip_regtests", action="store_true")
    help_text = (
        "Connect to an existing Dask scheduler instead of starting a local cluster. "
        "Typically produced by running, on a login/head node inside your job "
        "allocation: `dask scheduler --scheduler-file scheduler.json`, then on each "
        "worker node: `dask worker --scheduler-file scheduler.json`. Example: "
        "--scheduler-file scheduler.json"
    )
    parser.add_argument("--scheduler-file", type=Path, help=help_text)
    parser.add_argument("binary_dir", type=Path)
    parser.add_argument("version")
    return parser


def query_cp2k_feature_flags(cfg: Config) -> List[str]:
    version_cmd, version_env = cfg.build_command("cp2k", "--version")
    result = subprocess.run(version_cmd, env=version_env, capture_output=True)
    version_output = result.stdout.decode("utf8", errors="replace")
    flags_line = re.search(r" cp2kflags:(.*)\n", version_output)
    if not flags_line:
        print(version_output + "\nCould not parse feature flags.")
        sys.exit(1)
    return flags_line.group(1).split()


def print_settings(cfg: Config, flags: List[str]) -> None:
    print("\n----------------------------- Settings ---------------------------------")
    print(f"MPI ranks:      {cfg.mpiranks}")
    print(f"OpenMP threads: {cfg.ompthreads}")
    print(f"GPU devices:    {cfg.num_gpus}")
    print(f"Workers:        {cfg.num_workers}")
    print(f"Timeout [s]:    {cfg.timeout}")
    print(f"Work base dir:  {cfg.work_base_dir}")
    print(f"MPI exec:       {cfg.mpiexec}")
    print(f"Smoke test:     {cfg.smoketest}")
    print(f"Valgrind:       {cfg.valgrind}")
    print(f"Flag slow:      {cfg.flag_slow}")
    print(f"Debug:          {cfg.debug}")
    print(f"Binary dir:     {cfg.binary_dir}")
    print(f"CP2K data dir:  {cfg.cp2k_data_dir}")
    print(f"VERSION:        {cfg.version}")
    print("Flags:          " + ",".join(flags))


def copy_test_files(cfg: Config) -> None:
    # Have to copy everything upfront because the test dirs are not self-contained.
    print("------------------------------------------------------------------------")
    if is_relative_to(cfg.work_base_dir, cfg.cp2k_root / "tests"):
        print("Error: Work base dir must not be relative to cp2k/tests dir.")
        sys.exit(1)
    print("Copying test files ... ", end="")
    shutil.copytree(cfg.cp2k_root / "tests", cfg.work_base_dir)
    print("done")


def build_all_batches(cfg: Config) -> List[Batch]:
    batches: List[Batch] = []

    # Read UNIT_TESTS: one batch per line, each running a single standalone binary.
    unit_tests_fn = cfg.cp2k_root / "tests" / "UNIT_TESTS"
    for line in unit_tests_fn.read_text(encoding="utf8").split("\n"):
        line = line.split("#", 1)[0].strip()
        if line:
            batch = Batch(f"UNIT/{line}", cfg, len(batches))
            batch.workdir.mkdir(parents=True)
            batch.unittests.append(Unittest(line.split()[0], batch.workdir))
            batches.append(batch)

    # Read TEST_DIRS: one batch per directory, each running many input files.
    test_dirs_fn = cfg.cp2k_root / "tests" / "TEST_DIRS"
    for line in test_dirs_fn.read_text(encoding="utf8").split("\n"):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        batch = Batch(line, cfg, len(batches))

        test_files_fn = batch.src_dir / "TEST_FILES.toml"
        test_files_content = test_files_fn.read_text(encoding="utf8")
        for inp_fn, matcher_specs in tomllib.loads(test_files_content).items():
            batch.regtests.append(Regtest(inp_fn, matcher_specs, batch.workdir))
            if cfg.smoketest:
                break  # run only the first test of each directory
        batches.append(batch)

    _check_for_nested_test_dirs(batches)
    return batches


def _check_for_nested_test_dirs(batches: List[Batch]) -> None:
    for batch_a in batches:
        for batch_b in batches:
            if batch_a != batch_b and is_relative_to(batch_a.workdir, batch_b.workdir):
                print(f"Error: Test dirs {batch_a.name} and {batch_b.name} are nested.")
                sys.exit(1)


def select_batches(batches: List[Batch], cfg: Config, flags: List[str]) -> List[Batch]:
    selected_batches: List[Batch] = []
    num_restrictdirs = 0
    num_skipdirs = 0
    for batch in batches:
        if not batch.requirements_satisfied(flags, cfg.mpiranks):
            print(f"Skipping {batch.name} because its requirements are not satisfied.")
        elif not _name_matches_any_pattern(batch.name, cfg.restrictdirs):
            num_restrictdirs += 1
        elif _name_matches_any_pattern(batch.name, cfg.skipdirs):
            num_skipdirs += 1
        else:
            selected_batches.append(batch)

    if num_restrictdirs:
        print(f"Skipping {num_restrictdirs} test directories because of --restrictdir.")
    if num_skipdirs:
        print(f"Skipping {num_skipdirs} test directories because of --skipdir.")
    if not selected_batches:
        print("\nNo test directories selected, check --restrictdir filter.")
        sys.exit(1)
    return selected_batches


def start_dask_cluster(cfg: Config) -> Tuple[Client, Optional[LocalCluster]]:
    if cfg.scheduler_file:
        client = Client(scheduler_file=str(cfg.scheduler_file))
        return client, None
    cluster = LocalCluster(
        n_workers=cfg.num_workers, threads_per_worker=1, processes=True
    )
    client = Client(cluster)
    return client, cluster


def run_all_batches(
    client: Client, cfg: Config, cfg_future: Future, selected_batches: List[Batch]
) -> List[TestResult]:
    print(f"Launched {len(selected_batches)} test directories on ", end="")
    print(f"{cfg.num_workers} Dask worker(s)...\n")
    sys.stdout.flush()

    all_results: List[TestResult] = []
    with open(cfg.error_summary, "wt", encoding="utf8", errors="replace") as err_fh:
        futures = client.map(run_batch, selected_batches, cfg=cfg_future)
        for num_done, future in enumerate(as_completed(futures)):
            batch_result: BatchResult = future.result()
            all_results += batch_result.results
            _print_batch_result(batch_result, num_done, len(selected_batches))
            _write_batch_errors(err_fh, batch_result)
            if _count_failures(all_results) > cfg.max_errors:
                print(f"\nGot more than {cfg.max_errors} errors, aborting...")
                break
    return all_results


def _print_batch_result(batch_result: BatchResult, num_done: int, num_total: int) -> None:
    print(f">>> {batch_result.batch.workdir}")
    for result in batch_result.results:
        print(str(result))
    print(
        f"<<< {batch_result.batch.workdir} ({num_done + 1} of {num_total}) "
        f"done in {batch_result.duration:.2f} sec"
    )
    sys.stdout.flush()


def _write_batch_errors(err_fh: TextIO, batch_result: BatchResult) -> None:
    error_texts = []
    for result in batch_result.results:
        if result.error:
            error_texts.append(result.error)
    err_fh.write("\n".join(error_texts))
    err_fh.flush()


def _count_failures(results: List[TestResult]) -> int:
    num_failures = 0
    for result in results:
        if result.status != "OK":
            num_failures += 1
    return num_failures


def print_errors_section(all_results: List[TestResult]) -> None:
    print("------------------------------- Errors ---------------------------------")
    for result in all_results:
        if result.error:
            print(result.error)


def print_timings_section(all_results: List[TestResult]) -> List[float]:
    print("\n------------------------------- Timings --------------------------------")
    durations = []
    for result in all_results:
        durations.append(result.duration)
    timings = sorted(durations)

    print('Plot: name="timings", title="Timing Distribution", ylabel="time [s]"')
    for percent in (100, 99, 98, 95, 90, 80):
        y = percentile(timings, percent / 100.0)
        print(
            f'PlotPoint: name="{percent}th_percentile", plot="timings", '
            f'label="{percent}th %ile", y={y:.2f}, yerr=0.0'
        )
    return timings


def find_slow_tests(
    client: Client,
    cfg: Config,
    cfg_future: Future,
    all_results: List[TestResult],
    timings: List[float],
) -> Dict[str, List[float]]:
    print("\n" + "-" * 15 + "--------------- Slow Tests ---------------" + "-" * 15)
    threshold = 2 * percentile(timings, 0.95)

    outliers = []
    for result in all_results:
        if result.duration > 0.95 * threshold:
            outliers.append(result)

    maybe_slow = []
    for result in outliers:
        if result.fullname not in cfg.slow_suppressions:
            maybe_slow.append(result)
    num_suppressed = len(outliers) - len(maybe_slow)

    rerun_batches = []
    for result in maybe_slow:
        if result.batch not in rerun_batches:
            rerun_batches.append(result.batch)
    for batch in rerun_batches:
        print(f"Re-running {batch.name} to avoid false positives.")

    rerun_futures = client.map(run_batch, rerun_batches, cfg=cfg_future)
    rerun_times: Dict[str, float] = {}
    for rerun_result in client.gather(rerun_futures):
        for result in rerun_result.results:
            rerun_times[result.fullname] = result.duration

    slow_tests: Dict[str, List[float]] = {}
    for result in maybe_slow:
        durations = [result.duration, rerun_times[result.fullname]]
        if mean(durations) - stdev(durations) > threshold:
            slow_tests[result.fullname] = durations

    print(f"Duration threshold (2x 95th %ile): {threshold:.2f} sec")
    print(f"Found {len(slow_tests)} slow tests ({num_suppressed} suppressed):")
    for fullname, durations in slow_tests.items():
        print(f"    {fullname :<80s} ( {mean(durations):6.2f} ±{stdev(durations):4.2f} sec)")

    return slow_tests


def print_summary(
    cfg: Config,
    all_results: List[TestResult],
    slow_tests: Dict[str, List[float]],
    start_time: float,
) -> bool:
    print("\n------------------------------- Summary --------------------------------")
    total_duration = time.perf_counter() - start_time
    num_tests = len(all_results)
    failure_modes = ["RUNTIME FAIL", "TIMED OUT", "HUGE OUTPUT"]

    num_failed = 0
    num_wrong = 0
    num_na = 0
    num_ok = 0
    for result in all_results:
        if result.status in failure_modes:
            num_failed += 1
        elif result.status == "WRONG RESULT":
            num_wrong += 1
        elif result.status == "N/A":
            num_na += 1
        elif result.status == "OK":
            num_ok += 1

    status_ok = (num_ok == num_tests) and (not cfg.flag_slow or not slow_tests)
    print(f"Number of FAILED  tests {num_failed}")
    print(f"Number of WRONG   tests {num_wrong}")
    print(f"Number of CORRECT tests {num_ok}")
    print(f"Total number of   tests {num_tests}")

    summary = f"\nSummary: correct: {num_ok} / {num_tests}"
    if num_wrong > 0:
        summary += f"; wrong: {num_wrong}"
    if num_failed > 0:
        summary += f"; failed: {num_failed}"
    if num_na > 0:
        summary += f"; n/a: {num_na}"
    if cfg.flag_slow and slow_tests:
        summary += f"; slow: {len(slow_tests)}"
    summary += f"; {total_duration / 60.0:.0f}min"
    print(summary)

    if status_ok:
        print("Status: OK\n")
    else:
        print("Status: FAILED\n")

    return status_ok


def main() -> None:
    cfg = Config(build_argument_parser().parse_args())

    print("*************************** Testing started ****************************")
    start_time = time.perf_counter()

    flags = query_cp2k_feature_flags(cfg)
    print_settings(cfg, flags)
    copy_test_files(cfg)

    batches = build_all_batches(cfg)
    selected_batches = select_batches(batches, cfg, flags)

    client, cluster = start_dask_cluster(cfg)
    try:
        cfg_future = client.scatter(cfg, broadcast=True)
        all_results = run_all_batches(client, cfg, cfg_future, selected_batches)
        print_errors_section(all_results)
        timings = print_timings_section(all_results)
        slow_tests: Dict[str, List[float]] = {}
        if cfg.flag_slow:
            slow_tests = find_slow_tests(client, cfg, cfg_future, all_results, timings)
    finally:
        client.close()
        if cluster is not None:
            cluster.close()

    status_ok = print_summary(cfg, all_results, slow_tests, start_time)

    print("*************************** Testing ended ******************************")
    sys.exit(0 if status_ok else 1)


if __name__ == "__main__":
    main()

# EOF
