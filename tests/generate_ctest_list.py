#!/usr/bin/env python3

# Run once by tests/CMakeLists.txt via execute_process() at CMake configure
# time. Reads UNIT_TESTS, TEST_DIRS, and every TEST_FILES.toml, and writes
# out a plain .cmake file containing one add_test()/set_tests_properties()
# pair per unit test and per individual regtest input file.
#
# Deliberately standalone (no imports from do_regtest.py or
# do_regtest_dask.py): this only needs to read a handful of small text
# files and print CMake syntax, so it stays a plain, dependency-free script
# that can't accidentally pull in e.g. the `distributed` package.
#
# Feature-flag filtering (mpiranks/build-flag requirements) is *not* decided
# here -- it can't be, since the flags only exist once the binary this
# configure produces has actually been built and run. Every generated test
# instead gets its raw requirement string passed through as a --requirement
# argument to ctest_run_one.py, which checks it at `ctest` run time against
# the flags file produced once by the FIXTURES_SETUP step.

from pathlib import Path
from typing import List, Tuple
import argparse
import json
import re
import subprocess
import sys

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

SKIP_RETURN_CODE = 125  # must match ctest_run_one.py's SKIP_RETURN_CODE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp2k-root", type=Path, required=True)
    parser.add_argument("--tests-source-dir", type=Path, required=True)
    parser.add_argument("--regtesting-dir", type=Path, required=True)
    parser.add_argument("--flags-file", type=Path, required=True)
    parser.add_argument("--binary-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mpiranks", type=int, default=2)
    parser.add_argument("--ompthreads", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=400)
    parser.add_argument("--mpiexec", default="mpiexec -n {N} --bind-to none")
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--valgrind", action="store_true")
    parser.add_argument("--cp2k-data-dir", type=Path, default=None)
    parser.add_argument("--restrict-dir", action="append", default=[])
    return parser.parse_args()


def is_intel_mpi(mpiexec_cmd: str) -> bool:
    try:
        result = subprocess.run(
            [mpiexec_cmd, "--version"], capture_output=True, text=True, timeout=10
        )
        return "Intel" in result.stdout or "Intel" in result.stderr
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def build_mpi_prefix(mpiexec_template: str, mpiranks: int) -> str:
    if "{N}" not in mpiexec_template:  # backwards compatibility
        words = mpiexec_template.split(" ", 1)
        launcher_command = words[0]
        remaining_arguments = words[1] if len(words) > 1 else ""
        mpiexec_template = f"{launcher_command} -n {{N}} {remaining_arguments}".strip()
    if is_intel_mpi(mpiexec_template.split()[0]) and "--bind-to" in mpiexec_template:
        mpiexec_template = mpiexec_template.replace(" --bind-to none", "")
    return mpiexec_template.format(N=mpiranks)


def detect_num_gpus() -> int:
    def run_capture(shell_command: str) -> int:
        result = subprocess.run(
            shell_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        return int(result.stdout)

    nvidia_command = "nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l"
    amd_command = "rocm-smi --showid --csv | grep card | wc -l"
    return run_capture(nvidia_command) + run_capture(amd_command)


def cmake_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def cmake_bracket(value: str) -> str:
    return f"[==[{value}]==]"


class TestCase:
    def __init__(
        self,
        name: str,
        exe_stem: str,
        extra_args: List[str],
        out_name: str,
        requirements: List[str],
        matcher_specs: List[dict],
        kind: str,
    ):
        self.name = name
        self.exe_stem = exe_stem
        self.extra_args = extra_args
        self.out_name = out_name
        self.requirements = requirements
        self.matcher_specs = matcher_specs
        self.kind = kind

    @property
    def batch_name(self) -> str:
        # Unit tests have no further split: "UNIT/foo_unittest" is itself the
        # batch (do_regtest.py creates one Batch per unit test), whereas a
        # regtest name is "TEST_DIRS-name/input-file.inp" and the batch is
        # everything before the input file.
        if self.kind == "unittest":
            return self.name
        return self.name.rsplit("/", 1)[0]

    @property
    def label(self) -> str:
        return self.name.split("/", 1)[0]

    @property
    def is_mimic(self) -> bool:
        return Path(self.batch_name).parent.name == "MIMIC"


def collect_test_cases(
    cp2k_root: Path, restrict_patterns: List[str]
) -> Tuple[List[TestCase], int]:
    def is_restricted(name: str) -> bool:
        if not restrict_patterns:
            return False
        return not any(re.fullmatch(p, name) for p in restrict_patterns)

    cases: List[TestCase] = []

    unit_tests_fn = cp2k_root / "tests" / "UNIT_TESTS"
    for line in unit_tests_fn.read_text(encoding="utf8").split("\n"):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        exe_name = parts[0]
        requirements = parts[1:]
        batch_name = f"UNIT/{exe_name}"
        if is_restricted(batch_name):
            continue
        cases.append(
            TestCase(
                name=batch_name,
                exe_stem=exe_name,
                extra_args=[str(cp2k_root)],
                out_name=f"{exe_name}.out",
                requirements=requirements,
                matcher_specs=[],
                kind="unittest",
            )
        )

    num_regtest_dirs = 0
    test_dirs_fn = cp2k_root / "tests" / "TEST_DIRS"
    for line in test_dirs_fn.read_text(encoding="utf8").split("\n"):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        batch_name = parts[0]
        requirements = parts[1:]
        if is_restricted(batch_name):
            continue
        num_regtest_dirs += 1

        test_files_fn = cp2k_root / "tests" / batch_name / "TEST_FILES.toml"
        test_files_content = test_files_fn.read_text(encoding="utf8")
        for inp_fn, matcher_specs in tomllib.loads(test_files_content).items():
            cases.append(
                TestCase(
                    name=f"{batch_name}/{inp_fn}",
                    exe_stem="cp2k",
                    extra_args=[inp_fn],
                    out_name=f"{inp_fn}.out",
                    requirements=requirements,
                    matcher_specs=matcher_specs,
                    kind="regtest",
                )
            )

    return cases, num_regtest_dirs


def main() -> None:
    args = parse_args()

    ompthreads = args.ompthreads or (2 if "smp" in args.version else 1)
    use_mpi = args.version.startswith("p")
    mpiranks = args.mpiranks if use_mpi else 1
    mpi_prefix = build_mpi_prefix(args.mpiexec, mpiranks) if use_mpi else ""
    intel_mpi = use_mpi and is_intel_mpi(mpi_prefix.split()[0])
    num_gpus = args.num_gpus if args.num_gpus > 0 else detect_num_gpus()
    cp2k_data_dir = args.cp2k_data_dir or (args.cp2k_root / "data")

    huge_supps_fn = args.cp2k_root / "tests" / "HUGE_TESTS_SUPPRESSIONS"
    huge_suppressions = set(huge_supps_fn.read_text(encoding="utf8").split("\n"))

    cases, num_regtest_dirs = collect_test_cases(args.cp2k_root, args.restrict_dir)

    lines: List[str] = [
        "# Auto-generated by generate_ctest_list.py -- do not edit by hand.",
        "",
    ]
    runner = args.tests_source_dir / "ctest_run_one.py"
    gpu_offset = 0
    for case in cases:
        workdir = args.regtesting_dir / case.batch_name
        cmd = [
            "${Python_EXECUTABLE}",
            cmake_quote(str(runner)),
            "--kind", case.kind,
            "--binary-dir", cmake_quote(str(args.binary_dir)),
            "--version", args.version,
            "--exe-stem", case.exe_stem,
            "--workdir", cmake_quote(str(workdir)),
            "--out-name", cmake_quote(case.out_name),
            "--timeout", str(args.timeout),
            "--mpi-prefix", cmake_quote(mpi_prefix),
            "--ompthreads", str(ompthreads),
            "--cp2k-data-dir", cmake_quote(str(cp2k_data_dir)),
            "--num-gpus", str(num_gpus),
            "--mpiranks", str(mpiranks),
            "--gpu-offset", str(gpu_offset % num_gpus if num_gpus else 0),
            "--flags-file", cmake_quote(str(args.flags_file)),
        ]
        gpu_offset += 1
        for extra_arg in case.extra_args:
            cmd += ["--extra-arg", cmake_quote(extra_arg)]
        if intel_mpi:
            cmd.append("--intel-mpi")
        if case.is_mimic:
            cmd.append("--mimic")
        if args.valgrind:
            cmd.append("--valgrind")
        if case.requirements:
            cmd += ["--requirement", cmake_quote(" ".join(case.requirements))]
        if case.name in huge_suppressions:
            cmd.append("--huge-suppressed")
        if case.matcher_specs:
            cmd += ["--matcher-specs", cmake_bracket(json.dumps(case.matcher_specs))]

        test_name = cmake_quote(case.name)
        lines.append(f"add_test(NAME {test_name} COMMAND {' '.join(cmd)})")
        lines.append(
            "set_tests_properties(%s PROPERTIES\n"
            "  TIMEOUT %d\n"
            "  WORKING_DIRECTORY %s\n"
            "  FIXTURES_REQUIRED regtest_environment_setup\n"
            "  RESOURCE_LOCK %s\n"
            "  SKIP_RETURN_CODE %d\n"
            "  LABELS %s)"
            % (
                test_name,
                args.timeout + 60,
                cmake_quote(str(workdir)),
                cmake_quote(case.batch_name),
                SKIP_RETURN_CODE,
                cmake_quote(case.label),
            )
        )
        lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf8")

    # tests/CMakeLists.txt reads this back to launch the same way for the
    # one-off "cp2k --version" call in the FIXTURES_SETUP step, so that
    # step and every generated test agree on how to invoke the binary.
    mpi_prefix_file = args.output.with_name("mpi_prefix.txt")
    mpi_prefix_file.write_text(mpi_prefix, encoding="utf8")

    print(
        f"Generated {len(cases)} CTest tests "
        f"({num_regtest_dirs} regtest directories) -> {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

# EOF
