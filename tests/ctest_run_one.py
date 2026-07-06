#!/usr/bin/env python3

# Runs exactly one CP2K regtest input file or unit test as a single CTest
# test. This is the COMMAND that generate_ctest_list.py puts into every
# add_test() it emits.
#
# Everything that's expensive or environment-dependent (GPU count, mpiexec
# template, the actual runtime feature flags) is resolved once elsewhere --
# GPU/mpiexec at CMake-configure time by generate_ctest_list.py, feature
# flags once per `ctest` run by the FIXTURES_SETUP step
# (ctest_prepare_environment.py) -- and handed to this script as plain
# arguments. This script itself does no auto-detection, so it stays cheap
# to start even though it runs thousands of times per full test suite.

from pathlib import Path
from typing import Dict, List
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matchers import run_matcher
from regtest_requirements import requirements_satisfied

# Must match the SKIP_RETURN_CODE value generate_ctest_list.py sets as a
# test property for every test it generates.
SKIP_RETURN_CODE = 125


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["unittest", "regtest"], required=True)
    parser.add_argument("--binary-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--exe-stem", required=True)
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--out-name", required=True)
    parser.add_argument("--timeout", type=int, required=True)
    parser.add_argument("--mpi-prefix", default="")
    parser.add_argument("--ompthreads", type=int, required=True)
    parser.add_argument("--cp2k-data-dir", type=Path, required=True)
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--mpiranks", type=int, default=1)
    parser.add_argument("--gpu-offset", type=int, default=0)
    parser.add_argument("--intel-mpi", action="store_true")
    parser.add_argument("--mimic", action="store_true")
    parser.add_argument("--valgrind", action="store_true")
    parser.add_argument("--requirement", default="")
    parser.add_argument("--flags-file", type=Path, required=True)
    parser.add_argument("--huge-suppressed", action="store_true")
    parser.add_argument("--matcher-specs", default="[]")
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> List[str]:
    exe_name = f"{args.exe_stem}.{args.version}"
    cmd = [str(args.binary_dir / exe_name)]
    if args.valgrind:
        cmd = ["valgrind", "--error-exitcode=42", "--exit-on-first-error=yes"] + cmd
    if args.mpi_prefix:
        cmd = args.mpi_prefix.split() + cmd
    return cmd + args.extra_arg


def build_env(args: argparse.Namespace) -> Dict[str, str]:
    env = os.environ.copy()
    if args.num_gpus > 0:
        visible_gpu_devices = []
        for rank in range(args.mpiranks):
            device_index = (args.gpu_offset + rank) % args.num_gpus
            visible_gpu_devices.append(str(device_index))
        env["CUDA_VISIBLE_DEVICES"] = ",".join(visible_gpu_devices)
        env["HIP_VISIBLE_DEVICES"] = ",".join(visible_gpu_devices)
    env["OMP_NUM_THREADS"] = str(args.ompthreads)
    if args.intel_mpi:
        env["I_MPI_PIN"] = "0"
    env["CP2K_DATA_DIR"] = str(args.cp2k_data_dir)
    env["PIKA_COMMANDLINE_OPTIONS"] = (
        f"--pika:bind=none --pika:threads={args.ompthreads}"
    )
    if args.mimic:
        env["MCL_COMM_MODE"] = "TEST_STUB"
        env["MCL_PROGRAM"] = "1"
        env["MCL_TEST_DATA"] = "MCL_LOG_1"
    return env


def run_child_process(cmd: List[str], env: Dict[str, str], cwd: Path, timeout: int):
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


def print_output_tail(output: bytes, num_lines: int = 100) -> None:
    lines = output.decode("utf8", errors="replace").split("\n")
    print("\n".join(lines[-num_lines:]))


def dirsize(folder: Path) -> int:
    total = 0
    for path in folder.rglob("*"):
        total += path.stat().st_size
    return total


def main() -> None:
    args = parse_args()

    flags = args.flags_file.read_text(encoding="utf8").split()
    if args.requirement:
        requirement_tokens = args.requirement.split()
        if not requirements_satisfied(requirement_tokens, flags, args.mpiranks):
            print(f"Requirements not satisfied: {args.requirement}")
            print(f"Available flags: {' '.join(flags)}")
            sys.exit(SKIP_RETURN_CODE)

    cmd = build_command(args)
    env = build_env(args)
    out_path = args.workdir / args.out_name

    if args.kind == "unittest":
        start_dirsize = None
    else:
        start_dirsize = dirsize(args.workdir)

    output, returncode, timed_out = run_child_process(
        cmd, env, args.workdir, args.timeout
    )
    out_path.write_bytes(output)

    if timed_out:
        print_output_tail(output)
        print(f"Timed out after {args.timeout} seconds.")
        sys.exit(1)

    if returncode != 0:
        print_output_tail(output)
        print(f"Runtime failure with code {returncode}.")
        sys.exit(1)

    if args.kind == "unittest":
        print("OK")
        sys.exit(0)

    # Regtest: check for huge output, then run matchers.
    output_size = dirsize(args.workdir) - start_dirsize
    if output_size > 2 * 1024 * 1024 and not args.huge_suppressed:  # 2 MiB limit
        print(f"Test produced {output_size / 1024 / 1024:.2f} MiB of output.")
        sys.exit(1)

    matcher_specs = json.loads(args.matcher_specs)
    if not matcher_specs:
        print("OK")
        sys.exit(0)

    output_text = output.decode("utf8", errors="replace")
    all_ok = True
    for spec in matcher_specs:
        spec = dict(spec)  # copy so popping "file" below does not change the original
        alt_file = spec.pop("file", None)
        if alt_file:
            alt_path = out_path.parent / alt_file
            if not alt_path.exists():
                print(f"Spec: {spec}\nExpected output file not found: {alt_path}")
                all_ok = False
                continue
            match_output = alt_path.read_bytes().decode("utf8", errors="replace")
        else:
            match_output = output_text
        result = run_matcher(match_output, **spec)
        status = "OK" if result.status == "OK" else result.status
        value = f"{result.value:.10g}" if result.value else "-"
        print(f"{spec.get('matcher', '???'):<30s} {value:>17} {status:>12s}")
        if result.error:
            print(result.error)
        if result.status != "OK":
            all_ok = False

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

# EOF
