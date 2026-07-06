#!/usr/bin/env python3

# CTest FIXTURES_SETUP step shared by every generated regtest/unittest.
# CTest guarantees this runs exactly once per `ctest` invocation before any
# test that declares FIXTURES_REQUIRED on it, even under `ctest -j`. It does
# the two things that must happen once, not once per test:
#   1. Copy tests/ into a stable location in the build tree (test dirs are
#      not self-contained, so this mirrors do_regtest.py's shutil.copytree).
#   2. Query the just-built cp2k binary for its runtime feature flags and
#      write them to a file that ctest_run_one.py reads instead of querying
#      the binary itself -- doing that once here instead of 5000+ times
#      keeps `ctest` fast to start.

from pathlib import Path
import argparse
import re
import shutil
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp2k-root", type=Path, required=True)
    parser.add_argument("--regtesting-dir", type=Path, required=True)
    parser.add_argument("--binary-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--mpi-prefix", default="")
    parser.add_argument("--flags-out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.regtesting_dir.exists():
        shutil.rmtree(args.regtesting_dir)
    shutil.copytree(args.cp2k_root / "tests", args.regtesting_dir)

    # UNIT_TESTS lists standalone binaries, not TEST_DIRS-style directories,
    # so there is no "tests/UNIT" source directory for copytree() above to
    # have brought along. Create one working directory per unit test here,
    # matching do_regtest.py's `batch.workdir.mkdir(parents=True)`.
    unit_tests_fn = args.cp2k_root / "tests" / "UNIT_TESTS"
    for line in unit_tests_fn.read_text(encoding="utf8").split("\n"):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        exe_name_for_unit_test = line.split()[0]
        (args.regtesting_dir / "UNIT" / exe_name_for_unit_test).mkdir(parents=True)

    exe_name = f"cp2k.{args.version}"
    cmd = [str(args.binary_dir / exe_name)]
    if args.mpi_prefix:
        cmd = args.mpi_prefix.split() + cmd
    cmd += ["--version"]

    result = subprocess.run(cmd, capture_output=True, timeout=60)
    version_output = result.stdout.decode("utf8", errors="replace")
    flags_line = re.search(r" cp2kflags:(.*)\n", version_output)
    if not flags_line:
        print(version_output)
        print("Could not parse feature flags from cp2k --version output.")
        sys.exit(1)

    args.flags_out.parent.mkdir(parents=True, exist_ok=True)
    args.flags_out.write_text(flags_line.group(1).strip() + "\n", encoding="utf8")


if __name__ == "__main__":
    main()

# EOF
