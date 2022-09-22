#!/usr/bin/env bash

if [[ "${SLURM_NODEID}" -eq "0" ]]; then
    env
    module list
fi
${cp2k_build_dir}/bin/cp2k -i ${cp2k_bench_input}
