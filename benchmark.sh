#!/usr/bin/env bash

set -eux

cp2k_src_dir="${APPS_SRC}/cp2k"
cp2k_build_root_dir="${cp2k_src_dir}/build"
cp2k_bench_output_root_dir="${HOME}/cp2k-benchmarks"
date=$(date -Iseconds)
tag=${1}

declare -A build_dir
declare -A bench_input

configs=("scalapack" "dlaf-cpu" "dlaf-gpu")
nodes=(1 2 4 8)
block_sizes=(128 256)

build_dir[scalapack]="${cp2k_build_root_dir}/scalapack"
build_dir[dlaf-cpu]="${cp2k_build_root_dir}/dlaf-cpu"
build_dir[dlaf-gpu]="${cp2k_build_root_dir}/dlaf-gpu"

benchmark_reference_input="${cp2k_src_dir}/benchmarks/QS/H2O-256.inp"

#export MIMALLOC_EAGER_COMMIT_DELAY=0
#export MIMALLOC_LARGE_OS_PAGES=1

for n in "${nodes[@]}"
do
    for config in "${configs[@]}"
    do
        for block_size in "${block_sizes[@]}"
        do
            export cp2k_build_dir=${build_dir[${config}]}

            cp2k_bench_output_dir="${cp2k_bench_output_root_dir}/${date}-${tag}/${config}/${block_size}/${n}"
            mkdir -p "${cp2k_bench_output_dir}"

            export cp2k_bench_input="${cp2k_bench_output_dir}/H20-256.inp"
            cp "${benchmark_reference_input}" "${cp2k_bench_input}"

            sed -i "s|NCOL_BLOCKS.*|NCOL_BLOCKS ${block_size}|" "${cp2k_bench_input}"
            sed -i "s|NROW_BLOCKS.*|NROW_BLOCKS ${block_size}|" "${cp2k_bench_input}"

            srun --chdir "${cp2k_bench_output_dir}" -t 01:00:00 -N ${n} --account csstaff -C gpu "${PWD}/benchmark_batch.sh" | tee "${cp2k_bench_output_dir}/output.txt"
        done
    done
done
