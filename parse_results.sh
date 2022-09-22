#!/usr/bin/env bash

set -eux

bench_dir=${1}

output_file="${bench_dir}/results.csv"
echo "config,block_size,nodes,total_time,cholesky_time" > "${output_file}"

readarray -t results < <(cd "${bench_dir}" && find . -mindepth 3 -maxdepth 3 -type d)
for result in "${results[@]}"
do
    echo ${result}
    config=$(echo "${result}" | cut -f2 -d'/')
    block_size=$(echo "${result}" | cut -f3 -d'/')
    nodes=$(echo "${result}" | cut -f4 -d'/')

    total_time=$(cd "${bench_dir}" && grep '^ CP2K  ' "${result}/output.txt" | awk '{ print $7 }')
    cholesky_time=$(cd "${bench_dir}" && grep '^ cp_fm_cholesky_decompose  ' "${result}/output.txt" | awk '{ print $7 }')
    echo "${config},${block_size},${nodes},${total_time},${cholesky_time}" >> "${output_file}"
done
