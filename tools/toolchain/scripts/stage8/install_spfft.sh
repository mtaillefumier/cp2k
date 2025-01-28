#!/bin/bash -e

# TODO: Review and if possible fix shellcheck errors.
# shellcheck disable=all

[ "${BASH_SOURCE[0]}" ] && SCRIPT_NAME="${BASH_SOURCE[0]}" || SCRIPT_NAME=$0
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_NAME")/.." && pwd -P)"

spfft_ver="1.1.0"
spfft_sha256="d4673b3135aebfa1c440723226fe976d518ff881285b3d4787f1aa8210eac81e"
source "${SCRIPT_DIR}"/common_vars.sh
source "${SCRIPT_DIR}"/tool_kit.sh
source "${SCRIPT_DIR}"/signal_trap.sh
source "${INSTALLDIR}"/toolchain.conf
source "${INSTALLDIR}"/toolchain.env

[ -f "${BUILDDIR}/setup_SpFFT" ] && rm "${BUILDDIR}/setup_SpFFT"

! [ -d "${BUILDDIR}" ] && mkdir -p "${BUILDDIR}"
cd "${BUILDDIR}"

case "${with_spfft}" in
  __INSTALL__)
    echo "==================== Installing spfft ===================="
    pkg_install_dir="${INSTALLDIR}/SpFFT-${spfft_ver}"
    install_lock_file="$pkg_install_dir/install_successful"
    if verify_checksums "${install_lock_file}"; then
      echo "SpFFT-${spfft_ver} is already installed, skipping it."
    else
      if [ -f SpFFT-${spfft_ver}.tar.gz ]; then
        echo "SpFFT-${spfft_ver}.tar.gz is found"
      else
        download_pkg_from_cp2k_org "${spfft_sha256}" "SpFFT-${spfft_ver}.tar.gz"

      fi
      if [ "${MATH_MODE}" = "mkl" ]; then
        EXTRA_CMAKE_FLAGS="-DSPFFT_MKL=ON -DSPFFT_FFTW_LIB=MKL"
      else
        EXTRA_CMAKE_FLAGS=""
      fi
      echo "Installing from scratch into ${pkg_install_dir}"
      [ -d SpFFT-${spfft_ver} ] && rm -rf SpFFT-${spfft_ver}
      tar -xzf SpFFT-${spfft_ver}.tar.gz
      cd SpFFT-${spfft_ver}
      mkdir build-cpu
      cd build-cpu
      cmake \
        -DCMAKE_INSTALL_PREFIX="${pkg_install_dir}/cpu" \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -DCMAKE_CXX_COMPILER="${MPICXX}" \
        -DCMAKE_VERBOSE_MAKEFILE=ON \
        -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DSPFFT_OMP=ON \
        -DSPFFT_MPI=ON \
        -DSPFFT_STATIC=ON \
        -DSPFFT_FORTRAN=ON \
        -DSPFFT_INSTALL=ON \
        ${EXTRA_CMAKE_FLAGS} .. \
        > cmake.log 2>&1 || tail -n ${LOG_LINES} cmake.log
      make -j $(get_nprocs) > make.log 2>&1 || tail -n ${LOG_LINES} make.log
      make -j $(get_nprocs) install > install.log 2>&1 || tail -n ${LOG_LINES} install.log

      cd ..

      if [ "$ENABLE_CUDA" = "__TRUE__" ]; then
        [ -d build-cuda ] && rm -rf "build-cuda"
        mkdir build-cuda
        cd build-cuda
        cmake \
          -DCMAKE_INSTALL_PREFIX="${pkg_install_dir}/cuda" \
          -DCMAKE_INSTALL_LIBDIR=lib \
          -DCMAKE_CXX_COMPILER="${MPICXX}" \
          -DCMAKE_CUDA_FLAGS="-std=c++14 -allow-unsupported-compiler" \
          -DCMAKE_VERBOSE_MAKEFILE=ON \
          -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
          -DSPFFT_OMP=ON \
          -DSPFFT_MPI=ON \
          -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
          -DSPFFT_STATIC=ON \
          -DSPFFT_FORTRAN=ON \
          -DSPFFT_INSTALL=ON \
          -DSPFFT_GPU_BACKEND=CUDA \
          ${EXTRA_CMAKE_FLAGS} .. > cmake.log 2>&1 || tail -n ${LOG_LINES} cmake.log
        make -j $(get_nprocs) > make.log 2>&1 || tail -n ${LOG_LINES} make.log
        make -j $(get_nprocs) install > install.log 2>&1 || tail -n ${LOG_LINES} install.log
      fi
      write_checksums "${install_lock_file}" "${SCRIPT_DIR}/stage8/$(basename ${SCRIPT_NAME})"

      if [ "$ENABLE_HIP" = "__TRUE__" ]; then
        case "${GPUVER}" in
          K20X | K40 | K80 | P100 | V100 | A100 | A40 | H100)
            [ -d build-cuda ] && rm -rf "build-cuda"
            mkdir build-cuda
            cd build-cuda
            cmake \
              -DCMAKE_INSTALL_PREFIX="${pkg_install_dir}/hip" \
              -DCMAKE_INSTALL_LIBDIR=lib \
              -DCMAKE_CXX_COMPILER="${MPICXX}" \
              -DCMAKE_CUDA_FLAGS="-std=c++14 -allow-unsupported-compiler" \
              -DCMAKE_VERBOSE_MAKEFILE=ON \
              -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
              -DSPFFT_OMP=ON \
              -DSPFFT_MPI=ON \
              -DSPFFT_STATIC=ON \
              -DSPFFT_FORTRAN=ON \
              -DSPFFT_INSTALL=ON \
              -DSPFFT_GPU_BACKEND=CUDA \
              ${EXTRA_CMAKE_FLAGS} .. > cmake.log 2>&1 || tail -n ${LOG_LINES} cmake.log
            make -j $(get_nprocs) > make.log 2>&1 || tail -n ${LOG_LINES} make.log
            make -j $(get_nprocs) install > install.log 2>&1 || tail -n ${LOG_LINES} install.log
            ;;
          Mi50 | Mi100 | Mi200 | Mi250)
            [ -d build-hip ] && rm -rf "build-hip"
            mkdir build-hip
            cd build-hip
            cmake \
              -DCMAKE_INSTALL_PREFIX="${pkg_install_dir}/hip" \
              -DCMAKE_INSTALL_LIBDIR=lib \
              -DCMAKE_VERBOSE_MAKEFILE=ON \
              -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
              -DSPLA_OMP=ON \
              -DSPLA_FORTRAN=ON \
              -DSPLA_INSTALL=ON \
              -DSPLA_STATIC=ON \
              -DSPLA_GPU_BACKEND=ROCM \
              ${EXTRA_CMAKE_FLAGS} .. \
              > cmake.log 2>&1 || tail -n ${LOG_LINES} cmake.log
            make -j $(get_nprocs) > make.log 2>&1 || tail -n ${LOG_LINES} make.log
            make -j $(get_nprocs) install > install.log 2>&1 || tail -n ${LOG_LINES} install.log
            ;;
          *) ;;

        esac
      fi
      write_checksums "${install_lock_file}" "${SCRIPT_DIR}/stage8/$(basename ${SCRIPT_NAME})"
    fi
    SPFFT_ROOT="${pkg_install_dir}"
    SPFFT_CFLAGS="-I'${pkg_install_dir}/include'"
    SPFFT_CPU_LDFLAGS="-L'${pkg_install_dir}/cpu/lib' -Wl,-rpath,'${pkg_install_dir}/cpu/lib'"
    SPFFT_CUDA_LDFLAGS="-L'${pkg_install_dir}/cuda/lib' -Wl,-rpath,'${pkg_install_dir}/cuda/lib'"
    SPFFT_HIP_LDFLAGS="-L'${pkg_install_dir}/hip/lib' -Wl,-rpath,'${pkg_install_dir}/hip/lib'"
    ;;
  __SYSTEM__)
    echo "==================== Finding spfft from system paths ===================="
    check_command pkg-config --modversion spfft
    add_include_from_paths SPFFT_CFLAGS "spfft.h" $INCLUDE_PATHS
    add_lib_from_paths SPFFT_LDFLAGS "libspfft.*" $LIB_PATHS
    ;;
  __DONTUSE__)
    # Nothing to do
    ;;
  *)
    echo "==================== Linking spfft to user paths ===================="
    pkg_install_dir="$with_spfft"

    # use the lib64 directory if present (multi-abi distros may link lib/ to lib32/ instead)
    SPFFT_LIBDIR="${pkg_install_dir}/lib"
    [ -d "${pkg_install_dir}/lib64" ] && SPFFT_LIBDIR="${pkg_install_dir}/lib64"

    check_dir "${SPFFT_LIBDIR}"
    check_dir "${pkg_install_dir}/include"
    SPFFT_CFLAGS="-I'${pkg_install_dir}/include'"
    SPFFT_LDFLAGS="-L'${SPFFT_LIBDIR}' -Wl,-rpath,'${SPFFT_LIBDIR}'"
    ;;
esac
if [ "$with_spfft" != "__DONTUSE__" ]; then
  SPFFT_LIBS="-lspfft"
  if [ "$with_spfft" != "__SYSTEM__" ]; then
    cat << EOF > "${BUILDDIR}/setup_spfft"
export SPFFT_LIBS="-lspfft"
export SPFFT_ROOT_CUDA=${pkg_install_dir}/cuda
export SPFFT_ROOT_HIP=${pkg_install_dir}/hip
export SPFFT_ROOT_CPU=${pkg_install_dir}/cpu
export SPFFT_ROOT="IF_CUDA(\${SPFFT_ROOT_CUDA}|IF_HIP(\${SPFFT_ROOT_HIP}|\${SPFFT_ROOT_CPU}))"
export SPFFT_INCLUDE_DIR=\${SPFFT_ROOT}/include/spfft
prepend_path PKG_CONFIG_PATH "\${SPFFT_ROOT}/lib/pkgconfig"
prepend_path CMAKE_PREFIX_PATH "\${SPFFT_ROOT}"
prepend_path LD_LIBRARY_PATH "\${SPFFT_ROOT}/lib"
prepend_path LD_RUN_PATH "\${SPFFT_ROOT}/lib"
prepend_path LIBRARY_PATH "\${SPFFT_ROOT}/lib"
prepend_path CPATH "\${SPFFT_ROOT}/include"
EOF
  else
    cat << EOF > "${BUILDDIR}/setup_spfft"
export SPFFT_ROOT=${pkg_install_dir}
export SPFFT_ROOT_CUDA=${pkg_install_dir}
export SPFFT_ROOT_HIP=${pkg_install_dir}
export SPFFT_ROOT_CPU=${pkg_install_dir}
EOF
  fi
  cat << EOF >> "${BUILDDIR}/setup_spfft"
export SPFFT_VER="${spfft_ver}"
export SPFFT_HIP_LDFLAGS="${SPFFT_HIP_LDFLAGS}"
export SPFFT_CPU_LDFLAGS="${SPFFT_CPU_LDFLAGS}"
export SPFFT_CUDA_LDFLAGS="${SPFFT_CUDA_LDFLAGS}"
export CP_DFLAGS="\${CP_DFLAGS} IF_MPI(-D__SPFFT|)"
export CP_CFLAGS="\${CP_CFLAGS} IF_MPI(-I\${SPFFT_ROOT}/include/spfft|)"
export CP_LDFLAGS="\${CP_LDFLAGS} IF_MPI(IF_CUDA(\${SPFFT_CUDA_LDFLAGS}|IF_HIP(\${SPFFT_HIP_LDFLAGS}|\${SPFFT_CPU_LDFLAGS}))|)"
export SPFFT_LIBRARY="-lspfft"
export CP_LIBS="IF_MPI(${SPFFT_LIBS}|) \${CP_LIBS}"
EOF
  cat "${BUILDDIR}/setup_spfft" >> $SETUPFILE
fi

load "${BUILDDIR}/setup_spfft"
write_toolchain_env "${INSTALLDIR}"

cd "${ROOTDIR}"
report_timing "spfft"
