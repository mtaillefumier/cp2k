FROM ubuntu:22.04 as builder

ARG CUDA_ARCH=80

ENV DEBIAN_FRONTEND noninteractive

ENV FORCE_UNSAFE_CONFIGURE 1

ENV PATH="/spack/bin:${PATH}"

ENV MPICH_VERSION=3.4.3

ENV CMAKE_VERSION=3.25.2

RUN apt-get -y update

RUN apt-get install -y apt-utils

# install basic tools
RUN apt-get install -y --no-install-recommends gcc g++ gfortran git make unzip file \
  vim wget pkg-config python3-pip curl tcl m4 cpio automake xz-utils patch \
  apt-transport-https ca-certificates gnupg software-properties-common perl tar bzip2

# install CMake
RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz -O cmake.tar.gz && \
    tar zxvf cmake.tar.gz --strip-components=1 -C /usr

# get latest version of spack
RUN git clone https://github.com/spack/spack.git

# set the location of packages built by spack
RUN spack config add config:install_tree:root:/opt/local
# set cuda_arch for all packages
# RUN spack config add packages:all:variants:cuda_arch=${CUDA_ARCH}

# find all external packages
RUN spack external find --all

# find compilers
RUN spack compiler find

# install MPICH
RUN spack install --only=dependencies mpich@${MPICH_VERSION} %gcc
RUN spack install mpich@${MPICH_VERSION} %gcc

# for the MPI hook
RUN echo $(spack find --format='{prefix.lib}' mpich) > /etc/ld.so.conf.d/mpich.conf
RUN ldconfig

ENV SPEC_OPENBLAS="cp2k@develop%gcc build_type=Release build_system=cmake +fftw +sirius +elpa +libxc +libint +smm=libxsmm +spglib +cosma +pexsi +plumed +libvori +openmp ^openblas+fortran"
ENV SPEC_MKL="cp2k@develop%gcc build_type=Release build_system=cmake +fftw +sirius +elpa +libxc +libint +smm=libxsmm +spglib +cosma +pexsi +plumed +libvori +mpi +openmp ^intel-oneapi-mkl+cluster"

# install all dependencies
RUN spack install --only=dependencies $SPEC_OPENBLAS ^mpich 
RUN spack install --only=dependencies $SPEC_MKL ^mpich
RUN spack install --only=dependencies $SPEC_OPENBLAS ^openmpi 
RUN spack install --only=dependencies $SPEC_MKL ^openmpi
RUN spack install dbcsr ^intel-oneapi-mkl+cluster ^mpich
RUN spack install dbcsr ^openblas+fortran ^mpich
RUN spack install dbcsr ^intel-oneapi-mkl+cluster ^openmpi
RUN spack install dbcsr ^openblas+fortran ^openmpi

