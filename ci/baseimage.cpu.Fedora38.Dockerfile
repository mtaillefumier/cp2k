FROM fedora:38 as builder

ARG CUDA_ARCH=80

ENV DEBIAN_FRONTEND noninteractive

ENV FORCE_UNSAFE_CONFIGURE 1

ENV PATH="/spack/bin:${PATH}"

ENV MPICH_VERSION=4.0.3

ENV CMAKE_VERSION=3.25.2

RUN dnf -y update


RUN dnf -y install cmake gcc git make autogen automake vim \
	                 wget gnupg tar gcc-c++ boost-devel gfortran doxygen libtool \
                   m4 libpciaccess-devel clingo xz bzip2 gzip unzip zlib-devel \
                   ncurses-devel libxml2-devel gsl-devel zstd openblas-devel \
                   flexiblas-devel patch bzip2-devel
#

# get latest version of spack
RUN git clone https://github.com/spack/spack.git

# set the location of packages built by spack
RUN spack config add config:install_tree:root:/opt/spack
# set cuda_arch for all packages
# RUN spack config add packages:all:variants:cuda_arch=${CUDA_ARCH}

# find all external packages
RUN spack external find --all --exclude python

# find compilers
RUN spack compiler find

# install MPICH
RUN spack install --only=dependencies mpich@${MPICH_VERSION} %gcc
RUN spack install mpich@${MPICH_VERSION} %gcc
RUN spack install intel-oneapi-mkl+cluster
RUN spack install openblas+fortran
# for the MPI hook
RUN echo $(spack find --format='{prefix.lib}' mpich) > /etc/ld.so.conf.d/mpich.conf
RUN ldconfig

ENV SPEC_OPENBLAS="cp2k@develop%gcc +sirius +elpa +libxc +libint smm=libxsmm +spglib +cosma +pexsi +plumed +libvori +openmp ^openblas+fortran"
ENV SPEC_MKL="cp2k@develop%gcc +sirius +elpa +libxc +libint smm=libxsmm +spglib +cosma +pexsi +plumed +libvori +mpi +openmp ^intel-oneapi-mkl+cluster"

# install all dependencies
RUN spack install --only=dependencies $SPEC_OPENBLAS ^mpich 
RUN spack install --only=dependencies $SPEC_OPENBLAS ^openmpi 
RUN spack install dbcsr ^openblas+fortran ^mpich
RUN spack install dbcsr ^openblas+fortran ^openmpi
RUN spack install --only=dependencies $SPEC_MKL ^mpich
RUN spack install --only=dependencies $SPEC_MKL ^openmpi
RUN spack install dbcsr ^intel-oneapi-mkl+cluster ^mpich
RUN spack install dbcsr ^intel-oneapi-mkl+cluster ^openmpi

