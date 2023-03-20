ARG BASE_IMAGE
FROM $BASE_IMAGE

ARG SPECDEV

# show the spack's spec
RUN spack spec -I $SPECDEV

RUN spack env create --with-view /opt/cp2k cp2k-env
RUN spack -e cp2k-env add $SPECDEV

# copy source files of the pull request into container
COPY . /cp2k-src

# build CP2K
RUN spack --color always -e cp2k-env dev-build --source-path /cp2k-src +enable_regtests $SPECDEV

# we need a fixed name for the build directory
# here is a hacky workaround to link ./spack-build-{hash} to ./spack-build
RUN cd /cp2k-src && ln -s $(find . -name "spack-build-*" -type d) spack-build
