CFLAGS   = -D__MKL -pthread -g -O2 -mavx2 -ftree-loop-vectorize -ftree-vectorize -funroll-loops  -std=c99 -march=native -Wall -Wextra -D__LIBXSMM -DHAVE_MKL -I/opt/intel/compilers_and_libraries_2019.0.117/linux/mkl/include
CC=gcc

all: grid_collocate_miniapp.x

grid_collocate_cpu.o: grid_prepare_pab.o mapping.o
grid_collocate_replay.o: grid_collocate_cpu.o

%.o: %.c
	$(CC) -c $(CFLAGS) $<

grid_collocate_miniapp.x: grid_collocate_miniapp.o grid_collocate_replay.o grid_collocate_cpu.o grid_prepare_pab.o utils.o thpool.o coefficients.o grid_integrate_cpu.o collocation_integration.o
	$(CC) $(CFLAGS) -o $@ $^ -lxsmmnoblas -lxsmm -lmkl_rt -lm

clean:
	rm -fv grid_*.o grid_*.x utils.o tmp.o thpool.o

#EOF
