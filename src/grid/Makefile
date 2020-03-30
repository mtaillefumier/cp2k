CFLAGS   = -pthread -pg -g -O2 -mavx2 -ftree-loop-vectorize -ftree-vectorize -std=c99 -march=native -Wall -Wextra -D__LIBXSMM -DHAVE_MKL -I/opt/intel/compilers_and_libraries_2019.0.117/linux/mkl/include
CC=gcc

all: grid_collocate_miniapp.x

grid_collocate_cpu.o: grid_prepare_pab.o mapping.o
grid_collocate_replay.o: grid_collocate_cpu.o

mapping.o : mapping.ispc
	ispc -O2 mapping.ispc -o mapping.o

%.o: %.c
	$(CC) -c $(CFLAGS) $<

grid_collocate_miniapp.x: grid_collocate_miniapp.o grid_collocate_replay.o grid_collocate_cpu.o grid_prepare_pab.o mapping.o tmp.o utils.o
	$(CC) $(CFLAGS) -o $@ $^ -lxsmmnoblas -lxsmm -lmkl_rt -lm

clean:
	rm -fv grid_*.o grid_*.x utils.o tmp.o

#EOF
