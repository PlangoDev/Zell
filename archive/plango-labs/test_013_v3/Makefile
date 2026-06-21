# BrainFormer v13-v3 — nvcc build for Kaggle T4 (sm_75).
# One CUDA TU (main.cu) + the C data layer + the C++ thread pool, linked together.
NVCC ?= nvcc
ARCH := -gencode arch=compute_75,code=sm_75
NVCCFLAGS := -O3 $(ARCH) -std=c++14 --use_fast_math \
             -Xcompiler "-O3 -funroll-loops -pthread"
LDFLAGS := -lcublas -lpthread

SRC := src/main.cu src/data.c src/threads.cpp

brain: $(SRC) src/common.h src/config.h src/data.h src/cuda_util.h src/threads.h
	$(NVCC) $(NVCCFLAGS) src/main.cu src/data.c src/threads.cpp -o brain $(LDFLAGS)

clean:
	rm -f brain

.PHONY: clean
