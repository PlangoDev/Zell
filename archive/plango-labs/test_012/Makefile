# ─────────────────────────────────────────────────────────────────────────────
# Makefile for test 012 — LLM vs the Brain, on WORDS (C++).
#
#   make           build the optimized binary  ./showdown
#   make run       build, then run it
#   make clean     remove build artifacts
#
# -O3 -march=native -ffast-math is what makes this fast: it SIMD-vectorizes the
# dot-product / reduction loops (the single biggest speed flag, as in test_011).
# ─────────────────────────────────────────────────────────────────────────────
CXX     ?= c++
# -ffast-math treats FP add as associative so the dot-product / reduction loops
# SIMD-vectorize; -flto inlines the hot kernels across files; -fno-math-errno
# drops errno bookkeeping in expf. Everything here is portable — the same sources
# build on macOS, Linux, and Windows (mingw / MSVC) with no external libraries.
CXXFLAGS := -O3 -ffast-math -march=native -funroll-loops -fno-math-errno -flto -pthread -Wall -Wextra -std=c++17
LDFLAGS := -pthread -flto
SRC     := src/main.cpp src/corpus.cpp src/llm.cpp src/brain.cpp src/analytics.cpp src/threads.cpp
HDR     := src/common.h src/config.h src/corpus.h src/llm.h src/brain.h src/analytics.h src/threads.h
BIN     := showdown

$(BIN): $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(BIN) $(LDFLAGS)

run: $(BIN)
	./$(BIN)

clean:
	rm -f $(BIN)

.PHONY: run clean
