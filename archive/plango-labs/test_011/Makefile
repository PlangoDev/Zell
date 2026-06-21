# ─────────────────────────────────────────────────────────────────────────────
# Makefile for test 011 — the C head-to-head.
#
#   make           build the optimized binary  ./showdown
#   make run       build, then run it
#   make data      regenerate the binary dataset (needs python + sklearn once)
#   make clean      remove build artifacts
#
# -O3 -march=native lets the compiler vectorize the hot loops, which is most
# of why this runs in a couple of seconds instead of a couple of minutes.
# ─────────────────────────────────────────────────────────────────────────────
CC      ?= cc
# -ffast-math lets the compiler SIMD-vectorize the dot-product / reduction loops
# (FP add is treated as associative). This is the single biggest speed flag here.
CFLAGS  := -O3 -march=native -ffast-math -funroll-loops -pthread -Wall -Wextra -std=c11
LDFLAGS := -lm -pthread
SRC     := src/main.c src/dataset.c src/llm.c src/brain.c src/threads.c src/analytics.c \
           src/memory.c
HDR     := src/common.h src/dataset.h src/llm.h src/brain.h src/threads.h src/analytics.h \
           src/memory.h
BIN     := showdown

$(BIN): $(SRC) $(HDR)
	$(CC) $(CFLAGS) $(SRC) -o $(BIN) $(LDFLAGS)

run: $(BIN)
	./$(BIN)

data:
	python3 tools/export_data.py

clean:
	rm -f $(BIN)

.PHONY: run data clean
