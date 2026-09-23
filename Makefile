CC      := clang
MIN_MACOS ?= 14.0
CFLAGS  := -O2 -std=c11 -mmacosx-version-min=$(MIN_MACOS) -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -Isrc -Ibuild -MMD -MP
OBJCFLAGS := -O2 -fobjc-arc -mmacosx-version-min=$(MIN_MACOS) -Wall -Wno-deprecated-declarations -Isrc -Ibuild -MMD -MP
LDFLAGS := -mmacosx-version-min=$(MIN_MACOS) -framework Metal -framework Foundation -framework AudioToolbox -framework CoreAudio -framework CoreFoundation -lm

SRC_C := src/main.c src/json.c src/safetensors.c src/wav.c src/tokenizer.c src/mel.c src/model.c src/encoder.c src/decoder.c src/mic.c src/resample.c src/warmup.c src/nemoasr.c
LIB_SRC := src/json.c src/safetensors.c src/tokenizer.c src/mel.c src/model.c src/encoder.c src/decoder.c src/resample.c src/warmup.c src/nemoasr.c
LIB_OBJ := $(patsubst src/%.c,build/%.o,$(LIB_SRC)) build/gpu.o
OBJ   := $(patsubst src/%.c,build/%.o,$(SRC_C)) build/gpu.o

all: nemoasr-c

# Static library for embedding (no CLI, no CoreAudio). Link with -framework Metal -framework Foundation.
lib: build/libnemoasr.a

build/libnemoasr.a: $(LIB_OBJ)
	ar rcs $@ $(LIB_OBJ)

libtest: tools/libtest.c build/libnemoasr.a build/wav.o
	$(CC) $(CFLAGS) tools/libtest.c build/wav.o build/libnemoasr.a -framework Metal -framework Foundation -o libtest

build:
	@mkdir -p build

# Embed the Metal source (with the shared params header prepended) as a C string literal.
build/kernels_metal.h: src/kernels.metal src/kernel_params.h | build
	@{ printf 'const char kernels_metal_src[] =\n'; \
	   cat src/kernel_params.h src/kernels.metal | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/^/"/' -e 's/$$/\\n"/'; \
	   printf ';\n'; } > $@

build/%.o: src/%.c build/kernels_metal.h | build
	$(CC) $(CFLAGS) -c $< -o $@

build/gpu.o: src/gpu.m build/kernels_metal.h | build
	$(CC) $(OBJCFLAGS) -c $< -o $@

nemoasr-c: $(OBJ)
	$(CC) $(OBJ) $(LDFLAGS) -o $@

clean:
	rm -rf build nemoasr-c kbench libtest

-include $(OBJ:.o=.d)

.PHONY: all clean lib

kbench: tools/kbench.c build/gpu.o build/kernels_metal.h
	$(CC) $(CFLAGS) tools/kbench.c build/gpu.o -framework Metal -framework Foundation -o kbench
