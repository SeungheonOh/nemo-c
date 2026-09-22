CC      := clang
CFLAGS  := -O2 -std=c11 -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -Isrc -Ibuild -MMD -MP
OBJCFLAGS := -O2 -fobjc-arc -Wall -Wno-deprecated-declarations -Isrc -Ibuild -MMD -MP
LDFLAGS := -framework Metal -framework Foundation -framework AudioToolbox -framework CoreAudio -framework CoreFoundation -lm

SRC_C := src/main.c src/json.c src/safetensors.c src/wav.c src/tokenizer.c src/mel.c src/model.c src/encoder.c src/decoder.c src/mic.c src/resample.c
OBJ   := $(patsubst src/%.c,build/%.o,$(SRC_C)) build/gpu.o

all: nemoasr-c

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
	rm -rf build nemoasr-c

-include $(OBJ:.o=.d)

.PHONY: all clean

kbench: tools/kbench.c build/gpu.o build/kernels_metal.h
	$(CC) $(CFLAGS) tools/kbench.c build/gpu.o -framework Metal -framework Foundation -o kbench
