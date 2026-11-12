NVCC ?= nvcc
TARGET := cuda_image_batch
.PHONY: all clean
all: $(TARGET)
$(TARGET): main.cu
	$(NVCC) -O2 -std=c++17 -lineinfo $< -o $@
clean:
	rm -f $(TARGET)
	rm -rf data/output
