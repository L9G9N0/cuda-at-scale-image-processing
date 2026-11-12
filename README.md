# CUDA Batch Image Processing: RGB 3x3 Box Blur

## Purpose
A custom CUDA kernel processes a batch of RGB images. Each GPU thread computes one output pixel by averaging the nine neighboring pixels independently for each color channel. Edge coordinates are clamped to the image boundary.

The default workload is **100 images of 256 x 256 pixels**, processed in one program execution. A deterministic Python generator creates binary PPM (P6) images.

## GPU implementation
- Custom CUDA kernel `BoxBlurRgb` in `main.cu`.
- 16 x 16 threads per block.
- `cudaMalloc`, host-to-device and device-to-host `cudaMemcpy`, and `cudaFree`.
- CUDA events measure kernel execution time.
- Writes processed PPM files and per-image timing data to CSV.

## Google Colab instructions
1. Open the included `CUDA_at_Scale_Batch_Image_Processing.ipynb` in Google Colab.
2. Select **Runtime -> Change runtime type -> T4 GPU** (or another NVIDIA GPU).
3. Run all cells and upload `cuda_at_scale_independent_project.zip` when prompted.
4. The notebook checks `nvidia-smi` and `nvcc`, installs NumPy, generates the dataset, compiles and runs the CUDA program, and downloads `execution_evidence.zip`.

If using a terminal instead of the notebook:
```bash
unzip cuda_at_scale_independent_project.zip -d cuda_project
cd cuda_project
!nvidia-smi
python3 -m pip install numpy
bash run.sh
```
In a normal shell, omit the `!` before `nvidia-smi`.

## CLI
```bash
./cuda_image_batch --input-dir data/input --output-dir data/output --iterations 3
```
Options: `--input-dir`, `--output-dir`, `--iterations`.

Increase the workload:
```bash
IMAGE_COUNT=200 IMAGE_WIDTH=512 IMAGE_HEIGHT=512 ITERATIONS=5 bash run.sh
```

## Evidence
A successful run creates:
- `artifacts/run.log`: complete execution log.
- `artifacts/results.csv`: per-image dimensions and kernel timing.
- `artifacts/experiment_summary.txt`: workload configuration.
- `data/output/`: processed images.

Create the evidence archive:
```bash
zip -qr execution_evidence.zip artifacts data/output
```

## Limitations
Requires an NVIDIA GPU and CUDA toolkit. CUDA does not run natively on an Apple Silicon Mac. This implementation uses custom CUDA kernels, not NPP. Reported timing is kernel-only and excludes file I/O and memory transfers. Input images are synthetic and reproducibly generated for the assignment.
