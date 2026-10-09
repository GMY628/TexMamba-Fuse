# TexMamba-Fuse

Official code for **Common and Unique Representations for Multi-focus Image Fusion: A Feature Decomposition Paradigm with Text-Driven Enhancement** (TMM 2026).

## Environment

The project was tested with the following environment:

- Windows
- Python 3.11
- NVIDIA GPU and CUDA Toolkit
- PyTorch 2.14.1 + CUDA 13.2
- torchvision 0.29.1

Create and activate the Conda environment:

```bash
conda create -n tmamba_fuse python=3.11 -y
conda activate tmamba_fuse
```

Install a CUDA-enabled PyTorch build that matches your local CUDA Toolkit, then install the remaining dependencies:

```bash
pip install torch torchvision
pip install -r requirements.txt
```

`salesforce-lavis==1.0.2` pins an old OpenCV build. If you use NumPy 2.x, replace it with a NumPy-2-compatible OpenCV build:

```bash
pip install --upgrade opencv-python-headless==5.0.0.93
```

## Selective Scan CUDA Extension

The model requires the local `selective_scan_cuda_oflex` extension:

```bash
cd kernels/selective_scan
pip install .
cd ../..
```

The current build configuration targets NVIDIA compute capability `8.9` (`sm_89`) and uses MSVC/C++20. Update the `-gencode` setting in `kernels/selective_scan/setup.py` when compiling for another GPU architecture.

Verify the extension:

```bash
python -c "import selective_scan_cuda_oflex; print('selective_scan_cuda_oflex OK')"
```

## Pretrained Weights

The pretrained weights are included in the repository:

```text
checkpoints/stage1.pth
checkpoints/stage2.pth
```

## Testing

Run the two stages from the repository root:

```bash
python Test_stage1.py
python Test_stage2.py
```

The test scripts use the sample images under `test_img/` and the pretrained weights under `checkpoints/`.

## Training

Prepare the HDF5 training data with `dataprocessing.py` and `dataprocessing_text_npy.py`, update the dataset paths in the training scripts if necessary, and run:

```bash
python Train_stage1.py
python Train_stage2.py
```

## License

This project is released under the [MIT License](LICENSE).
