# Image Compression with SVD

This project demonstrates grayscale image compression with singular value decomposition (SVD). It converts images into matrices, builds low-rank approximations, and reports the normalized Frobenius reconstruction error for selected ranks.

The original notebook has been preserved in `notebooks/Compression.ipynb`; the runnable Python version lives in `src/image_compression/main.py`.

## Project Structure

```text
.
|-- data/                         # Input images
|   |-- Berkeley1.png
|   `-- Berkeley2.png
|-- notebooks/
|   `-- Compression.ipynb         # Original notebook
|-- outputs/                      # Generated plots, ignored by git
|-- src/
|   `-- image_compression/
|       |-- __init__.py
|       `-- main.py               # CLI analysis script
|-- requirements.txt
`-- README.md
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Usage

Run the default analysis on both Berkeley images:

```bash
python -m src.image_compression.main
```

This saves singular-value plots and rank-approximation figures to `outputs/`, and prints a table of reconstruction errors for each image.

Use custom images, ranks, or output directory:

```bash
python -m src.image_compression.main \
  --images data/Berkeley1.png data/Berkeley2.png \
  --ranks 20 50 100 \
  --output-dir outputs
```

## Method

For an image matrix `A`, the script computes the compact SVD:

```text
A = U Sigma V^T
```

A rank-`k` approximation keeps only the first `k` singular values:

```text
A_k = U[:, :k] Sigma[:k] V^T[:k, :]
```

The reported error is:

```text
e_k = ||A - A_k||_F^2 / ||A||_F^2
```

Lower values mean the rank-`k` reconstruction preserves more of the original image energy.

## Dependencies

- `numpy` for matrix operations and SVD
- `matplotlib` for plots
- `opencv-python` for image loading and grayscale conversion
