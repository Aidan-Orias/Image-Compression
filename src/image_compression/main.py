"""Run SVD-based low-rank image compression experiments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_IMAGES = (
    PROJECT_ROOT / "data" / "Berkeley1.png",
    PROJECT_ROOT / "data" / "Berkeley2.png",
)
DEFAULT_RANKS = (30, 80, 100)


def load_grayscale_image(path: Path) -> np.ndarray:
    """Load an image as grayscale floats in the range [0, 1]."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image.astype(np.float32) / 255.0


def compute_svd(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the compact SVD for an image matrix."""
    return np.linalg.svd(image, full_matrices=False)


def low_rank_approximation(
    u_matrix: np.ndarray,
    singular_values: np.ndarray,
    vt_matrix: np.ndarray,
    rank: int,
) -> np.ndarray:
    """Build a rank-k approximation from a compact SVD."""
    return (u_matrix[:, :rank] * singular_values[:rank]) @ vt_matrix[:rank, :]


def normalized_frobenius_error(
    original: np.ndarray,
    approximation: np.ndarray,
    singular_values: np.ndarray,
) -> float:
    """Return ||A - Ak||_F^2 / ||A||_F^2."""
    squared_error = np.linalg.norm(original - approximation, "fro") ** 2
    squared_norm = np.sum(singular_values**2)
    return float(squared_error / squared_norm)


def plot_singular_values(
    singular_values: np.ndarray,
    image_name: str,
    output_dir: Path,
) -> None:
    """Plot and save singular values for one image."""
    figure, axis = plt.subplots()
    indices = np.arange(1, len(singular_values) + 1)

    axis.plot(indices, singular_values, marker="o", linestyle="None")
    axis.set_xlabel("Index k")
    axis.set_ylabel("Singular value sigma_k")
    axis.set_title(f"Singular values of {image_name}")
    axis.grid(True)

    figure.tight_layout()
    figure.savefig(output_dir / f"{image_name}_singular_values.png", dpi=200)
    plt.close(figure)


def plot_rank_approximations(
    image: np.ndarray,
    u_matrix: np.ndarray,
    singular_values: np.ndarray,
    vt_matrix: np.ndarray,
    ranks: list[int],
    image_name: str,
    output_dir: Path,
) -> list[tuple[int, float, float]]:
    """Plot and save low-rank reconstructions for one image."""
    figure, axes = plt.subplots(1, len(ranks), figsize=(4.5 * len(ranks), 4))
    axes = np.atleast_1d(axes)

    denominator = min(image.shape)
    results = []

    for axis, rank in zip(axes, ranks):
        approximation = low_rank_approximation(
            u_matrix,
            singular_values,
            vt_matrix,
            rank,
        )
        error = normalized_frobenius_error(image, approximation, singular_values)
        percent_used = 100.0 * rank / denominator
        results.append((rank, error, percent_used))

        axis.imshow(np.clip(approximation, 0.0, 1.0), cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(f"k={rank}\ne_k={error:.4f}, used={percent_used:.1f}%")
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_dir / f"{image_name}_rank_approximations.png", dpi=200)
    plt.close(figure)

    return results


def analyze_image(image_path: Path, ranks: list[int], output_dir: Path) -> None:
    """Run singular-value and low-rank approximation analysis for one image."""
    image_name = image_path.stem
    image = load_grayscale_image(image_path)
    u_matrix, singular_values, vt_matrix = compute_svd(image)

    max_rank = min(image.shape)
    invalid_ranks = [rank for rank in ranks if rank < 1 or rank > max_rank]
    if invalid_ranks:
        raise ValueError(
            f"{image_path.name} supports ranks 1 through {max_rank}; "
            f"invalid ranks: {invalid_ranks}"
        )

    plot_singular_values(singular_values, image_name, output_dir)
    results = plot_rank_approximations(
        image,
        u_matrix,
        singular_values,
        vt_matrix,
        ranks,
        image_name,
        output_dir,
    )

    print(f"\n{image_path.name}")
    print("rank\tenergy_error\tmatrix_dimension_used")
    for rank, error, percent_used in results:
        print(f"{rank}\t{error:.6f}\t{percent_used:.2f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compress grayscale images with low-rank SVD approximations.",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        type=Path,
        default=list(DEFAULT_IMAGES),
        help="Image files to analyze.",
    )
    parser.add_argument(
        "--ranks",
        nargs="+",
        type=int,
        default=list(DEFAULT_RANKS),
        help="SVD ranks to use for reconstructions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Directory for generated plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in args.images:
        analyze_image(image_path, args.ranks, output_dir)


if __name__ == "__main__":
    main()
