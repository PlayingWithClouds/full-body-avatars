"""Preview DiffLocks strands: a PNG from the front, side and back, and an OBJ of polylines.

Usage: python strands_preview.py <difflocks_output_strands.npz> [--strands 20000]

The OBJ holds a random subset of the strands as line elements; Blender imports them as edges
(File > Import > Wavefront), and Object > Convert > Curve turns them into curves.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path)
    parser.add_argument("--strands", type=int, default=20000, help="strands kept for the OBJ")
    parser.add_argument(
        "--head",
        type=Path,
        default=Path(__file__).parent.parent
        / "difflocks/data_loader/difflocks_bodydata/smplx_base.ply",
        help="body mesh drawn in grey behind the strands, if it exists",
    )
    args = parser.parse_args()

    positions = np.load(args.npz)["positions"]  # (strands, points, 3)
    strands, points, _ = positions.shape
    lengths = np.linalg.norm(np.diff(positions, axis=1), axis=-1).sum(axis=1)
    print(f"{strands} strands of {points} points, length {np.median(lengths):.3f} median")
    print(f"raw size {positions.nbytes / 1e6:.0f} MB as {positions.dtype}")

    rng = np.random.default_rng(0)
    kept = rng.choice(strands, min(args.strands, strands), replace=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    shown = positions[kept[:4000]]
    # DiffLocks works in SMPL-X's axes: y up, the face looking along +z, its left at +x.
    up = 1
    views = (("front", 0, -1), ("side", 2, 1), ("back", 0, 1))
    head = args.head if args.head.exists() else None
    head_points = _ply_vertices(head) if head else None
    for ax, (name, across, sign) in zip(axes, views, strict=True):
        if head_points is not None:
            near = head_points[head_points[:, up] > positions[..., up].min() - 0.05]
            ax.scatter(sign * near[:, across], near[:, up], s=0.2, color="0.75", zorder=0)
        for strand in shown:
            ax.plot(sign * strand[:, across], strand[:, up], lw=0.15, color="saddlebrown", alpha=0.5)
        ax.set_aspect("equal")
        ax.set_title(name)
        ax.axis("off")
    preview = args.npz.with_name("strands_preview.png")
    plt.tight_layout()
    plt.savefig(preview, dpi=150)

    obj = args.npz.with_name("strands.obj")
    with obj.open("w") as out:
        for strand in positions[kept]:
            np.savetxt(out, strand, fmt="v %.5f %.5f %.5f")
        indices = np.arange(len(kept) * points).reshape(len(kept), points) + 1
        for line in indices:
            out.write("l " + " ".join(map(str, line)) + "\n")
    print(f"wrote {preview} and {obj} ({len(kept)} strands)")


def _ply_vertices(path: Path) -> np.ndarray:
    import trimesh

    return np.asarray(trimesh.load(path, process=False).vertices)


if __name__ == "__main__":
    main()
