"""Command line entry point: one subcommand per pipeline stage."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from avatar_splats.config import VIDEOS, Config, load_config
from avatar_splats.data import probe


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="avatar-splats", description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml", help="pipeline config file")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inspect", help="describe the capture videos")
    preprocess_parser = commands.add_parser(
        "preprocess", help="segment the performer, find body keypoints and face landmarks"
    )
    preprocess_parser.add_argument(
        "videos", nargs="*", choices=VIDEOS, help="videos to process (default: all)"
    )

    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.command == "inspect":
        return inspect(config)
    if args.command == "preprocess":
        return preprocess(config, args.videos or VIDEOS)
    raise AssertionError(f"unhandled command {args.command}")


def inspect(config: Config) -> int:
    missing = 0
    for name in VIDEOS:
        path = getattr(config.capture, name)
        if not path.exists():
            print(f"{name:<18} missing: {path}")
            missing += 1
            continue
        info = probe(path)
        print(
            f"{name:<18} {info.width}x{info.height}  {info.fps:.2f} fps  "
            f"{info.num_frames} frames  {info.duration:.1f} s"
        )
    return 1 if missing else 0


def preprocess(config: Config, videos: Sequence[str]) -> int:
    # Imported here so that commands which need no models start without loading torch.
    from avatar_splats.preprocess import preprocess_capture

    for name, path in preprocess_capture(config, videos).items():
        print(f"{name:<18} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
