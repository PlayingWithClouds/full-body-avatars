"""Command line entry point: one subcommand per pipeline stage."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

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
    commands.add_parser("fit", help="fit the body and head models and build the avatar's mesh")
    commands.add_parser("train", help="train the avatar's Gaussians")
    commands.add_parser("export", help="write the avatar file for the game runtime")
    faces_parser = commands.add_parser(
        "faces", help="convert a video's face landmarks to the avatar's face, per frame"
    )
    faces_parser.add_argument(
        "videos",
        nargs="*",
        choices=[video for video in VIDEOS if video != "body"],
        help="head videos to convert (default: head_expressions)",
    )
    motion_parser = commands.add_parser(
        "motion", help="extract the motion of the people in a video as a clip for the avatar"
    )
    motion_parser.add_argument("video", type=Path, help="video of one or more people moving")
    motion_parser.add_argument("--name", help="name of the results (default: the video's)")
    motion_parser.add_argument(
        "--refit", action="store_true", help="fit again from the observations of an earlier run"
    )

    args = parser.parse_args(argv)
    config = load_config(args.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    # Stages are imported when run, so that commands which need no models start without torch.
    if args.command == "inspect":
        return inspect(config)
    if args.command == "preprocess":
        from avatar_splats.preprocess import preprocess_capture

        return report(preprocess_capture(config, args.videos or VIDEOS))
    if args.command == "fit":
        from avatar_splats.fitting import fit_capture

        return report(fit_capture(config))
    if args.command == "train":
        from avatar_splats.training import train_capture

        return report(train_capture(config))
    if args.command == "export":
        from avatar_splats.export import export_capture

        return report({"avatar": export_capture(config)})
    if args.command == "faces":
        from avatar_splats.faces import convert_video

        written = {}
        for video in args.videos or ["head_expressions"]:
            written |= {f"{video} {k}": path for k, path in convert_video(config, video).items()}
        return report(written)
    if args.command == "motion":
        from avatar_splats.motion import extract_motion

        return report(extract_motion(config, args.video, args.name or args.video.stem, args.refit))
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


def report(written: dict[str, Path]) -> int:
    for name, path in written.items():
        print(f"{name:<18} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
