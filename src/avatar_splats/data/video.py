"""Reading capture videos."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    num_frames: int

    @property
    def duration(self) -> float:
        """Length in seconds."""
        return self.num_frames / self.fps


def probe(path: str | Path) -> VideoInfo:
    """Describe the first video stream of a file.

    Containers do not always record their frame count; then the packets are counted, which
    reads the file but does not decode it.
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        num_frames = stream.frames or sum(1 for p in container.demux(stream) if p.size)
        return VideoInfo(
            width=stream.codec_context.width,
            height=stream.codec_context.height,
            fps=float(stream.average_rate),
            num_frames=num_frames,
        )


def read_frames(
    path: str | Path, step: int = 1, max_frames: int | None = None
) -> Iterator[np.ndarray]:
    """Decode every ``step``-th frame as an RGB ``uint8`` array of shape ``(H, W, 3)``."""
    if step < 1:
        raise ValueError(f"step must be at least 1, got {step}")

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        yielded = 0
        for index, frame in enumerate(container.decode(stream)):
            if max_frames is not None and yielded >= max_frames:
                return
            if index % step == 0:
                yield frame.to_ndarray(format="rgb24")
                yielded += 1
