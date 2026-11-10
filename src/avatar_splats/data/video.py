"""Reading capture videos."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from fractions import Fraction
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


def frame_size(path: str | Path) -> tuple[int, int]:
    """Width and height of a video's frames, read from its header."""
    with av.open(str(path)) as container:
        context = container.streams.video[0].codec_context
        return context.width, context.height


def open_video(path: str | Path, gpu: bool = False) -> av.container.InputContainer:
    """Open a video for decoding; ``gpu``: on the GPU's video decoder (NVDEC) where it can,
    else on the CPU. The GPU decodes about as fast and leaves the CPU free: 4K H.264 took one
    core instead of eight."""
    if not gpu:
        return av.open(str(path))
    from av.codec.hwaccel import HWAccel

    return av.open(str(path), hwaccel=HWAccel(device_type="cuda", allow_software_fallback=True))


def scaled_size(width: int, height: int, max_side: int | None) -> tuple[int, int]:
    """Even width and height of a ``width`` by ``height`` image whose longer side is at most
    ``max_side`` (unchanged where it is not longer, or ``max_side`` is None)."""
    factor = 1.0 if max_side is None else min(1.0, max_side / max(width, height))
    return round(width * factor) // 2 * 2, round(height * factor) // 2 * 2


def read_frames(
    path: str | Path,
    step: int = 1,
    max_frames: int | None = None,
    start: int = 0,
    max_side: int | None = None,
    gpu: bool = False,
) -> Iterator[np.ndarray]:
    """Decode every ``step``-th frame from frame ``start`` on, as RGB ``uint8`` ``(H, W, 3)``.

    A later ``start`` seeks to the keyframe before it instead of decoding everything before;
    the frames after the seek are numbered by their timestamps. ``max_side``: frames scaled
    down so their longer side is at most this (area-averaged, in the same pass as the colour
    conversion); ``gpu``: decoded on the GPU (``open_video``).
    """
    if step < 1:
        raise ValueError(f"step must be at least 1, got {step}")

    with open_video(path, gpu) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        width, height = stream.codec_context.width, stream.codec_context.height
        size = scaled_size(width, height, max_side)
        rate, offset = stream.average_rate, stream.start_time or 0
        if start:
            container.seek(offset + int(start / rate / stream.time_base), stream=stream)
        yielded = 0
        for counted, frame in enumerate(container.decode(stream)):
            index = round((frame.pts - offset) * stream.time_base * rate) if start else counted
            if index < start:
                continue
            if max_frames is not None and yielded >= max_frames:
                return
            if (index - start) % step == 0:
                if size == (width, height):
                    yield frame.to_ndarray(format="rgb24")
                else:
                    yield frame.reformat(
                        width=size[0], height=size[1], format="rgb24", interpolation="AREA"
                    ).to_ndarray()
                yielded += 1


def write_video(path: str | Path, frames: Iterable[np.ndarray], fps: float) -> None:
    """Encode RGB ``uint8`` frames of shape ``(H, W, 3)`` as an H.264 video."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = None
        for frame in frames:
            if stream is None:
                stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1000))
                stream.height, stream.width = (s - s % 2 for s in frame.shape[:2])
                stream.pix_fmt = "yuv420p"
            frame = np.ascontiguousarray(frame[: stream.height, : stream.width])
            container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
        if stream is not None:
            container.mux(stream.encode())
