from pathlib import Path

import av
import numpy as np
import pytest


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """A 10-frame, 32x48 video whose frame ``i`` is filled with gray level ``20 * i``."""
    path = tmp_path / "synthetic.mp4"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width, stream.height = 32, 48
        stream.pix_fmt = "yuv420p"
        stream.options = {"qp": "0"}
        for i in range(10):
            image = np.full((48, 32, 3), 20 * i, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    return path
