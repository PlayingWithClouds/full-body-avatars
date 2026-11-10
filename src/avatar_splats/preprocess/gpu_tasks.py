"""MediaPipe tasks on the GPU where there is one.

MediaPipe runs its GPU models through OpenGL ES on EGL. Where a desktop session lets EGL choose,
it may pick Mesa, which has only a software renderer, and MediaPipe then refuses to start. So
NVIDIA's EGL is chosen explicitly where it is installed, unless the environment already chose;
this has to happen before anything in the process opens EGL, so it happens on import. Where the
GPU still cannot be had, a task runs on the CPU and a warning says so.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from mediapipe.tasks.python import BaseOptions

logger = logging.getLogger(__name__)

NVIDIA_EGL = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
"""glvnd's description of NVIDIA's EGL library."""

if NVIDIA_EGL.exists():
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", str(NVIDIA_EGL))


def create_task(task, options_type, model_path: Path, gpu: bool, **options):
    """``task.create_from_options`` with the model at ``model_path``, on the GPU if ``gpu`` and
    MediaPipe gets one, on the CPU otherwise."""
    if gpu:
        base = BaseOptions(model_asset_path=str(model_path), delegate=BaseOptions.Delegate.GPU)
        try:
            return task.create_from_options(options_type(base_options=base, **options))
        except RuntimeError as error:
            logger.warning(
                "%s runs on the CPU: %s", task.__name__, str(error).splitlines()[0][:300]
            )
    base = BaseOptions(model_asset_path=str(model_path))
    return task.create_from_options(options_type(base_options=base, **options))
