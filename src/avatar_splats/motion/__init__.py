"""Motion from videos of other people, one or several, as animation clips for the avatar."""

from avatar_splats.motion.clip import MotionClip, bind_relative_pose, motion_clip, write_motion
from avatar_splats.motion.fit import MotionFit, fit_motion, fitted_people
from avatar_splats.motion.observe import Observations, observe_people
from avatar_splats.motion.pipeline import RESULTS, extract_motion, motion_file_path, motion_path

__all__ = [
    "RESULTS",
    "MotionClip",
    "MotionFit",
    "Observations",
    "bind_relative_pose",
    "extract_motion",
    "fit_motion",
    "fitted_people",
    "motion_clip",
    "motion_file_path",
    "motion_path",
    "observe_people",
    "write_motion",
]
