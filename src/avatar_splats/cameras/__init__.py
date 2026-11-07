"""Cameras, and the turntable's rotation as seen from a fixed camera.

The camera never moves; the performer turns in front of it. World coordinates are Anny's: metres,
z up, and the camera sits on the -y side looking along +y at the performer's front. Cameras follow
the OpenCV convention: x right, y down, z forward.
"""

from avatar_splats.cameras.camera import Camera, intrinsics, world_to_camera
from avatar_splats.cameras.turntable import body_yaw, head_yaw, unwrap

__all__ = ["Camera", "body_yaw", "head_yaw", "intrinsics", "unwrap", "world_to_camera"]
