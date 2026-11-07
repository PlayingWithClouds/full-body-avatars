"""Re-posable Gaussian splat avatars from turntable video.

The pipeline runs in stages, each in its own subpackage and each writing its results to disk:

1. ``data``        decode the capture videos into frames
2. ``preprocess``  segmentation masks, body keypoints and face landmarks per frame
3. ``cameras``     camera intrinsics and the turntable orbit
4. ``body_models`` the Anny body, the FLAME head and the stitched hybrid mesh
5. ``fitting``     body shape, per-frame pose and facial expressions
6. ``gaussians``   Gaussians bound to the hybrid mesh, and their rendering
7. ``training``    photometric optimization of the bound Gaussians
8. ``export``      the skinned Gaussian format read by the game runtime
"""

__version__ = "0.1.0"
