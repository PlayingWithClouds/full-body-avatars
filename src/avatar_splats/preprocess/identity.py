"""Who a face is: to keep one actor apart from the others across many videos.

Faces are aligned upright first: a similarity transform (rotation, scale, shift) takes their eyes,
nose and, where known, mouth corners onto ArcFace's template (``TEMPLATE``), so a tilted head
looks the same as an upright one. A small face-recognition network (InsightFace's MobileFaceNet,
``w600k_mbf``, 13 MB, on the CPU) turns the aligned 112-pixel face into a unit vector; two faces
of one person have a cosine similarity of about 0.4 and more, of two people about 0.2 and less.

The actor is a reference photo's face, or else the face seen most often (``main_identity``).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

TEMPLATE = np.array(
    [(38.2946, 51.6963), (73.5318, 51.5014), (56.0252, 71.7366), (41.5493, 92.3655),
     (70.7299, 92.2041)]
)  # fmt: skip
"""ArcFace's 112 by 112 face: the eye on the image's left, the other eye, the nose tip, the mouth
corners left and right."""
FACE_SIZE = 112
MODEL = ("deepghs/insightface", "buffalo_s/w600k_mbf.onnx")
"""Hugging Face repository and file of the face-recognition network."""
NOSE_TIP = 1
MOUTH_CORNERS = (61, 291)
IRIS_CENTRES = (468, 473)
"""MediaPipe face mesh landmarks."""


def keypoint_face_points(keypoints: np.ndarray, min_confidence: float) -> np.ndarray | None:
    """``(3, 2)`` the eyes (the image's left first) and nose of ``(17, 3)`` COCO keypoints, the
    first three of ``TEMPLATE``; None where any is not placed."""
    points = keypoints[[0, 1, 2]]
    if not (np.nan_to_num(points[:, 2]) >= min_confidence).all():
        return None
    eyes = keypoints[[1, 2], :2]
    eyes = eyes[np.argsort(eyes[:, 0])]
    return np.vstack([eyes, keypoints[0, :2]])


def landmark_face_points(landmarks: np.ndarray) -> np.ndarray | None:
    """``(5, 2)`` the ``TEMPLATE`` points of ``(478, 3)`` MediaPipe face landmarks: iris centres
    and mouth corners each the image's left first; None where they are not finite."""
    if not np.isfinite(landmarks).all():
        return None
    eyes = landmarks[list(IRIS_CENTRES), :2]
    mouth = landmarks[list(MOUTH_CORNERS), :2]
    return np.vstack(
        [eyes[np.argsort(eyes[:, 0])], landmarks[NOSE_TIP, :2], mouth[np.argsort(mouth[:, 0])]]
    )


def alignment(points: np.ndarray, size: int = FACE_SIZE) -> tuple[np.ndarray, float]:
    """The ``(2, 3)`` similarity transform taking ``(K, 2)`` face points (the first K of
    ``TEMPLATE``'s) onto the template scaled to ``size`` pixels, and the head's roll in degrees
    (positive: turned clockwise in the image)."""
    target = TEMPLATE[: len(points)] * size / FACE_SIZE
    matrix, _ = cv2.estimateAffinePartial2D(points.astype(np.float32), target.astype(np.float32))
    roll = float(np.degrees(np.arctan2(-matrix[0, 1], matrix[0, 0])))
    return matrix, -roll


def aligned_face(frame: np.ndarray, points: np.ndarray, size: int = FACE_SIZE) -> np.ndarray:
    """``(size, size, 3)`` the face of ``points`` cut from an RGB frame, upright."""
    matrix, _ = alignment(points, size)
    return cv2.warpAffine(frame, matrix, (size, size), flags=cv2.INTER_AREA, borderValue=0)


def eye_distance(points: np.ndarray) -> float:
    """Pixels between the eyes of face points."""
    return float(np.linalg.norm(points[1] - points[0]))


class FaceEmbedder:
    """The face-recognition network, on the CPU (downloads on first use)."""

    def __init__(self, repository: str = MODEL[0], filename: str = MODEL[1], threads: int = 4):
        import onnxruntime
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repository, filename)
        options = onnxruntime.SessionOptions()
        options.log_severity_level = 3  # its output is declared for one face; it takes many
        # few threads that sleep when idle: by default one per core, spinning between calls,
        # which held every core busy beside the threads that call it
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = onnxruntime.InferenceSession(
            path, options, providers=["CPUExecutionProvider"]
        )
        self.input = self.session.get_inputs()[0].name

    def __call__(self, faces: np.ndarray, batch: int = 64) -> np.ndarray:
        """``(N, 512)`` unit vectors of ``(N, 112, 112, 3)`` RGB uint8 aligned faces."""
        out = []
        for start in range(0, len(faces), batch):
            x = (faces[start : start + batch].astype(np.float32) - 127.5) / 127.5
            out.append(self.session.run(None, {self.input: x.transpose(0, 3, 1, 2)})[0])
        if not out:
            return np.zeros((0, 512), np.float32)
        embeddings = np.concatenate(out)
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


def main_identity(
    embeddings: np.ndarray, same: float, max_faces: int = 3000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """The face seen most often among ``(N, 512)`` unit embeddings (NaN rows ignored): the one
    with the most others at least ``same`` alike, and the mean of those, a unit vector. Also the
    ``(n,)`` indices of the faces it was made of. At most ``max_faces`` are compared, drawn at
    random."""
    known = np.flatnonzero(np.isfinite(embeddings).all(axis=1))
    if not len(known):
        raise ValueError("no faces to find the actor among")
    if len(known) > max_faces:
        known = np.sort(np.random.default_rng(seed).choice(known, max_faces, replace=False))
    faces = embeddings[known]
    alike = faces @ faces.T >= same
    centre = int(alike.sum(axis=1).argmax())
    members = known[alike[centre]]
    mean = embeddings[members].mean(axis=0)
    return mean / np.linalg.norm(mean), members


def reference_identity(paths: list[Path], embedder: FaceEmbedder, landmarker: Path) -> np.ndarray:
    """The mean unit embedding of the faces in reference photos of the actor, found by MediaPipe
    (``landmarker``: its model file)."""
    from avatar_splats.preprocess.face import detect_face_landmarks

    faces = []
    for path in paths:
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        points = landmark_face_points(detect_face_landmarks([image], landmarker, 1.0)[0])
        if points is not None:
            faces.append(aligned_face(image, points))
    if not faces:
        raise ValueError(f"no face found in {[str(p) for p in paths]}")
    mean = embedder(np.stack(faces)).mean(axis=0)
    return mean / np.linalg.norm(mean)
