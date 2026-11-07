# avatar-splats

Re-posable Gaussian splat avatars for real-time use, built from three turntable videos of a
performer: a full-body turntable, a head turntable, and a head making expressions.

The Gaussians are bound to the triangles of one skinned mesh: the [Anny](https://github.com/naver/anny)
body with its head replaced by the [ICT-FaceKit](https://github.com/USC-ICT/ICT-FaceKit) face
model. Posing the skeleton and setting the face's ARKit-named expressions moves the Gaussians with
the mesh, so a game engine drives the avatar the way it drives a skinned character.

## Pipeline

| Stage       | Package                     | Produces                                              |
| ----------- | --------------------------- | ----------------------------------------------------- |
| Decode      | `avatar_splats.data`        | Frames from the capture videos                        |
| Preprocess  | `avatar_splats.preprocess`  | Masks, body keypoints, face landmarks per frame       |
| Cameras     | `avatar_splats.cameras`     | The fixed camera and the turntable angle per frame    |
| Body models | `avatar_splats.body_models` | Anny, ICT-FaceKit, and the avatar mesh joining them   |
| Fitting     | `avatar_splats.fitting`     | Body shape and pose, head identity, pose, expressions |
| Gaussians   | `avatar_splats.gaussians`   | Gaussians bound to the mesh, and their rendering      |
| Training    | `avatar_splats.training`    | Trained Gaussians and their held-out quality          |
| Export      | `avatar_splats.export`      | The `.avatar` file, and a reference runtime           |

Each stage writes its results under `outputs/`, so a later stage can be rerun, and any stage
inspected in a notebook, without repeating the ones before it.

### How it works

1. **Preprocess.** RT-DETR finds the performer in the first frame, SAM2.1 tracks their silhouette
   through the video, ViTPose finds 17 body keypoints, and MediaPipe 478 face landmarks.
2. **Fit the body.** The camera is fixed and the performer turns. Anny is fitted to all body
   frames at once: one shape, one pose, a turntable angle per frame (initialised from the
   shoulders), small head movements, and smooth per-vertex offsets for detail. Silhouettes are
   rendered differentiably by splatting points on the mesh with gsplat.
3. **Fit the head.** ICT-FaceKit is fitted to both head videos: one identity, and a head pose
   and 53 expression weights per frame, to the face landmarks and the silhouette above the chin.
4. **Build the avatar.** The fitted head is placed on the fitted body by matching the two heads'
   surfaces, both meshes are cut at a seam on the neck, and the head's vertices take their
   skinning weights from the body; the eyeballs follow Anny's eye bones.
5. **Train.** A fixed budget of Gaussians, denser on the head, is spread over the mesh. Each
   keeps its place, rotation and size relative to its triangle, so it follows any pose or
   expression. They are trained against every frame of all three videos.
6. **Export.** Mesh, skeleton, skinning, expression blendshapes and Gaussians go into one
   binary file; `export/avatar_file.py` documents the format and `export/runtime.py` is a
   reference for what a game runtime does with it each frame.

## Setup

Requires [uv](https://docs.astral.sh/uv/), [Task](https://taskfile.dev) and an NVIDIA GPU.
`task --list` shows every task; each is a plain `uv run` command you can also type yourself.

```sh
task setup
git clone --depth 1 https://github.com/USC-ICT/ICT-FaceKit.git assets/ict-facekit
```

Put the capture videos in `examples/` (see [examples/README.md](examples/README.md)), then:

```sh
task inspect                 # check the videos are found
task pipeline                # preprocess, fit, train, export
task notebook                # JupyterLab on notebooks/, one notebook per stage
```

Or one stage at a time: `task preprocess`, `task fit`, `task train`, `task export`. Models are
downloaded on first use: the detector, SAM2 and ViTPose from Hugging Face, the face landmarker
into `assets/`, Anny's data into `~/.cache/anny`.

## Layout

```
configs/      pipeline configs; paths are relative to the config file
src/          the avatar_splats package: all pipeline logic lives here
notebooks/    manual testing and visualization, one per stage; they import, never define, logic
tests/        unit tests on synthetic data, never on footage
examples/     capture videos (gitignored)
assets/       downloaded models (gitignored)
outputs/      stage results (gitignored)
```

## Development

```sh
task check                   # lint and tests
task format
```

Notebook outputs are stripped on commit, since they can contain frames of the footage.
