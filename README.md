# avatar-splats

Re-posable Gaussian splat avatars for real-time use, built from three turntable videos of a
performer: a full-body turntable, a static head turntable, and a head making expressions.

The Gaussians are bound to the triangles of one skinned mesh: the [Anny](https://github.com/naver/anny)
body with its head replaced by [FLAME](https://flame.is.tue.mpg.de). Posing the skeleton and
setting FLAME expressions moves the Gaussians with the mesh, so a game engine can drive the
avatar the same way it drives a skinned character.

## Pipeline

| Stage         | Package                     | Produces                                    |
| ------------- | --------------------------- | ------------------------------------------- |
| Decode        | `avatar_splats.data`        | Frames from the capture videos              |
| Preprocess    | `avatar_splats.preprocess`  | Masks, body keypoints, face landmarks       |
| Cameras       | `avatar_splats.cameras`     | Intrinsics and turntable orbit              |
| Body models   | `avatar_splats.body_models` | The hybrid Anny + FLAME mesh                |
| Fitting       | `avatar_splats.fitting`     | Shape, per-frame pose, expressions          |
| Gaussians     | `avatar_splats.gaussians`   | Mesh-bound Gaussians and their rendering    |
| Training      | `avatar_splats.training`    | Optimized Gaussians                         |
| Export        | `avatar_splats.export`      | Skinned Gaussian file for the game runtime  |

Each stage writes its results under `outputs/`, so a later stage can be rerun, and any stage
inspected in a notebook, without repeating the ones before it.

## Setup

Requires [uv](https://docs.astral.sh/uv/), [Task](https://taskfile.dev) and an NVIDIA GPU.
`task --list` shows every task; each is a plain `uv run` command you can also type yourself.

```sh
task setup
```

Put the capture videos in `examples/` (see [examples/README.md](examples/README.md)), then:

```sh
task inspect
task preprocess              # or name videos: task preprocess -- body
task notebook                # JupyterLab on notebooks/
```

Models are downloaded on first use: the detector, SAM2 and ViTPose from Hugging Face, the face
landmarker into `assets/`.

## Layout

```
configs/      pipeline configs; paths are relative to the config file
src/          the avatar_splats package: all pipeline logic lives here
notebooks/    manual testing and visualization, one per stage; they import, never define, logic
tests/        unit tests on synthetic data, never on footage
examples/     capture videos (gitignored)
assets/       downloaded body models (gitignored)
outputs/      stage results (gitignored)
```

## Development

```sh
task check                   # lint and tests
task format
```

Notebook outputs are stripped on commit, since they can contain frames of the footage.
