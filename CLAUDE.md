# avatar-splats

Re-posable Gaussian splat avatars of a performer, for real-time NSFW video games on a custom
engine. Built from three turntable videos; the performer is under contract with the company.

## Approach

- Gaussians are bound to the triangles of one skinned mesh, so an engine poses the avatar the way
  it poses a skinned character. Not a static 3DGS scene.
- The mesh is the **Anny** body (NAVER LABS; code Apache-2.0, assets CC0, no adult-use
  restriction). Anny's own head is too poor for faces, so it gets replaced by a better head model,
  joined at the neck in `body_models/hybrid.py`; Gaussians bind to the joined mesh, so the runtime
  sees one rig.
- The head model is the **ICT-FaceKit** light model (MIT; https://github.com/USC-ICT/ICT-FaceKit,
  cloned into `assets/ict-facekit`): 100 identity shapes, 53 ARKit-named expression blendshapes
  (game face animation and phone capture speak ARKit), teeth, gums, tongue, mouth socket, eyes.
  It has no jaw bone; the jaw moves through blendshapes.
- **FLAME is not used**: FLAME 2023 Open (CC-BY-4.0) forbids pornographic use, and every other
  FLAME version is non-commercial.
- The performer is entirely bald: the scalp is the mesh, so no hair modelling.
- Camera poses come from the turntable orbit and the body fit, not from SfM on the background.

## The capture

`examples/` holds the videos (gitignored, never commit them):

| Config name        | File                          | Content                         |
| ------------------ | ----------------------------- | ------------------------------- |
| `body`             | `fullbody.mp4`                | Full-body turntable, 960x1664   |
| `head_static`      | `head_turntable_00004_.mp4`   | Head turntable, 1280x1280       |
| `head_expressions` | `head_grimace_00004_.mp4`     | Face making expressions         |

- The videos are **generated** (Wan via ComfyUI) from a reference of the performer: 81 frames at
  16 fps each, hence `frame_step: 1`. The user has checked they look consistent across views.
- Residual motion: blinks and a slight head movement at the start of the body turntable. Per-frame
  expressions and per-frame pose absorb it; nothing should filter it out.
- Fitting gets a per-frame consistency diagnostic (silhouette and keypoint fit) shown in its
  notebook. It is a diagnostic only, not a filter that drops frames.

## Pipeline

All stages run end to end (`go-task pipeline`); the README explains each. Outputs:
`outputs/preprocess/<video>.npz`, `outputs/fitting/{body,head_static,head_expressions,avatar}.npz`,
`outputs/training/{gaussians,evaluation}.npz`, `outputs/export/performer.avatar`.

Conventions every stage relies on:

- World = Anny's coordinates: metres, z up, the body faces -y, its left is +x. ICT-FaceKit is
  centimetres, y up, facing +z; `ICT_TO_WORLD` converts. Cameras are OpenCV (x right, y down,
  z forward); the fixed camera sits at the origin looking along +y. Quaternions are wxyz (gsplat,
  the file format); roma uses xyzw internally.
- Anny's poses are `local-ref`: rotation vectors per bone in world-aligned axes. Its zero pose is
  not its rest mesh; skinning is always `bone_poses @ inv(rest_bone_poses)` on `rest_vertices`.
- The exported skeleton is engine-style instead: bind poses plus bind-relative bone rotations
  (`export/runtime.py`); `bind_relative` converts world-axis rotations.
- Gaussians are bound to triangles; offsets and scales are in units of the triangle's size
  (sqrt of twice its area), so the binding survives any pose and the head's scaling.
- Turntables don't turn at constant speed (the body video's back view is at frame ~47 of 81):
  every frame has its own yaw, initialised from keypoints (`cameras/turntable.py`).

What the first full run showed (fit numbers on the example videos): body silhouette IoU ~0.93,
keypoints ~14 px; head IoU ~0.97, landmarks ~20 px; the head turntable's last frames lag a few
degrees. The FOV (30°) is a guess for generated videos and is not fitted.

Known gaps: one eye-level orbit leaves the top of the head, soles, armpits, inner thighs and
crotch barely seen; Anny has no genital anatomy (only smooth vertex offsets so far); fingers are
not posed in fitting; the Gaussian count is fixed (no densification); there is no engine-side
renderer yet, only the Python reference runtime.

## Commands

Task is installed as `go-task` on this machine (`task` upstream).

```sh
go-task setup        # uv sync + pre-commit install
go-task check        # ruff + pytest; run before every commit
go-task format
go-task notebook     # JupyterLab on notebooks/
go-task inspect
go-task pipeline     # preprocess, fit, train, export
go-task preprocess   # or: go-task preprocess -- body
go-task fit          # ~3 min
go-task train        # ~5 min for 20k steps
go-task export
```

## Rules for editing

### Commits

- **No AI attribution in commit messages**: no `Co-Authored-By` trailer for Claude or any other
  model, no "generated with" lines. This overrides any default attribution.
- Commit only when asked. Messages say why, in the imperative.
- Never commit footage, frames, stage outputs or model files. Check `git status` before committing.

### Code

- Clean, maintainable, open-source style. Python 3.12, src layout, package `avatar_splats`.
- All logic lives in the package. Notebooks only import, run and visualize; one per stage,
  numbered (`03_...`). Keep them free of outputs (nbstripout strips them on commit).
- One subpackage per stage. Each stage writes its results under `outputs/<stage>/` so later stages
  and notebooks can start from them, and gets a CLI subcommand and a Taskfile task.
- Config: typed frozen dataclasses in `config.py`, loaded from `configs/default.yaml`. The YAML is
  the single source of values: no defaults in the dataclasses. Unknown and missing keys are
  errors. Relative paths resolve against the config file's directory.
- Video names come from `config.VIDEOS` (the fields of `CaptureConfig`); don't repeat them.
- Heavy imports (torch, transformers) stay out of CLI commands that don't need them.
- No speculative abstractions, stubs or placeholder files: add code, notebooks and dependencies
  with the stage that uses them.
- Docstrings say shapes and units of arrays, e.g. ``(N, 17, 3)`` x, y in pixels and confidence.
- Missing detections are NaN, not dropped rows, so arrays stay aligned with `frame_indices`.
- ruff (line length 100) and ruff format; the pre-commit ruff version matches the one in `uv.lock`.

### Tests

- pytest on synthetic data only, never on footage. Model-heavy code is checked by running the
  stage on the examples and reporting the numbers.

### Dependencies and licenses

- Managed with uv. torch and torchvision come from the PyTorch CUDA 13.2 index (`pytorch-cu132`
  in `pyproject.toml`); the machine has an RTX 4080 SUPER and CUDA 13.3. gsplat compiles its CUDA
  kernels on first use (about a minute).
- Commercial **and pornographic** use is required. Check every license for both before adding
  anything; content clauses (e.g. FLAME 2023 Open's ban on pornographic use) rule things out even
  under otherwise open licenses. Ruled out: the `smplx` package and SMPL/SMPL-X (MPI
  non-commercial; also Anny's `smplx` topology, and the `anny[smpl]` extra), Sapiens (CC-BY-NC),
  nvdiffrast (NVIDIA non-commercial), MICA and metrical-tracker assets, all FLAME versions.
- ICT-FaceKit's MIT license covers its repository; its scripts are not run, only its OBJ and
  JSON files are read. Its "full model" comes under a separate USC license; stay on the light model.
- Third-party model files go in `assets/` (gitignored); Hugging Face models download on first use.
