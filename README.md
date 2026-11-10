# avatar-splats

Re-posable Gaussian splat avatars for real-time use, built from three turntable videos of a
performer: a full-body turntable, a head turntable, and a head making expressions.

The avatar is one skinned, textured mesh, the [Anny](https://github.com/naver/anny) body with its
head replaced by the [ICT-FaceKit](https://github.com/USC-ICT/ICT-FaceKit) face model, with
Gaussians bound to its triangles and drawn over it. Posing the skeleton and setting the face's
ARKit-named expressions moves the mesh and the Gaussians together, so a game engine drives the
avatar the way it drives a skinned character. The textured mesh shows wherever the videos saw too
little for Gaussians (armpits, soles), and alone it is a cheap fallback at a distance.

## Pipeline

| Stage       | Package                     | Produces                                              |
| ----------- | --------------------------- | ----------------------------------------------------- |
| Decode      | `avatar_splats.data`        | Frames from the capture videos                        |
| Preprocess  | `avatar_splats.preprocess`  | Masks, body and hand keypoints, face landmarks        |
| Cameras     | `avatar_splats.cameras`     | The fixed camera and the turntable angle per frame    |
| Body models | `avatar_splats.body_models` | Anny, ICT-FaceKit, and the avatar mesh joining them   |
| Fitting     | `avatar_splats.fitting`     | Body shape and pose, head identity, pose, expressions |
| Gaussians   | `avatar_splats.gaussians`   | Gaussians bound to the mesh, and their rendering      |
| Rendering   | `avatar_splats.rendering`   | Textured mesh, and the Gaussians composited over it   |
| Training    | `avatar_splats.training`    | Textures, Gaussians and their held-out quality        |
| Export      | `avatar_splats.export`      | The `.avatar` file, and a reference runtime           |
| Motion      | `avatar_splats.motion`      | Animation clips from videos of one or more people     |
| Faces       | `avatar_splats.faces`       | A video's face as the avatar's face                   |

Each stage writes its results under `outputs/`, so a later stage can be rerun, and any stage
inspected in a notebook, without repeating the ones before it.

### How it works

1. **Preprocess.** RT-DETR finds the performer in the first frame, SAM2.1 tracks their silhouette
   through the video, ViTPose finds 17 body keypoints, MediaPipe 21 landmarks per hand (on
   crops around the wrists) and 478 face landmarks.
2. **Fit the body.** The camera is fixed and the performer turns. Anny is fitted to all body
   frames at once: one shape, one pose including the fingers, a turntable angle per frame
   (initialised from the shoulders), small head movements, and MakeHuman's local changes of
   the limbs and torso (breast volume, arm and thigh girth, hip width, ...). Silhouettes are
   rendered differentiably by splatting points on the mesh with gsplat.
3. **Fit the head.** ICT-FaceKit is fitted to both head videos: one identity, and a head pose
   and 53 expression weights per frame, to the face landmarks and the silhouette above the chin.
   Its `eyeLook` blendshapes move only the eyelids, so each frame also turns the eyeballs to put
   the pupils on the detected iris centres.
4. **Build the avatar.** The fitted head is placed on the fitted body by matching the two heads'
   surfaces and refining against the body video. Both meshes are cut at a seam on the neck, the
   head's neck is bent onto the body's and a band of triangles stitches them into one surface.
   The head's vertices take their skinning weights from the body; the eyeballs follow Anny's
   eye bones. The body keeps Anny's UV layout, the head ICT-FaceKit's.
5. **Train.** Textures are baked from the frames: each texel takes the colour the frames that
   see it squarely agree on, and unseen texels are filled from their surroundings. Fine detail
   comes from one frame per patch of triangles, not an average, so it neither blurs nor ghosts
   where the mesh's shape is off. The expressions
   video colours only what the turntables never see, inside the mouth: averaged in everywhere,
   its wrinkles and moving brows smear into the neutral face. A fixed budget
   of Gaussians, denser on the face and sparse on the hands, is spread over the parts of the mesh
   the videos saw well. Each keeps its place, rotation and size relative to its triangle, so it
   follows any pose or expression. Textures and Gaussians are trained together against every
   frame of all three videos, drawn as the game draws them: the mesh, then the Gaussians over
   it. Each frame's pose, and each head video's colour grading, is corrected along the way.
6. **Export.** Mesh, UVs, textures, skeleton, skinning, expression blendshapes and Gaussians go
   into one binary file; `export/avatar_file.py` documents the format and `export/runtime.py` is a
   reference for what a game runtime does with it each frame. The eyes are not drawn from the
   textures, which ghost there: `export/eyes.py` measures the iris' and sclera's colour and the
   iris' size in the head turntable's frontal, open-eyed frames, and the runtime draws the
   eyeballs with an eye material from those (the viewer's is `viewer/src/lib/render/eyes.ts`). Lashes
   and brows are grown as strands bound to the face's triangles (`export/face_hair.py`), so they
   follow blinks and expressions: lashes along ICT-FaceKit's eyelash cards, brows where the baked
   head texture shows them, in colours measured in the head turntable. The inside of the mouth gets stock
   materials too (`export/mouth.py`): the videos see it too rarely to bake.
7. **Motion** (any video, separately from the capture). RT-DETR finds up to `max_people` people
   on the frame where most are seen and SAM2.1 follows each, forwards and backwards. Each person
   gets ViTPose keypoints, MediaPipe hands and MediaPipe's ARKit face blendshapes (mapped to
   ICT-FaceKit's names). The avatar's own Anny body is fitted to each person, frame by frame:
   root placement started from the best of 24 orientations (standing and lying) chosen
   consistently over time, then every bone and finger. One camera sees no depth, so smooth
   motion, a pose prior, knees bending only backwards, bodies of different people not passing
   through each other, and silhouettes compared only where no one else is in front fill in what
   it can't see. Long videos are tracked in stretches and fitted in overlapping windows, so
   memory stays bounded and each phase stops once its loss levels off. The clip stores
   bind-relative bone rotations, root motion and expressions per person, on a levelled floor.

### Genitals

Anny has neither genitals nor an anus, and an eye-level orbit sees too little of them to fit
any, so `body_models/genitals.py` generates them on the fitted body while the avatar is built.
A fine surface is laid along the crotch's midline, from the mons to the buttocks, measured along
the skin so it follows the inner thighs, and stitched into a hole cut in Anny's crotch. The labia
and vestibule are taken from a sculpt (`assets/genitals/vagina.stl`, ours, not in the
repository): its height less a blurred copy, scaled to her size; the hood in front of it, the
perineum and a recessed anus with fine radial folds are generated. Each opening continues as a blind canal into the pelvis. Shapes are blendshapes
beside the face's: `vulvaSpread`, and `vaginaOpen1..4` and `anusOpen1..4`, which open an
entrance and successive quarters of its canal, so an engine can open a canal as deep as
something has entered it. The skin around the genitals takes the skin's skinning weights; the
canals follow the pelvis. Their outside is baked from the frames like the rest; their walls,
which no frame sees, get a mucosa colour. Sizes are in `body_models.genitals`.

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

Or one stage at a time: `task preprocess`, `task fit`, `task train`, `task export`. Motion from
any video: `task motion -- path/to/video.mp4` (results in `outputs/motion/<video name>/`,
inspected in `notebooks/06_motion.ipynb`). A head video's face becomes the avatar's face (ICT expression weights, head and eye rotation per
frame): `task faces` (default `head_expressions`, or `task faces -- head_static`), written to
`outputs/faces/<video>.{npz,faces.json,faces.jpg}`. MediaPipe's ARKit blendshapes, calibrated to
her neutral frontal frames, give the expressions; all 468 mesh landmarks refine them, with a prior
holding them near the blendshapes. `task viewer` shows a grid of heads, one per frame, each beside
the video's crop of that frame (`?faces=/faces/head_expressions.faces.json`, or drop the file; the
crops need the URL). Only the textured head is drawn, not a Gaussian layer per frame. Wrinkles
(nose, frown lines) are not geometry and the texture is neutral, so they do not show. Models are
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
