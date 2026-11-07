# Capture videos

Put the three videos of a performer here. Everything in this directory except this file is
gitignored: footage must never be committed.

| File                    | Content                                              |
| ----------------------- | ---------------------------------------------------- |
| `body_turntable.mp4`    | Full body on the turntable, one or more revolutions  |
| `head_static.mp4`       | Head on the turntable, neutral face                  |
| `head_expressions.mp4`  | Head facing the camera, face moving through expressions |

Other names or locations work too: point `capture` in `configs/default.yaml`, or a copy of it,
at the files. Check what was found with:

```sh
uv run avatar-splats inspect
```
