# Capture videos

Put the three videos of a performer here. Everything in this directory except this file is
gitignored: footage must never be committed.

| Config name        | Content                                                   |
| ------------------ | --------------------------------------------------------- |
| `body`             | Full body on the turntable, one revolution or more        |
| `head_static`      | Head on the turntable, neutral face                       |
| `head_expressions` | Head facing the camera, face moving through expressions   |

`capture` in `configs/default.yaml` names the files. Check what was found with:

```sh
uv run avatar-splats inspect
```
