# Model assets

Third-party models live here and are not committed. Each comes under its own license; check it
allows commercial and adult use before shipping anything built from it.

- **ICT-FaceKit** (MIT), the head model: clone it into `ict-facekit/`.

  ```sh
  git clone --depth 1 https://github.com/USC-ICT/ICT-FaceKit.git assets/ict-facekit
  ```

  Only its OBJ and JSON files are read; its scripts are never run.
- **MediaPipe face landmarker** (Apache-2.0): downloaded to `face_landmarker.task` on first use.

Anny's data (code Apache-2.0, assets CC0) comes with its Python package and is cached in
`~/.cache/anny`. Hugging Face models are cached in `~/.cache/huggingface`.
