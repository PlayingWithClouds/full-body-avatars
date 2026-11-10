# Open items

## Plan: a pure mesh avatar

Decided 2026-11-08. Gaussians add little over the textured mesh (held-out PSNR on the performer:
+1.2 dB body, +1.1 dB head turntable, +2.9 dB expressions) and every fix so far has constrained
them into a texture. The performer is bald, the videos are generated and seen from one eye-level
orbit, so the detail they can teach fits a mesh: body and head become a skinned, textured mesh
only, with expression-dependent detail on the face. Gaussians stay in the code until the mesh
matches them, then go.

1. **Sharper baking** (in progress). Each frame split into broad colour, averaged over many
   frames (no exposure patches), and fine detail from the frames that see a place most squarely
   (`training/textures.py`). Check: mesh-only held-out PSNR (was 17.3 / 19.2 / 18.4 dB), the
   texture's patches, and that `view_spread` of the mesh stays near 0.
2. **More triangles.** Subdivide the joined mesh (65k triangles to 260k or more) in
   `body_models`, carrying UVs, skin weights, parts and the head's expression blendshapes along;
   new head vertices stay after `head_start`. Then a smooth displacement on the denser mesh,
   fitted to the now aligned frames, with a strong smoothness prior: the old displacement hid pose
   errors.
3. **Expression detail.** What Gaussians gave the face under expressions (wrinkles, mouth inside,
   teeth): wrinkle textures blended by expression weights, as games do, baked from the
   expression video's frames by expression.
4. **Training without Gaussians.** Pose and colour corrections refined against the textured
   mesh alone; export and viewer without Gaussians.

## Other open items

Noted 2026-11-08, to pick up in a later session. Check each against the code first.

1. **Neck seam in the head videos.** Head videos are compared above the fitted chin
   (`training/views.py`, `head_views`). Just there, the head mesh's neck is bent onto Anny's and
   disagrees with the video's neck, so the silhouette loss pulled Gaussians off the skin (the
   "collar"). The hard size and offset limits now stop the symptom, not the conflict. Fix: end
   the compared region a few centimetres above the chin, or leave the mask loss out near that line.

2. **Patchy texture.** Triangle-shaped colour patches on the breasts and upper arms: neighbouring
   triangles are baked from different frames with slightly different brightness. Fix: blend the
   views more smoothly when baking (`training/textures.py`).

3. **Soft edges from the backdrop colour.** A plain threshold against the backdrop gives a worse
   mask than SAM: the generated backdrop is light grey with blotchy shading, floor shadows and
   compression banding (agreement with SAM's masks 56-71%). Instead keep SAM's mask and estimate a
   soft alpha only within a few pixels of its edge, from the known backdrop colour. Then
   `edge_band` could shrink and the silhouette would be sharper.

4. **Per-frame finger poses.** The hands have no Gaussians (`hand_density: 0`): fingers are posed
   once per video and never match every frame, and Gaussians smeared the video's fingers around
   the mesh's. Per-frame finger poses would let them come back.

5. **Size limit checked on the rest mesh only.** Strong expressions (`mouthRollUpper`, `jawOpen`,
   `eyeBlink_R`) stretch triangles up to 2.4x, so Gaussians around the mouth and eyes can reach
   about 15 mm instead of 6.

6. **Breasts larger than Anny can make them.** The `cupsize` phenotype is at its maximum; the
   local breast changes cover part of the rest.

7. **gsplat's antialiased mode.** Would fade discs seen edge-on (now a thin, opaque line). The
   viewer's antialias option would have to be on to match.

8. **Expression wrinkles baked into the head texture.** At rest the face shows a frown and forehead
   wrinkles, probably baked from the grimace video's frames, so they appear under every
   expression. Bake the head texture from neutral frames, or weight frames by how neutral they are.

Also open: notebooks 03-05 have not been run since the latest changes (01 and 02 run cleanly), and
nothing is committed since 54b9060.
