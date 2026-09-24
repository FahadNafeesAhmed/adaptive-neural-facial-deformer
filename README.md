# Adaptive Neural Facial Deformer

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)
![OpenUSD](https://img.shields.io/badge/OpenUSD-26.8-1F2937)
![Docker](https://img.shields.io/badge/Docker-CPU%20inference-2496ED?logo=docker&logoColor=white)
![uv](https://img.shields.io/badge/uv-packaging-DE5FE9)
![pytest](https://img.shields.io/badge/tests-34%20passing-0A9EDC?logo=pytest&logoColor=white)

![Ground truth, the raw scan the solver receives, and the solved rig, rendered in Blender](docs/media/demo.gif)

*Left: ground truth. Middle: the raw scan the solver receives, with noise, holes, stray points and a
turned head. Right: the solved rig, 53 blendshape weights and a head pose recovered from those dots,
exported to USD and rendered in Blender.* Full-resolution video: [`docs/media/demo.mp4`](docs/media/demo.mp4).

**Included in this repository:** the trained model ([`weights/solver.pt`](weights/solver.pt)),
the [final benchmark](results/results.md) with its [raw numbers](results/results.json), and the
[full training log](results/training_log.jsonl). No training is needed to try it.

**Point clouds in. Rig sliders out.**

A face scanner hands you thousands of unlabeled, noisy points. An animator needs 53 sliders.
This project solves one into the other: from a raw, partial, arbitrarily posed 3D face scan it
recovers the **53 ARKit-style blendshape weights**, the **head pose** and the performer's
**identity**, then writes the result as a USD rig with editable blend shapes and animation
curves that opens in Blender, Maya or Houdini.

A point-patch transformer trained on endless synthetic captures does the heavy lifting. A short
classical refinement adds the last millimetre. Both are measured head-to-head against a tuned
non-rigid ICP solver on identical scans.

On noisy scans the hybrid solver lands **0.76 mm** from the true surface: **5.7 times more accurate
than tuned ICP and 17 times faster**. The network alone reaches 1.32 mm in 1.9 ms per scan.

## Contents

- [Built with](#built-with)
- [Why this exists](#why-this-exists)
- [What it does](#what-it-does)
- [How it fits a character pipeline](#how-it-fits-a-character-pipeline)
- [Architecture](#architecture)
- [Design decisions worth knowing](#design-decisions-worth-knowing)
- [Quick start](#quick-start)
- [The demo](#the-demo)
- [Command line](#command-line)
- [Data model](#data-model)
- [Rig data contracts](#rig-data-contracts)
- [Testing](#testing)
- [Benchmark](#benchmark)
- [Related work](#related-work)
- [Known limitations](#known-limitations)
- [Roadmap](#roadmap)
- [Repository layout](#repository-layout)
- [Author](#author)

## Built with

| Area | What |
|---|---|
| Face model | [ICT-FaceKit](https://github.com/USC-ICT/ICT-FaceKit) (MIT): 26,719 vertices, 100 PCA identity modes, 53 expression blend shapes with ARKit names |
| Learning | PyTorch 2.11 on CUDA 12.8, bf16 autocast, AdamW with a one-cycle schedule |
| Network | Point-patch transformer, 5.4 M parameters: farthest-point sampling, k-NN patches, mini-PointNet tokens, 6-layer encoder, 6D rotation head |
| Classical solver | Batched non-rigid ICP on the GPU: trimmed correspondences, weighted Procrustes, point-to-plane least squares, FISTA box-constrained QP |
| Pipeline | OpenUSD (`usd-core` 26.8): `UsdSkelBlendShape`, `UsdSkelAnimation`, animated head transform, composed review scene |
| Data contracts | Standard-library rig metadata, pose samples, mesh topology and sequence splits (`anf_deformer`) |
| Packaging | uv + hatchling, two command-line tools, a CPU inference Docker image |
| Tests | pytest and unittest, 34 tests, ruff |
| Hardware | Trained on a single RTX 5060 Laptop GPU (8 GB) |

## Why this exists

A face rig is linear: `face = neutral + Σ identity_i · I_i + Σ weight_j · E_j`, where each
`E_j` is a sculpted shape such as `jawOpen` or `mouthSmile_L` and each `weight_j` lives in
`[0, 1]`. Going from sliders to a face is one matrix multiply. Going from a *scan* to sliders
is the hard direction, and it is the one performance capture needs every frame.

A scan does not arrive as a mesh with known vertices. It arrives as a point cloud:

| Capture problem | What it breaks |
|---|---|
| No vertex correspondence | Nothing says which point is the nose tip |
| Sensor noise (up to ~1 mm) | Plain least squares fits the noise |
| Holes from hair, hands and glare | Whole regions are unconstrained |
| Outliers from the background | Fits are pulled off the surface |
| Unknown head pose | Pose and correspondence depend on each other |
| Self-occlusion | A turned head hides one cheek entirely |

Classical solvers (non-rigid ICP) alternate between guessing correspondences and solving for
parameters. They are precise when they start near the answer and get stuck when they do not.
A learned solver has seen millions of turned, noisy, holed faces and lands near the answer in
one forward pass. Combining the two gives both properties.

## What it does

Three solvers share one interface, `Solver.solve(points, method)`:

| Method | What happens | Speed |
|---|---|---|
| `icp` | Coarse-to-fine non-rigid ICP from five head-yaw starts; the best fit is chosen by its own residual, never by ground truth | ~1.3 s per scan |
| `neural` | One forward pass of the point-patch transformer predicts rotation, translation, identity and 53 weights | ~2 ms per scan (batched) |
| `hybrid` | The network's prediction seeds 10 fine ICP iterations | ~76 ms per scan |

Every method returns the same four things: a 3×3 head rotation, a translation in centimetres,
100 identity coefficients and 53 blendshape weights in `[0, 1]`. A sequence of scans becomes
one USD file: identity averaged across the performance, weights animated per frame, head
motion on the root transform.

## How it fits a character pipeline

| Pipeline need | Where it lives |
|---|---|
| Predict blendshape weights from capture | `src/anfd/solve.py`, `src/anfd/model.py` |
| Train on noisy raw capture, not only clean assets | `src/anfd/synth.py` corruption model |
| Keep the result editable by artists | `src/anfd/usd_io.py`: real blend shapes and curves, not a baked vertex cache |
| Move data between DCC tools | OpenUSD; UsdSkel's own binding query confirms all 53 blend shapes bound under each `SkelRoot` |
| Prove it beats the established method | `scripts/evaluate.py` against a tuned ICP solver on seeded, identical scans |
| Run the same environment anywhere | `Dockerfile` (CPU inference) and a locked `uv.lock` |
| Catch bad training data before training | `anf_deformer` rig metadata, topology and split contracts |

## Architecture

```mermaid
flowchart LR
    ICT[ICT-FaceKit<br/>neutral + 100 identity + 53 expression shapes] --> FM[facemodel.py<br/>batched differentiable rig]
    FM --> SYN[synth.py<br/>synthetic capture generator]
    SYN -- noisy point clouds --> NET[model.py<br/>point-patch transformer]
    SYN -- noisy point clouds --> ICP[fitting.py<br/>coarse-to-fine non-rigid ICP]
    NET -- initial solve --> HYB[hybrid<br/>network + 10 ICP iterations]
    SYN -- ground truth --> MET[metrics.py<br/>errors in mm]
    NET & ICP & HYB --> MET
    HYB --> USD[usd_io.py<br/>UsdSkel rig + head transform]
    USD --> DCC[Blender / Maya / Houdini]
```

Eight layers, each small enough to read in one sitting:

| Layer | Responsibility | Code |
|---|---|---|
| 1. Rig | Load ICT-FaceKit once into an `.npz` cache; evaluate thousands of faces per batch as one matrix multiply, differentiably | `src/anfd/facemodel.py` |
| 2. Capture simulation | Random identity, sparse expression and head pose, then area-weighted surface sampling, occlusion, holes, noise and outliers, all on the GPU | `src/anfd/synth.py` |
| 3. Classical solver | Trimmed correspondences, Procrustes, point-to-plane least squares, box-constrained QP, coarse-to-fine schedule | `src/anfd/fitting.py` |
| 4. Neural solver | Point-patch transformer with a 6D rotation head and sigmoid weights | `src/anfd/model.py` |
| 5. Training | Mesh-space losses on endless fresh scans, fixed seeded evaluation sets | `scripts/train_solver.py` |
| 6. Evaluation | Millimetre metrics, method comparison, breakdown by head yaw | `src/anfd/metrics.py`, `scripts/evaluate.py` |
| 7. Pipeline I/O | Point cloud readers, UsdSkel export, `anfd-solve`, Docker | `src/anfd/solve.py`, `src/anfd/usd_io.py` |
| 8. Data contracts | Validated rig metadata, pose samples, topology and splits for the corrective deformer | `src/anf_deformer/` |

### The network

```
4,096 points ─► farthest-point sample 256 centres ─► 32 nearest neighbours around each
            ─► shared mini-PointNet per patch (max-pooled, order-invariant) ─► 256 tokens
            ─► + position embedding of each centre ─► [CLS] + 6-layer Transformer (dim 256, 8 heads)
            ─► 6D rotation │ translation │ 100 identity │ 53 sigmoid blendshape weights
```

The input is shifted by its median point first, so the network sees shape rather than position
and a few junk points cannot drag the centre. The design follows Point-MAE (Pang et al., ECCV
2022); the rotation head uses the continuous 6D representation of Zhou et al. (CVPR 2019).

## Design decisions worth knowing

**A strong baseline or nothing.** The first ICP version was off by nearly 6 mm even when handed
the true head pose. Three fixes brought that to 0.9 mm: point-to-plane residuals (vertices are 3.4 mm
apart, so point-to-vertex matching has built-in error), bounds enforced inside the solve instead
of clamping afterwards, and a regularisation weight tuned by sweep instead of guessed. The
network is only compared against this corrected version.

**Coarse to fine.** From a wrong starting pose, a weakly regularised fit bends identity and
expression to absorb the pose error and settles near 12 mm. The schedule runs 8 rigid-only
iterations, 10 heavily regularised shape iterations, then 22 fine ones: pose first, coarse shape
second, detail last.

**Bounds inside the solve.** Blend shapes overlap: left and right smile, cheek squint, dimple.
Clamping an unconstrained least-squares answer leaves the compensation of a clamped weight
spread across its neighbours. The expression step is a box-constrained QP solved with FISTA,
step size from the largest eigenvalue of the normal matrix.

**Grade the mesh, not the numbers.** Different weight mixes can produce nearly the same face.
Training loss is measured on posed vertices, pose-free vertices and expression-only vertices,
with weight and rotation terms on top, so a harmless weight difference is not punished and the
face the artist sees is what gets optimised.

**Surface samples, never vertices.** Scan points are drawn at random positions inside triangles,
weighted by area. A network trained on vertex positions could learn vertex identity as a
shortcut that no real scanner provides.

**Endless data, fixed tests.** Every training step generates 64 new scans on the GPU, so nothing
is memorised. Evaluation uses seeded scan sets cached on disk, so every method is scored on
exactly the same inputs, and a test checks that the same seed regenerates identical scans.

**Output a rig, not a mesh.** The export is a `SkelRoot` with a base mesh, 53 sparse
`UsdSkelBlendShape` targets, a `UsdSkelAnimation` of weights and an animated head transform.
A round-trip test rebuilds the face from the USD file alone and matches the Python rig to
within 0.01 mm, head pose included.

**Identity belongs to the performer.** Identity is solved per frame but averaged across a
sequence before export, because one actor's face shape does not change between frames.

## Quick start

Requirements: Python 3.11, [uv](https://docs.astral.sh/uv/), an NVIDIA GPU for training
(inference also runs on CPU).

```bash
uv sync --extra dev
git clone --depth 1 https://github.com/USC-ICT/ICT-FaceKit.git external/ICT-FaceKit
uv run pytest -q
```

The first test run parses the 154 ICT meshes into `data/ict_model.npz` (about 25 s); every run
after that loads it instantly.

### Use the pretrained model

`weights/solver.pt` is the model behind every number on this page: 5.42 M parameters, trained for
40,000 steps (2.56 million generated scans, 122 minutes on an RTX 5060 Laptop GPU). Every tool
picks it up automatically when there is no local training run:

```bash
uv run python scripts/evaluate.py --n 256
uv run python scripts/demo_usd.py --method hybrid
uv run anfd-solve scans/ -o performance.usda
```

Evaluation reproduces [`results/results.md`](results/results.md); the demo writes the USD scene
behind the video at the top of this page.

### Train your own

```bash
uv run python scripts/train_solver.py --steps 40000 --batch 64 --run solver
```

Training runs at 0.18 s per step with a 3.3 GB peak and checkpoints to `runs/solver/solver.pt`
every 2,000 steps; that local run then takes precedence over the pretrained weights. The log of
the run that produced `weights/solver.pt` is in
[`results/training_log.jsonl`](results/training_log.jsonl). Set `ANFD_MODEL` to move the cached
face model away from `data/ict_model.npz`; the Docker image does this.

## The demo

`scripts/demo_usd.py` synthesises a four-second performance: expressions keyframed every half
second with smoothstep blending, a slow head turn of ±25° and a nod of ±8°. Every frame is
scanned with the full corruption model, solved, and written to `runs/solver/demo/`:

| File | Contents |
|---|---|
| `truth.usda` | The ground-truth rig and animation |
| `scan.usda` | The raw point clouds the solver saw, as animated `UsdGeomPoints` |
| `solved.usda` | The solved rig: 53 editable blend shapes, weight curves, head transform |
| `scene.usda` | All three side by side, 30 cm apart, composed by reference |
| `curves.png` | Truth against solved weight curves for the six most active shapes |

Open `scene.usda` in Blender (File, Import, Universal Scene Description) or `usdview` and scrub
the timeline to compare the solved head with the truth, frame by frame. In Blender 5.2 each head
arrives with all 53 blend shapes as animated shape keys, the head turn as a transform cache and
the scan as an animated point cloud. Set the scene to 30 fps; the importer keeps Blender's
default of 24.

The GIF at the top of this page is rendered headlessly from the same file:

```bash
blender -b --factory-startup -P scripts/blender_render.py -- runs/solver/demo/scene.usda runs/solver/demo/frames
uv run --extra media python scripts/make_media.py runs/solver/demo/frames runs/solver/demo/demo
```

The solved weight curves follow the truth across the whole performance:

![Truth against solved weight curves for the six most active blend shapes](docs/media/weight_curves.png)

Per-frame solving shows as small jitter, and `eyeBlink_L` peaks at 0.65 against a true 0.85; both
are discussed under [known limitations](#known-limitations).

## Command line

```bash
uv run anfd-solve scans/ -o performance.usda --method hybrid --units mm --json weights.json
uv run anf-deformer validate-rig path/to/rig.json
```

`anfd-solve` reads `.ply` (ascii and binary), `.npy`, `.xyz`, `.txt` and `.csv`. A folder is a
sequence in filename order. Scans are expected head-centred, Y up, face toward +Z; `--units`
converts from millimetres or metres. `--json` also writes per-frame weights by name.

Docker runs the same solver anywhere, CPU only. The image bakes in `weights/solver.pt` by
default; pass `--build-arg CKPT=runs/solver/solver.pt` to ship your own run instead:

```bash
docker build -t anfd .
docker run --rm -v "$PWD/scans:/in" -v "$PWD/out:/out" anfd /in -o /out/performance.usda --method neural
```

## Data model

What every solver returns, batched:

```python
{
    "rot":   Tensor[B, 3, 3],   # head rotation about the face centre
    "trans": Tensor[B, 3],      # head translation, cm
    "id_w":  Tensor[B, 100],    # identity coefficients, ~N(0, 1)
    "ex_w":  Tensor[B, 53],     # blendshape weights in [0, 1], ICT/ARKit names
}
# posed = (rig(id_w, ex_w) - centre) @ rot.T + centre + trans
```

The capture model is one dataclass, so every corruption is explicit and adjustable:

```python
@dataclass
class CaptureConfig:
    n_points: int = 4096
    noise_cm: tuple[float, float] = (0.0, 0.1)      # per-scan Gaussian sigma, 0 to 1 mm
    max_holes: int = 3
    hole_radius_cm: tuple[float, float] = (0.8, 2.5)
    outlier_frac: tuple[float, float] = (0.0, 0.02)
    yaw_deg: float = 35.0
    pitch_deg: float = 20.0
    roll_deg: float = 10.0
    trans_cm: float = 2.0
    occlusion_prob: float = 0.7                     # single-view vs multi-view capture
```

## Rig data contracts

The corrective deformer on the roadmap will train on control-to-mesh samples exported from a
production rig, so bad data has to be caught before training, not after. `anf_deformer.data`
and `anf_deformer.geometry` define and validate that data using only the standard library:

- **Rig metadata** ([format](docs/data/rig-metadata.md)): ordered control names, value ranges,
  neutral values, fixed vertex count and head-local coordinates, as versioned JSON.
- **Pose samples**: one control vector and the mesh it produces, checked against the metadata.
- **Mesh topology** (`MeshTopology`): validated triangle indices and a vertex count matched to
  the rig schema. Exporters must also keep vertex order identical across poses; matching counts
  alone cannot prove that.
- **Sequence splits** ([contract](docs/data/sequence-splits.md)): whole animation sequences go
  to train, validation or test, so frames from one performance never leak across sets.

See the [validation guide](docs/guides/metadata-validation.md) for exit codes.

## Testing

```bash
uv run pytest -q
```

The 34 tests prove that the torch rig matches the numpy rig; that a USD round trip, head pose
included, stays within 0.01 mm; that seeded scans are identical and weights stay in `[0, 1]`;
that Procrustes recovers a known rigid motion and ICP solves clean scans to under 2 mm from the
true pose; that network rotations are orthonormal with determinant 1; that ASCII and binary PLY
files read correctly; and that the rig data contracts and their CLI reject bad input.

## Benchmark

`scripts/evaluate.py` scores every method on the same 256 seeded synthetic scans per set. Errors
are mean distances in millimetres over the 7,801-vertex face region. The network was trained for
40,000 steps: 2.56 million generated scans in 122 minutes on one RTX 5060 Laptop GPU. The tables
below are copied from [`results/results.md`](results/results.md); the unrounded numbers are in
[`results/results.json`](results/results.json).

| Metric | Meaning |
|---|---|
| Surface | Posed mesh against the true surface, what you see overlaid on the scan |
| Expression | Error from the expression weights alone, identity held at truth: what the animator inherits |
| Weight MAE | Mean absolute blendshape weight error |

**Noisy scans**: up to 1 mm of noise, holes, outliers, occlusion, head turned up to 35°.

| Method | Surface mm | Surface p90 | Expression mm | Weight MAE | Rotation ° | ms per scan |
|---|---|---|---|---|---|---|
| ICP, 5 starts | 4.35 | 7.72 | 3.22 | 0.244 | 3.21 | 1,298 |
| Neural | 1.32 | 1.67 | 0.73 | 0.059 | 0.51 | 1.9 |
| **Hybrid** | **0.76** | **0.94** | **0.54** | 0.063 | **0.50** | 76 |
| ICP given the true pose (upper bound) | 1.08 | 1.54 | 0.77 | 0.107 | 0.76 | 296 |

**Clean scans**: no noise, holes or outliers, full coverage.

| Method | Surface mm | Surface p90 | Expression mm | Weight MAE | Rotation ° | ms per scan |
|---|---|---|---|---|---|---|
| ICP, 5 starts | 4.31 | 7.70 | 3.37 | 0.247 | 3.23 | 1,299 |
| Neural | 1.22 | 1.46 | 0.70 | 0.062 | 0.51 | 1.8 |
| **Hybrid** | **0.65** | **0.77** | **0.48** | **0.055** | **0.50** | 77 |
| ICP given the true pose (upper bound) | 0.92 | 1.27 | 0.69 | 0.091 | 0.69 | 296 |

Surface error on noisy scans by how far the head is turned:

| Method | Yaw 0–10° | Yaw 10–20° | Yaw 20–35° |
|---|---|---|---|
| ICP, 5 starts | 3.80 | 4.14 | 4.85 |
| Neural | 1.24 | 1.33 | 1.36 |
| **Hybrid** | **0.73** | **0.76** | **0.77** |
| ICP given the true pose | 1.00 | 1.04 | 1.17 |

![Surface error during training against the classical solver](docs/media/training_curve.png)

What the numbers say:

- **The hybrid is 5.7 times more accurate than ICP and 17 times faster.** The network alone is
  3.3 times more accurate and about 680 times faster, batched.
- **Head pose is what breaks ICP.** Its error grows from 3.80 to 4.85 mm as the head turns; the
  hybrid stays flat at 0.73 to 0.77 mm because the network supplies the pose.
- **The hybrid beats ICP even when ICP is handed the true pose** (0.76 against 1.08 mm). The
  network supplies a good identity and expression as well as a pose. ICP starting from a
  neutral face settles into a nearby wrong mix of overlapping shapes.
- **Noise costs little.** Clean to noisy moves the hybrid from 0.65 to 0.76 mm.
- **Refinement improves the fit more than the sliders.** On noisy scans, weight error barely
  changes between neural and hybrid (0.059 and 0.063) while surface error falls 42%. Blend
  shapes overlap, so several weight mixes fit almost equally well. A prior that keeps refined
  weights near the network's prediction is the planned fix.

## Related work

The closest work is from EA's own research division: **Rig Inversion by Training a
Differentiable Rig Function** (Marquis Bolduc and Phan, SEED, SIGGRAPH Asia 2022). It inverts a
production face rig from captured meshes by putting the loss on the mesh, through a learned
differentiable rig, instead of on the rig parameters, and names architectures tailored to mesh
data as future work. This project uses the same mesh-space loss, but takes raw, unordered,
partial point clouds with no shared topology, and uses a point-cloud transformer.

- Holden, Saito and Komura, *Learning an Inverse Rig Mapping for Character Animation*
  (SCA 2015) and *Learning Inverse Rig Mappings by Nonlinear Regression* (TVCG 2017): the
  original learned inverse rig.
- Racković et al., *Accurate and Interpretable Solution of the Inverse Rig for Realistic
  Blendshape Models with Quadratic Corrective Terms* (2023): optimisation-based inversion with
  bounded weights, the family the ICP baseline belongs to.
- Liu, Tran and Liu, *3D Face Modeling from Diverse Raw Scan Data* (ICCV 2019), and Bahri et al.,
  *Shape My Face* (IJCV 2021): PointNet-style encoders on raw face scans, fitted to a morphable
  model rather than to artist blend shapes.
- Grishchenko et al., *Blendshapes GHUM* (2023): real-time ARKit-style blend shapes and a 6D
  rotation predicted from image landmarks rather than 3D scans.
- Song, Shi and Reed, *Accurate Face Rig Approximation with Deep Differential Subspace
  Reconstruction* (SIGGRAPH 2020): the forward direction, relevant to the corrective deformer.

## Known limitations

- **Synthetic training only.** Real scanners add their own artifacts (structured-light striping,
  photogrammetry smoothing, hair). The corruption model approximates them; validation on real
  captures is the next step.
- **Linear rig.** ICT-FaceKit is linear blend shapes and real skin is not. The solver can only
  express what the rig can, which is what the corrective deformer addresses.
- **Per-frame solving.** There is no temporal model yet, so frame-to-frame jitter is not
  suppressed; it is visible in the weight curves above.
- **Eyes are the least observable shapes.** At full strength the eye-gaze and eyelid shapes move
  the scanned surface only 2.5 to 3.7 mm, against a median of 8.6 mm across all 53 shapes, and
  the scan barely samples the eye region. In the demo `eyeBlink_L` peaks at 0.65 against a true
  0.85. Production pipelines track gaze from images for this reason.
- **Batch-1 latency.** A single scan takes 39 ms through the network, mostly in the 256-step
  farthest-point sampling loop; batched, it is 1.9 ms per scan.
- **Head-centred input.** Scans must arrive roughly centred with Y up and the face toward +Z.

## Roadmap

1. **Neural corrective deformer:** learn the per-vertex residual the linear rig cannot express
   (the cheek bulge when smiling with the jaw open), conditioned on the solved weights, fed by
   the rig data contracts, and compared against simpler baselines rather than assumed to help.
2. **Temporal solver** for smooth, jitter-free curves across a performance.
3. **Landmark tracking** from images as a camera-space pose prior.
4. **Inference optimisation:** ONNX and TensorRT export, CUDA graphs for the sampling loop.
5. **Maya and Houdini importers** that load the solved USD straight into production rigs.

## Repository layout

```
src/
  anfd/
    facemodel.py     ICT model loading, npz cache, batched differentiable rig
    synth.py         GPU capture generator and seeded evaluation sets
    fitting.py       coarse-to-fine non-rigid ICP, Procrustes, FISTA box QP
    model.py         point-patch transformer solver
    metrics.py       millimetre evaluation metrics
    solve.py         Solver API, point cloud readers, anfd-solve
    usd_io.py        UsdSkel rig export, point cloud export, scene composition
  anf_deformer/
    data/            rig metadata, pose samples, JSON interchange, sequence splits
    geometry/        fixed mesh topology
    cli.py           anf-deformer validate-rig
scripts/
  train_solver.py    training loop
  evaluate.py        ICP, neural and hybrid compared by noise level and head yaw
  demo_usd.py        synthetic performance to a truth, scan and solved USD scene
  blender_render.py  headless Blender render of that scene
  make_media.py      rendered frames to MP4 and GIF
weights/
  solver.pt          pretrained model, 40,000 steps, 5.42 M parameters
results/
  results.md         final benchmark tables
  results.json       unrounded numbers, including the breakdown by head yaw
  training_log.jsonl every logged step and evaluation of the pretrained run
tests/               rig, USD, capture, geometry, ICP, network and IO tests
tests/unit/          data contract tests
docs/                data contract formats and the validation guide
docs/media/          demo GIF and MP4, weight curves, training curve
Dockerfile           CPU inference image (bundles weights/solver.pt)
```

## Author

Fahad Nafees Ahmed

Face model: [ICT-FaceKit](https://github.com/USC-ICT/ICT-FaceKit), USC Institute for Creative
Technologies, MIT License. Point-patch tokenisation after Point-MAE (Pang et al., ECCV 2022).
6D rotation representation after Zhou et al. (CVPR 2019).
