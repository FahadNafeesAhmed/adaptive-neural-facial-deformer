# Adaptive Neural Facial Deformer

Planned research and engineering project exploring efficient facial deformation, contact quality, local control, and adaptive selection of training poses.

**Status: repository structure only.** No model, training pipeline, integration, dataset, or benchmark has been implemented. All components below describe intended responsibilities, not completed features or demonstrated results.

## Planned workflow

Reference rig -> pose generation -> validated data -> baseline training -> deformation evaluation -> adaptive sampling -> Maya integration.

The initial scope is one character with fixed mesh topology. The proposed model will learn corrective deformation relative to a suitable basic rig. Contact constraints and adaptive sampling will be evaluated against simpler baselines; improvements are not assumed.

## Repository layout

```text
adaptive-neural-facial-deformer/
|-- README.md
|-- .gitignore
|-- docs/
|   |-- architecture/        Design decisions and component interfaces
|   |-- research/            Literature, sources, and comparison methods
|   |-- data/                Dataset contracts and provenance guidance
|   `-- guides/              Setup, learning, and usage documentation
|-- assets/
|   |-- rigs/                Reference character rigs (local only)
|   `-- meshes/              Reference meshes (local only)
|-- data/
|   |-- raw/                 Original exports (local only)
|   |-- processed/           Validated training arrays (local only)
|   |-- splits/              Training, validation, and test assignments
|   `-- metadata/            Control definitions and dataset provenance
|-- configs/
|   |-- data/                Export and preprocessing settings
|   |-- models/              Model settings
|   |-- training/            Training settings
|   |-- evaluation/          Metrics and benchmark settings
|   `-- sampling/            Pose selection settings
|-- src/
|   `-- anf_deformer/
|       |-- data/            Data loading and validation
|       |-- geometry/        Mesh operations, regions, and contact geometry
|       |-- models/          Baselines and neural deformation models
|       |-- training/        Losses and optimization
|       |-- evaluation/      Accuracy, contact, locality, and timing metrics
|       |-- sampling/        Initial and adaptive pose selection
|       `-- runtime/         Model packaging and inference
|-- integrations/
|   |-- maya/                Rig export and live deformer integration
|   `-- usd/                 Supported scene and animation interchange
|-- scripts/                 Future command-line entry points
|-- notebooks/               Exploratory analysis
|-- tests/
|   |-- unit/                Component-level checks
|   |-- integration/         Pipeline and integration checks
|   `-- fixtures/            Small, redistributable test assets
|-- experiments/             Experiment definitions and analysis notes
|-- reports/
|   |-- benchmarks/          Measured comparisons
|   `-- figures/             Selected evaluation visuals
|-- deployment/
|   |-- docker/              Future reproducible training environment
|   `-- cloud/               Future cloud execution setup
`-- outputs/                 Generated runs, models, and logs (local only)
```

Empty directories are tracked with `.gitkeep` placeholders. No executable files, dependencies, CI workflows, or deployment configuration are included in this initial structure.

## Data and results

- Rig and mesh assets require appropriate usage rights. No third-party assets are included.
- Raw data, processed arrays, trained weights, and generated outputs are excluded from Git by default.
- Keep held-out test data separate from training and adaptive pose selection.
- Reports should identify the reference rig, data split, hardware, baselines, and limitations. No performance results are currently available.

## First milestone

Select a usable reference rig, validate control-to-mesh exports, train a compact baseline, and compare its output with the original rig on unseen poses.
