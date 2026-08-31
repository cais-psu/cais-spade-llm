# Product references

This directory records paths and provenance for the controlled local corpus.
NIST STL source files remain in their existing repository locations and are not
copied.

## Existing sources

- NIST STL catalog:
  `ros2/cais_lab_robotics/cad_models/`
- NIST assembly source page:
  <https://www.nist.gov/el/intelligent-systems-division-73500/robotic-grasping-and-manipulation-assembly/assembly>
- Local NIST instructions:
  `cais_spade_llm/spec2primitives/references/products/NIST_assembly_instructions.pdf`

## Phase 1 approved exact refs

`approved_sources.json` is the complete retrieval allowlist. It currently
records `NIST_assembly_instructions.pdf` and the exact filenames of all 34 STL
files under `ros2/cais_lab_robotics/cad_models/`, together with the expected
SHA-256 of every source. The resolver and document pipeline accept every PDF
registered with the same strict manifest contract.

The local resolver serves each registered document's ordered pages, extracted
text, and source SHA-256, or a bounded CAD geometry summary. It does not copy
source files, return raw STL bytes, add sources automatically, or expose scene roles, configured
poses, Gazebo state, or evaluator information. A source whose bytes differ from
the authority-pinned inventory digest is unavailable.

Manual lifecycle: place the PDF in this controlled directory, add its exact ref,
repository path, source URL, page count, and SHA-256 to `approved_sources.json`, prepare
the generic cache, start the system, run generalized PA grounding, and create
the late semantic projection only when PA understanding is sufficient. A
changed PDF hash or changed model/schema configuration invalidates the prepared
cache without a code change or document-purpose prompt.

## NIST instructions provenance

- Official source page: [NIST Robotic Grasping and Manipulation Assembly](https://www.nist.gov/el/intelligent-systems-division-73500/robotic-grasping-and-manipulation-assembly/assembly)
- Retrieved: `2026-08-24`
- Conversion note: the official NIST assembly instructions were saved locally as
  `NIST_assembly_instructions.pdf` for the approved document corpus; this local
  file is a conversion/archive copy, not a modified specification.
- SHA-256:
  `a0aa044e88aee3f1d1011eb8e681a626c4b459706bded36b684a21f0fd189e03`

## Scene CAD manifest

All dimensions below are the measured STL bounds in millimetres. They are used
only to configure this scene's meshes and simple collisions; no physics-accuracy
claim is made.

| CAD filename | Printer/role | STL bounds (mm) | SHA-256 |
|---|---|---:|---|
| `GMC_Laser_Plate_Virtual.STL` | centered assembly plate | `384 x 384 x 8.9916` | `5ecfce4b78fccdbd5430a8a8fc3d4d7673ab9bede91f6134ee83cd2b77af4a0e` |
| `Gear_Plate.STL` | installed static gear fixture | `60 x 120 x 5` | `5a087e7e8a0803d4a74a1bd273a346d5551747113ac8e66ce8ed4d12e7472555` |
| `Gear_Shaft.STL` | three installed static gear shafts | `approximately 10 x 10 x 20` | `0f6c7b27502a308f49dfdbaf1c145bd7446ff1291ba44af9e8ba55729fe3f9ee` |
| `KET4_Square_4mm.STL` | `prusa_mk3` | `4 x 4 x 50` | `c36d7df967ab5db74f23928b4683605d05a8eca936f5204f2cab207156da5002` |
| `KET8_Square_8mm.STL` | `prusa_mk3` | `8 x 7 x 50` | `55037037a4514deac9c974ad52c50b741c089719afe6edddd49b2653ab56c039` |
| `KET12_Square_12mm.STL` | `prusa_mk3` | `12 x 8 x 50` | `bbd3b89e9986f74c08af5bcc8917318f5ea997d265e9400ea95f22285307b653` |
| `KET16_Square_16mm.STL` | `prusa_mk3` | `16 x 10 x 50` | `6042733fcbe002990d4237d56fb0f9c62ec7f3c45040ad5f49da77450ec3ca15` |
| `RGOCG4-50_Round_4mm.STL` | `prusa_mk4_1` | `4 x 4 x 50` | `f4e19521792d8ebfc9df260abebc633831bad129d33f274fb879de19676c4f89` |
| `RGOCG8-50_8mm.STL` | `prusa_mk4_1` | `8 x 8 x 50` | `35b2e14b3aedf1fd8b6bc98a3ec5ec26dc2ea1894c8cce41f7a9da08719e30b6` |
| `RGOCG12-50_12mm.STL` | `prusa_mk4_1` | `12 x 12 x 50` | `67d0dfc1163241f644975f2bc966d6b6a8143a33b25e17c6a1415d028b554829` |
| `RGOCG16-50_16mm.STL` | `prusa_mk4_1` | `16 x 16 x 50` | `f254ed1c747195f38d322f5872beb30ed82fa332c5b416b59e0bcc0a9ca14e3d` |

The existing `gear_small`, `gear_medium`, and `gear_large` Gazebo models on
`prusa_mk4_2` use `Gear_Small.STL`, `Gear_Medium.STL`, and `Gear_Large.STL` from
the same NIST CAD corpus.

The NIST instructions install the gear fixture by inserting three M6 bolts from
the underside of the task board, fastening `Gear_Plate` at the corresponding
three locations, and tightening three `Gear_Shaft` parts onto the remaining
exposed M6 threads. The dedicated scene represents that completed installation
as a static fixture and does not render the hidden fasteners.

This scene-only milestone does not derive tolerances or claim that a source has
been grounded. Later case work must record which exact source supports each
grounded value and must obey the MUST do-not-leak boundary in `../../AGENTS.md`.
