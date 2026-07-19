# Gazebo models

Place reusable Gazebo model directories here. Each model directory contains its
own `model.config`, `model.sdf`, and `meshes/` directory. Raw CAD exports remain
in `../cad_models/`; only Gazebo-ready model assets belong here.

The digital-twin gear model names remain:

- `gear_small`
- `gear_medium`
- `gear_large`

The three gear directories contain the Gazebo-ready copies of the uploaded STL
files. Each SDF uses metre scale `0.001`, offsets the original CAD coordinates
to the link center, and uses a 20 mm simple-cylinder collision. Keep future raw
CAD revisions in `../cad_models/` and then deliberately refresh the matching
mesh copy here.

The gear STL visuals use a 180-degree local-X correction because the uploaded
CAD top face points along local `-Z`. Their translations are recomputed after
rotation so the meshes remain centered on the unchanged cylinder collisions.
