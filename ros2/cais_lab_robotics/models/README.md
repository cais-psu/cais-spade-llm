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

`table_recovery_framework.world` declares exact mesh collisions separately from
those reusable cylinder models. Its `*_contact.STL` files retain every original
binary triangle record, including normals and attributes, in a deterministic
order: original indices sorted by their bit-reversed index using
`(triangle_count - 1).bit_length()` bits. `contact_meshes.json` records source,
output, and triangle-multiset hashes. Visuals continue using the original files.
This changes contact sampling, not the occupied surface or mass properties.

The recovery world, gear collisions, and the repository-owned `table` surface
allow 64 contacts. Gazebo uses the minimum of the world and both collision
limits. The `table` geometry and original author information are preserved;
`table/source.json` records the upstream source and sole parameter change.
Observed settling is still required by the recovery evidence checks.
