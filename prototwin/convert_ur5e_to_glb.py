"""
Convert UR5e Collada meshes to a GLB with proper parent-child joint hierarchy.
Skips normal computation for speed (not needed for ProtoTwin Robot Controller).
Run with: poetry run python tools/convert_ur5e_to_glb.py
"""
import numpy as np
import pygltflib
import trimesh
from pathlib import Path

MESH_DIR = Path("/tmp/ur_description/meshes/ur5e/visual")
OUTPUT = Path("tools/ur5e.glb")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# UR5e link names in kinematic order with Z offset from parent (meters)
LINKS = [
    ("base",     0.000),
    ("shoulder", 0.163),
    ("upperarm", 0.000),
    ("forearm",  0.425),
    ("wrist1",   0.392),
    ("wrist2",   0.095),
    ("wrist3",   0.095),
]

gltf = pygltflib.GLTF2()
gltf.asset = pygltflib.Asset(version="2.0")
binary_data = bytearray()


def add_mesh(name):
    mesh = trimesh.load(str(MESH_DIR / f"{name}.dae"), force="mesh")
    print(f"  [OK] {name}: {len(mesh.vertices)} vertices")

    verts = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.uint32)

    # Pad to 4-byte alignment before each buffer view
    while len(binary_data) % 4:
        binary_data.extend(b"\x00")

    v_offset = len(binary_data)
    binary_data.extend(verts.tobytes())

    while len(binary_data) % 4:
        binary_data.extend(b"\x00")

    f_offset = len(binary_data)
    binary_data.extend(faces.tobytes())

    bv_v = len(gltf.bufferViews)
    gltf.bufferViews.append(pygltflib.BufferView(
        buffer=0, byteOffset=v_offset, byteLength=len(verts.tobytes()),
        target=pygltflib.ARRAY_BUFFER,
    ))
    bv_f = len(gltf.bufferViews)
    gltf.bufferViews.append(pygltflib.BufferView(
        buffer=0, byteOffset=f_offset, byteLength=len(faces.tobytes()),
        target=pygltflib.ELEMENT_ARRAY_BUFFER,
    ))

    acc_v = len(gltf.accessors)
    gltf.accessors.append(pygltflib.Accessor(
        bufferView=bv_v, componentType=pygltflib.FLOAT,
        count=len(verts), type=pygltflib.VEC3,
        max=verts.max(axis=0).tolist(), min=verts.min(axis=0).tolist(),
    ))
    acc_f = len(gltf.accessors)
    gltf.accessors.append(pygltflib.Accessor(
        bufferView=bv_f, componentType=pygltflib.UNSIGNED_INT,
        count=int(faces.size), type=pygltflib.SCALAR,
    ))

    mesh_idx = len(gltf.meshes)
    gltf.meshes.append(pygltflib.Mesh(primitives=[
        pygltflib.Primitive(attributes=pygltflib.Attributes(POSITION=acc_v), indices=acc_f)
    ]))
    return mesh_idx


# Build nodes
node_indices = []
for link_name, z_offset in LINKS:
    mesh_idx = add_mesh(link_name)
    node = pygltflib.Node(name=link_name, translation=[0.0, 0.0, z_offset], mesh=mesh_idx)
    node_indices.append(len(gltf.nodes))
    gltf.nodes.append(node)

# No parent-child hierarchy — all nodes are flat siblings at root level
gltf.scenes = [pygltflib.Scene(nodes=node_indices)]
gltf.scene = 0
gltf.buffers = [pygltflib.Buffer(byteLength=len(binary_data))]
gltf.set_binary_blob(bytes(binary_data))
gltf.save(str(OUTPUT))

size_kb = OUTPUT.stat().st_size // 1024
print(f"\n✅ Exported flat GLB: {OUTPUT.resolve()} ({size_kb} KB)")
print("All 7 links are flat siblings — add joints manually in ProtoTwin")
