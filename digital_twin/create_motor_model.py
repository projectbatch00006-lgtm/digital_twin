"""Generate the dashboard-ready industrial motor model in Blender.

Run inside Blender with:
    blender --background --python digital_twin/create_motor_model.py

The script exports only motor meshes to Electric_Motor.glb. It deliberately
does not create a floor, camera, or light, because those can make the motor
appear tiny in a browser viewer.
"""

import math
from pathlib import Path

import bpy


ROOT = Path(__file__).resolve().parent
EXPORT_PATH = ROOT / "Electric_Motor.glb"


def material(name, color, metallic=0.0, roughness=0.5):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1.0)
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness
    return mat


BLUE = material("Motor Blue", (0.02, 0.16, 0.42), 0.35, 0.32)
STEEL = material("Machined Steel", (0.55, 0.60, 0.66), 0.9, 0.2)
BLACK = material("Matte Black", (0.015, 0.02, 0.025), 0.15, 0.45)
YELLOW = material("Safety Yellow", (0.8, 0.38, 0.015), 0.2, 0.35)


def assign(obj, mat):
    obj.data.materials.append(mat)
    bevel = obj.modifiers.new("Manufacturing edge bevel", "BEVEL")
    bevel.width = 0.035
    bevel.segments = 2
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=bevel.name)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def cylinder(name, radius, depth, location, mat, vertices=64):
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices, radius=radius, depth=depth,
        location=location, rotation=(0, math.radians(90), 0)
    )
    obj = bpy.context.object
    obj.name = name
    return assign(obj, mat)


def cube(name, scale, location, mat):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return assign(obj, mat)


# Clean every previous object, including accidental floors and cameras.
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

# Main motor axis is X; the model is centered around Z=1.5 and spans roughly
# 6.5 units, which gives browser viewers a predictable, compact bounding box.
cylinder("Stator housing", 1.0, 3.0, (0, 0, 1.5), BLUE)

# Cooling ribs are narrow rings around the housing.
for index, x in enumerate([i * 0.12 - 1.44 for i in range(25)]):
    cylinder(f"Cooling rib {index + 1:02d}", 1.08, 0.045, (x, 0, 1.5), BLUE, 48)

cylinder("Drive end shield", 1.04, 0.28, (1.62, 0, 1.5), BLUE)
cylinder("Drive bearing", 0.42, 0.24, (1.86, 0, 1.5), STEEL)
cylinder("Fan cowl", 1.04, 0.76, (-1.86, 0, 1.5), BLACK)
cylinder("Fan cowl rim", 1.08, 0.12, (-2.26, 0, 1.5), BLACK)
cylinder("Rotor shaft", 0.18, 2.4, (2.55, 0, 1.5), STEEL)
cube("Shaft key", (0.7, 0.12, 0.12), (3.15, 0, 1.67), STEEL)

# Terminal box and lid.
cube("Terminal box", (1.25, 1.0, 0.62), (0.25, 0, 2.65), BLUE)
cube("Terminal lid", (1.32, 1.07, 0.10), (0.25, 0, 3.02), BLACK)
for index, y in enumerate((-0.28, 0.28), start=1):
    cylinder(f"Cable gland {index}", 0.14, 0.28, (0.25, y, 2.36), BLACK, 32)

# Mounting base and feet.
cube("Mounting base", (4.45, 2.7, 0.28), (0, 0, 0.28), BLUE)
for index, x in enumerate((-1.45, 1.45), start=1):
    cube(f"Mounting foot {index}", (0.75, 2.9, 0.28), (x, 0, 0.08), BLUE)

# A small warning plate helps the digital twin read visually without adding a
# large floor or background object.
cube("Safety plate", (0.65, 0.035, 0.38), (-0.45, -1.36, 1.5), YELLOW)

# Ensure every exported object is a mesh, has applied transforms, and is part
# of one compact selection. No camera, plane, or light is exported.
bpy.ops.object.select_all(action="DESELECT")
mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
for obj in mesh_objects:
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

# Set a clean world background for Blender previews only; it is not exported
# as geometry and therefore cannot affect browser framing.
bpy.context.scene.world.color = (0.015, 0.02, 0.035)

bpy.ops.export_scene.gltf(
    filepath=str(EXPORT_PATH),
    export_format="GLB",
    use_selection=True,
    export_apply=True,
    export_cameras=False,
    export_lights=False,
)

print(f"Exported dashboard motor model: {EXPORT_PATH}")
