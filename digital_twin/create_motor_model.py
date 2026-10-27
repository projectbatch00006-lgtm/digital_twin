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


# When run from Blender's Text Editor, __file__ may point inside the .blend
# file rather than to a real .py file. Use Blender's saved file location in
# that case so the GLB is exported beside Electric_Motor.blend.
script_path = Path(__file__).resolve() if "__file__" in globals() else None
if script_path and script_path.suffix.lower() == ".py":
    ROOT = script_path.parent
else:
    if not bpy.data.filepath:
        raise RuntimeError("Save the Blender file before running the export script.")
    blend_path = Path(bpy.data.filepath)
    ROOT = blend_path.parent
EXPORT_PATH = ROOT / "Electric_Motor.glb"


def material(name, color, metallic=0.0, roughness=0.5, alpha=1.0):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, alpha)
    shader.inputs["Alpha"].default_value = alpha
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness
    if alpha < 1.0:
        # Blender 4.x uses surface_render_method; older versions use blend_method.
        if hasattr(mat, "surface_render_method"):
            mat.surface_render_method = "DITHERED"
        elif hasattr(mat, "blend_method"):
            mat.blend_method = "BLEND"
    return mat


CASING = material("Neutral Casing", (0.24, 0.27, 0.30), 0.45, 0.32, 0.30)
STEEL = material("Machined Steel", (0.52, 0.55, 0.58), 0.9, 0.2)
DARK_STEEL = material("Dark Steel", (0.06, 0.07, 0.08), 0.65, 0.3)
WINDING = material("Insulated Winding", (0.32, 0.34, 0.36), 0.2, 0.42)


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


def cube(name, scale, location, mat, rotation=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = scale
    obj.rotation_euler = rotation
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return assign(obj, mat)


def torus(name, major_radius, minor_radius, location, mat):
    bpy.ops.mesh.primitive_torus_add(
        major_radius=major_radius, minor_radius=minor_radius,
        major_segments=64, minor_segments=16,
        location=location, rotation=(0, math.radians(90), 0)
    )
    obj = bpy.context.object
    obj.name = name
    return assign(obj, mat)


# Clean every previous object, including accidental floors and cameras.
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

# Main motor axis is X; the model is centered around Z=1.5 and spans roughly
# 6.5 units, which gives browser viewers a predictable, compact bounding box.
cylinder("Transparent stator housing", 1.0, 3.0, (0, 0, 1.5), CASING)

# Internal structure: laminated stator core, insulated windings, rotor core,
# shaft, and drive bearing. The neutral palette keeps the model professional
# while the translucent housing makes the internal parts visible.
cylinder("Stator laminated core", 0.82, 2.72, (0, 0, 1.5), DARK_STEEL)
for index, x in enumerate([i * 0.34 - 1.02 for i in range(7)]):
    torus(f"Stator winding {index + 1:02d}", 0.68, 0.075, (x, 0, 1.5), WINDING)
cylinder("Rotor core", 0.46, 2.35, (0.25, 0, 1.5), STEEL)
for index, x in enumerate([i * 0.28 - 0.72 for i in range(6)]):
    cylinder(f"Rotor lamination {index + 1:02d}", 0.49, 0.035, (x, 0, 1.5), DARK_STEEL, 48)

# Rotor cage bars and end-ring detail, giving the interior a recognizable
# induction-motor construction instead of a plain cylinder.
for index in range(10):
    angle = 2 * math.pi * index / 10
    y = 0.36 * math.cos(angle)
    z = 1.5 + 0.36 * math.sin(angle)
    cube(f"Rotor cage bar {index + 1:02d}", (2.15, 0.045, 0.075), (0.25, y, z), STEEL, (angle, 0, 0))
cylinder("Rotor drive end ring", 0.48, 0.10, (1.45, 0, 1.5), STEEL, 48)
cylinder("Rotor fan end ring", 0.48, 0.10, (-0.95, 0, 1.5), STEEL, 48)

# Stator slot teeth add depth around the inner stator circumference.
for index in range(18):
    angle = 2 * math.pi * index / 18
    y = 0.76 * math.cos(angle)
    z = 1.5 + 0.76 * math.sin(angle)
    cube(f"Stator slot tooth {index + 1:02d}", (2.45, 0.055, 0.11), (0, y, z), DARK_STEEL, (angle, 0, 0))

# Cooling ribs are narrow rings around the housing.
for index, x in enumerate([i * 0.12 - 1.44 for i in range(25)]):
    cylinder(f"Cooling rib {index + 1:02d}", 1.08, 0.045, (x, 0, 1.5), CASING, 48)

cylinder("Drive end shield", 1.04, 0.28, (1.62, 0, 1.5), CASING)
cylinder("Drive bearing", 0.42, 0.24, (1.86, 0, 1.5), STEEL)
cylinder("Drive bearing inner race", 0.27, 0.30, (1.87, 0, 1.5), DARK_STEEL)
cylinder("Fan bearing", 0.38, 0.20, (-1.36, 0, 1.5), STEEL)
cylinder("Fan cowl", 1.04, 0.76, (-1.86, 0, 1.5), DARK_STEEL)
cylinder("Fan cowl rim", 1.08, 0.12, (-2.26, 0, 1.5), DARK_STEEL)
cylinder("Rotor shaft", 0.18, 2.4, (2.55, 0, 1.5), STEEL)
cube("Shaft key", (0.7, 0.12, 0.12), (3.15, 0, 1.67), STEEL)

# Cooling fan blades inside the rear cowl.
for index in range(8):
    angle = 2 * math.pi * index / 8
    y = 0.52 * math.cos(angle)
    z = 1.5 + 0.52 * math.sin(angle)
    cube(f"Cooling fan blade {index + 1:02d}", (0.10, 0.62, 0.12), (-2.18, y, z), STEEL, (angle, 0, 0))

# Terminal box and lid.
cube("Terminal box", (1.25, 1.0, 0.62), (0.25, 0, 2.65), CASING)
cube("Terminal lid", (1.32, 1.07, 0.10), (0.25, 0, 3.02), DARK_STEEL)
for index, y in enumerate((-0.28, 0.28), start=1):
    cylinder(f"Cable gland {index}", 0.14, 0.28, (0.25, y, 2.36), DARK_STEEL, 32)
for index, y in enumerate((-0.30, 0.0, 0.30), start=1):
    cylinder(f"Terminal post {index}", 0.055, 0.18, (0.25, y, 2.82), STEEL, 24)

# Mounting base and feet.
cube("Mounting base", (4.45, 2.7, 0.28), (0, 0, 0.28), CASING)
for index, x in enumerate((-1.45, 1.45), start=1):
    cube(f"Mounting foot {index}", (0.75, 2.9, 0.28), (x, 0, 0.08), CASING)

# Mounting-bolt heads make the base readable in the 3D inspection view.
for index, x in enumerate((-1.45, 1.45), start=1):
    for y in (-1.0, 1.0):
        cylinder(f"Mounting bolt {index}-{y}", 0.10, 0.08, (x, y, 0.28), STEEL, 24)

# A small warning plate helps the digital twin read visually without adding a
# large floor or background object.
cube("Identification plate", (0.65, 0.035, 0.38), (-0.45, -1.36, 1.5), DARK_STEEL)

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
