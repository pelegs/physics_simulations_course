import bpy

# Define the path to your TXT file
txt_file_path = "/home/pelegs/Documents/DHBW/Mannheim/physics_simulations_course/code/thermodynamics/tests/coordinates1.txt"


# Load the TXT data
def load_coordinates(file_path):
    coordinates = {}
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines[1:]:
            parts = line.split()
            frame = int(parts[0])
            sphere_count = int(parts[1])
            radii = [float(parts[2 + i]) for i in range(sphere_count)]
            positions = parts[2 + sphere_count :]
            coordinates[frame] = {
                "radii": radii,
                "positions": [
                    (
                        float(positions[i]),
                        float(positions[i + 1]),
                        float(positions[i + 2]),
                    )
                    for i in range(0, len(positions), 3)
                ],
            }
    return coordinates


# Apply coordinates and radii to spheres
def apply_coordinates_to_spheres(spheres, coordinates):
    for frame, data in coordinates.items():
        bpy.context.scene.frame_set(frame)
        radii = data["radii"]
        positions = data["positions"]
        for i, (x, y, z) in enumerate(positions):
            sphere_name = f"Sphere_{i+1}"
            if sphere_name in spheres:
                sphere = spheres[sphere_name]
                sphere.location = (x, y, z)
                sphere.scale = (
                    radii[i],
                    radii[i],
                    radii[i],
                )  # Set the radius (sphere's scale)
                sphere.keyframe_insert(data_path="location")
                sphere.keyframe_insert(data_path="scale")


# Clear existing keyframes
def clear_keyframes(obj):
    obj.animation_data_clear()


# Create or get spheres
def create_or_get_spheres(count):
    spheres = {}
    for i in range(count):
        sphere_name = f"Sphere_{i+1}"
        if sphere_name in bpy.data.objects:
            spheres[sphere_name] = bpy.data.objects[sphere_name]
        else:
            bpy.ops.mesh.primitive_uv_sphere_add()
            sphere = bpy.context.object
            sphere.name = sphere_name
            spheres[sphere_name] = sphere
    return spheres


# Load coordinates and determine the number of spheres
coords = load_coordinates(txt_file_path)
if coords:
    # Determine the number of spheres from the first frame
    sphere_count = len(coords[next(iter(coords))]["positions"])
    spheres = create_or_get_spheres(sphere_count)

    # Clear existing keyframes
    for sphere in spheres.values():
        clear_keyframes(sphere)

    # Apply coordinates and radii to spheres
    apply_coordinates_to_spheres(spheres, coords)

    # Set the scene end frame to the last frame from the TXT file
    bpy.context.scene.frame_end = max(coords.keys())
else:
    print("No coordinates found in the file.")
