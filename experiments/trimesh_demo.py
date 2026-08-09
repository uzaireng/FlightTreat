import trimesh

mesh = trimesh.creation.box(
    extents=[2.0, 3.0, 4.0]
)

print("Trimesh demonstration")
print(f"Vertices: {len(mesh.vertices)}")
print(f"Faces: {len(mesh.faces)}")
print(f"Surface: {mesh.area:.4f}")
print(f"Volume: {mesh.volume:.4f}")
print(f"Watertight: {mesh.is_watertight}")


"""not watertight example"""
# open_box = mesh.copy()
# open_box.update_faces(open_box.faces[:-1])
# print(f"Open Box Waterweight: {open_box.is_watertight}")


# Flat plane (like piece of 2d paper)
