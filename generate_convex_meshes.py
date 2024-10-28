import os
import trimesh
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_convex_mesh(input_dir):
    """
    Generates convex collision meshes for all collision STL files in the specified directory.
    
    Args:
        input_dir (str): Path to the directory containing collision meshes.
    """
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.STL') and not file_name.endswith('.convex.stl'):
            input_path = os.path.join(input_dir, file_name)
            convex_output_path = input_path + '.convex.stl'

            if os.path.exists(convex_output_path):
                logging.info(f"Convex mesh already exists: {convex_output_path}")
                continue

            # Load the mesh
            try:
                mesh = trimesh.load_mesh(input_path)
                if not isinstance(mesh, trimesh.Trimesh):
                    logging.warning(f"Skipping {input_path}: Not a Trimesh object.")
                    continue
            except Exception as e:
                logging.error(f"Failed to load mesh {input_path}: {e}")
                continue

            # Compute convex hull
            try:
                convex_hull = mesh.convex_hull
            except Exception as e:
                logging.error(f"Failed to compute convex hull for {input_path}: {e}")
                continue

            # Export the convex hull
            try:
                convex_hull.export(convex_output_path)
                logging.info(f"Generated convex mesh: {convex_output_path}")
            except Exception as e:
                logging.error(f"Failed to export convex mesh {convex_output_path}: {e}")

if __name__ == "__main__":
    mesh_dir = "/sfs/gpfs/tardis/home/cb5th/.conda/envs/maniskill_env/lib/python3.10/site-packages/mani_skill/assets/robots/fetch/fetch_description/meshes/"
    generate_convex_mesh(mesh_dir)