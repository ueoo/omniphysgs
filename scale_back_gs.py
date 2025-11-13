import os
from tqdm import tqdm, trange
import sys
import json
import shutil
from p_tqdm import p_umap

from gaussian3d.scene import GaussianModel

# scene_names = ["lamp"]

scene_names = ["box", "bow", "cloth", "flower", "newton", "lamp", "shirt", "laptop"]

jobs = []

for scene_name in scene_names:
    scene_path = f"/datal/outputs/{scene_name}"
    copy_path = f"/datal/outputs_ominiphysgs/{scene_name}_ply"
    save_path = f"/datal/outputs_ominiphysgs/{scene_name}_ply_scaled"

    os.makedirs(copy_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    json_path = f"/datal/data_NeuROK_sim/{scene_name}_images/transforms.json"
    with open(json_path, "r") as f:
        data = json.load(f)
    scale = data["scale"]

    num_frames = 150

    for i in range(num_frames):
        gaussian_path = os.path.join(scene_path, "gaussians_rendered", f"frame_{i:04d}.ply")
        copy_gaussian_path = os.path.join(copy_path, f"gaussian_{i:03d}.ply")
        shutil.copy(gaussian_path, copy_gaussian_path)

        out_gaussian_path = os.path.join(save_path, f"gaussian_{i:03d}.ply")
        jobs.append((gaussian_path, scale, out_gaussian_path))


def one_job(job):
    gaussian_path, scale, out_gaussian_path = job
    gaussian = GaussianModel(3)
    gaussian.load_ply(gaussian_path)
    gaussian._xyz = gaussian._xyz / scale
    gaussian.save_ply(out_gaussian_path)


p_umap(one_job, jobs, num_cpus=8, desc=f"Scaling {scene_name} Gaussians")
