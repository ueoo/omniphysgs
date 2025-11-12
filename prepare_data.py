import os
import subprocess


scene_names = ["cloth", "flower", "newton", "box", "bow", "shirt", "lamp", "laptop"]


data_root_dir = "/svl/u/yuegao/NeuROK/PhysDreamer/phys_dreamer/data_NeuROK_sim"
data_out_dir = "/svl/u/yuegao/NeuROK/omniphysgs/dataset"

items_to_copy = ["point_cloud", "cameras.json"]


def copy_items(src_path, dest_dir):
    subprocess.run(f"cp -r {src_path} {dest_dir}", shell=True)


def one_scene_data(scene_name):

    gs_scene_dir = f"{data_root_dir}/{scene_name}_gs"

    dataset_scene_dir = f"{data_out_dir}/{scene_name}"
    os.makedirs(dataset_scene_dir, exist_ok=True)

    for item in items_to_copy:
        copy_items(f"{gs_scene_dir}/{item}", dataset_scene_dir)


for scene_name in scene_names:
    one_scene_data(scene_name)
