import argparse
import copy
import json
import math
import os
import sys

import cv2
import imageio
import numpy as np
import torch
import torchvision

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from tqdm import tqdm

from gaussian3d.gaussian_renderer import GaussianModel, render
from gaussian3d.scene.cameras import Camera as GSCamera
from gaussian3d.scene.gaussian_model import GaussianModel
from gaussian3d.utils.graphics_utils import focal2fov
from gaussian3d.utils.loss_utils import l1_loss, l2_loss, ssim
from gaussian3d.utils.sh_utils import eval_sh
from gaussian3d.utils.system_utils import searchForMaxIteration
from gaussian3d.utils.general_utils import inverse_sigmoid

from .camera_view_utils import get_camera_view
from .filling_utils import *
from .transformation_utils import *


class PipelineParamsNoparse:
    """Same as PipelineParams but without argument parser."""

    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


def load_gaussian_ckpt(model_path, sh_degree=3, iteration=-1):
    # Find checkpoint
    checkpt_dir = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        iteration = searchForMaxIteration(checkpt_dir)
    checkpt_path = os.path.join(checkpt_dir, f"iteration_{iteration}", "point_cloud.ply")

    # Load guassians
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(checkpt_path)
    return gaussians


def initialize_resterize(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
):
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterize = GaussianRasterizer(raster_settings=raster_settings)
    return rasterize


def load_params_from_gs(pc: GaussianModel, pipe, scaling_modifier=1.0, override_color=None):
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        shs = pc.get_features
    else:
        colors_precomp = override_color

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.

    return {
        "pos": means3D,
        "screen_points": means2D,
        "shs": shs,
        "colors_precomp": colors_precomp,
        "opacity": opacity,
        "scales": scales,
        "rotations": rotations,
        "cov3D_precomp": cov3D_precomp,
    }


def load_params(gaussians, pipeline, preprocessing_params, material_params, model_params, export_path="./"):

    params = load_params_from_gs(gaussians, pipeline)

    init_pos = params["pos"]
    init_cov = params["cov3D_precomp"]
    init_screen_points = params["screen_points"]
    init_opacity = params["opacity"]
    init_shs = params["shs"]
    print(f"Init pos shape: {init_pos.shape}")
    print(f"Init pos x min: {init_pos[:, 0].min()}, max: {init_pos[:, 0].max()}")
    print(f"Init pos y min: {init_pos[:, 1].min()}, max: {init_pos[:, 1].max()}")
    print(f"Init pos z min: {init_pos[:, 2].min()}, max: {init_pos[:, 2].max()}")

    init_e_cat, init_p_cat = None, None

    # throw away low opacity kernels
    mask = init_opacity[:, 0] > preprocessing_params.opacity_threshold
    init_pos = init_pos[mask, :]
    init_cov = init_cov[mask, :]
    init_opacity = init_opacity[mask, :]
    init_screen_points = init_screen_points[mask, :]
    init_shs = init_shs[mask, :]

    # throw away large kernels
    # this is useful when the Gaussian asset is of low quality
    # init_cov = init_cov * 0.5
    # init_cov_mat = get_mat_from_upper(init_cov)
    # mask = filter_cov(init_cov_mat, threshold=2e-4)
    # init_cov[~mask] = 0.5 * init_cov[~mask]

    # rotate and translate
    rotation_matrices = generate_rotation_matrices(
        torch.tensor(preprocessing_params.rotation_degree),
        preprocessing_params.rotation_axis,
    )
    rotated_pos = apply_rotations(init_pos, rotation_matrices)

    # select a sim area and save params of unslected particles
    unselected_pos, unselected_cov, unselected_opacity, unselected_shs = (
        None,
        None,
        None,
        None,
    )
    if preprocessing_params.sim_area is not None:
        boundary = preprocessing_params.sim_area
        assert len(boundary) == 6
        mask = torch.ones(rotated_pos.shape[0], dtype=torch.bool).to(device="cuda")
        for i in range(3):
            mask = torch.logical_and(mask, rotated_pos[:, i] > boundary[2 * i])
            mask = torch.logical_and(mask, rotated_pos[:, i] < boundary[2 * i + 1])

        unselected_pos = init_pos[~mask, :]
        unselected_cov = init_cov[~mask, :]
        unselected_opacity = init_opacity[~mask, :]
        unselected_shs = init_shs[~mask, :]

        rotated_pos = rotated_pos[mask, :]
        init_cov = init_cov[mask, :]
        init_opacity = init_opacity[mask, :]
        init_shs = init_shs[mask, :]

    factor = preprocessing_params.get("scale_factor", 0.95)
    transformed_pos, scale_origin, original_mean_pos = transform2origin(rotated_pos, factor)
    transformed_pos = shift2center05(transformed_pos)
    # modify covariance matrix accordingly
    init_cov = apply_cov_rotations(init_cov, rotation_matrices)
    init_cov = scale_origin * scale_origin * init_cov

    temp_gs_num = transformed_pos.shape[0]
    filling_params = preprocessing_params.particle_filling
    if filling_params is not None:
        print("Filling internal particles...")
        pos = fill_particles(
            pos=transformed_pos,
            opacity=init_opacity,
            cov=init_cov,
            grid_n=filling_params["n_grid"],
            max_samples=filling_params["max_particles_num"],
            grid_dx=1.0 / filling_params["n_grid"],
            density_thres=filling_params["density_threshold"],
            search_thres=filling_params["search_threshold"],
            max_particles_per_cell=filling_params["max_particles_per_cell"],
            search_exclude_dir=filling_params["search_exclude_direction"],
            ray_cast_dir=filling_params["ray_cast_direction"],
            boundary=filling_params["boundary"],
            smooth=filling_params["smooth"],
        ).cuda()
        print(f'Exporting filled particles to {os.path.join(export_path, "filled_particles.ply")}')
        particle_position_tensor_to_ply(pos, os.path.join(export_path, "filled_particles.ply"))

    if filling_params is not None and filling_params["visualize"] == True:
        shs, opacity, cov = init_filled_particles(
            pos[:temp_gs_num],
            init_shs,
            init_cov,
            init_opacity,
            pos[temp_gs_num:],
        )
    else:
        if filling_params is None:
            pos = transformed_pos
        cov = torch.zeros((pos.shape[0], 6), device="cuda")
        cov[:temp_gs_num] = init_cov
        shs = init_shs
        opacity = init_opacity

    if model_params.normalize_features:
        # get normalized features
        n_particles = transformed_pos.shape[0]
        normalized_cov = flatten_and_normalize(init_cov, n_particles)
        normalized_shs = flatten_and_normalize(init_shs, n_particles)
        normalized_opacity = flatten_and_normalize(init_opacity, n_particles)
        features = torch.cat((pos, normalized_shs, normalized_cov, normalized_opacity), dim=1)  # (n, feat_dim)
    else:
        n_particles = transformed_pos.shape[0]
        flattened_cov = init_cov.reshape(n_particles, -1)
        flattened_shs = init_shs.reshape(n_particles, -1)
        flattened_opacity = init_opacity.reshape(n_particles, -1)
        features = torch.cat([pos, flattened_shs, flattened_cov, flattened_opacity], dim=1)  # (n, feat_dim)

    mpm_params = {
        "pos": pos,
        "cov": cov,
        "opacity": opacity,
        "shs": shs,
        "features": features,
    }
    unselected_params = {
        "pos": unselected_pos,
        "cov": unselected_cov,
        "opacity": unselected_opacity,
        "shs": unselected_shs,
    }
    translate_params = {
        "rotation_matrices": rotation_matrices,
        "scale_origin": scale_origin,
        "original_mean_pos": original_mean_pos,
    }

    print(f"Trans (preprocessed GS) pos shape: {pos.shape}")
    print(f"Trans (preprocessed GS) pos x min: {pos[:, 0].min()}, max: {pos[:, 0].max()}")
    print(f"Trans (preprocessed GS) pos y min: {pos[:, 1].min()}, max: {pos[:, 1].max()}")
    print(f"Trans (preprocessed GS) pos z min: {pos[:, 2].min()}, max: {pos[:, 2].max()}")

    return mpm_params, init_e_cat, init_p_cat, unselected_params, translate_params, init_screen_points


def convert_SH(
    shs_view,
    viewpoint_camera,
    pc: GaussianModel,
    position: torch.tensor,
    rotation: torch.tensor = None,
):
    shs_view = shs_view.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
    dir_pp = position - viewpoint_camera.camera_center.repeat(shs_view.shape[0], 1)
    if rotation is not None:
        n = rotation.shape[0]
        dir_pp[:n] = torch.matmul(rotation, dir_pp[:n].clone().unsqueeze(2)).squeeze(
            2
        )  # replace inplace operation for backward

    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

    return colors_precomp


def export_rendering(rendering, step, folder, height=None, width=None):

    cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    if height is None or width is None:
        height = cv2_img.shape[0] // 2 * 2
        width = cv2_img.shape[1] // 2 * 2
    cv2.imwrite(
        os.path.join(folder, f"{step}.png".rjust(8, "0")),
        255 * cv2_img,
    )


def save_video(folder, output_filename, start=0, end=9999, fps=30):

    filenames = os.listdir(folder)
    filenames = [f for f in filenames if int(f.split(".")[0]) >= start and int(f.split(".")[0]) < end]
    filenames = sorted(filenames)

    image = []
    for filename in filenames:
        if filename.endswith(".png"):
            image.append(imageio.v2.imread(os.path.join(folder, filename)))
    imageio.mimsave(output_filename, image, fps=fps)


def interpolate_rgb(rgb1, rgb2, t):
    return rgb1 * (1 - t) + rgb2 * t


def render_mpm_gaussian(
    model_path,
    pipeline,
    render_params,
    step,
    viewpoint_center_worldspace,
    observant_coordinates,
    gaussians,
    background,
    pos,
    cov,
    shs,
    opacity,
    rot,
    screen_points,
    logits=None,
):
    current_camera = get_camera_view(
        model_path,
        default_camera_index=render_params.default_camera_index,
        center_view_world_space=viewpoint_center_worldspace
        - 0.3,  # TODO: 0.3 is a magic number during early development, should be removed
        observant_coordinates=observant_coordinates,
        show_hint=render_params.show_hint,
        init_azimuthm=render_params.init_azimuthm,
        init_elevation=render_params.init_elevation,
        init_radius=render_params.init_radius,
        move_camera=render_params.move_camera,
        current_frame=step,
        delta_a=render_params.delta_a,
        delta_e=render_params.delta_e,
        delta_r=render_params.delta_r,
    )
    rasterize = initialize_resterize(current_camera, gaussians, pipeline, background)
    if logits is None:
        colors_precomp = convert_SH(shs, current_camera, gaussians, pos, rot)
    else:
        vis = torch.softmax(logits, dim=1)
        vis = vis[:, 1] - vis[:, 0]
        vis = (vis - vis.min()) / (vis.max() - vis.min())
        rgb1 = torch.tensor([255, 0, 0], device="cuda").float() / 255
        rgb2 = torch.tensor([0, 0, 255], device="cuda").float() / 255
        colors_precomp = interpolate_rgb(rgb1, rgb2, vis.unsqueeze(1))

    rendering, raddi = rasterize(
        means3D=pos,
        means2D=screen_points,
        shs=None,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=None,
        rotations=None,
        cov3D_precomp=cov,
    )
    return rendering


def particle_position_tensor_to_ply(position_tensor, filename):
    # position is (n,3)
    if os.path.exists(filename):
        os.remove(filename)
    position = position_tensor.clone().detach().cpu().numpy()
    num_particles = (position).shape[0]
    position = position.astype(np.float32)
    with open(filename, "wb") as f:  # write binary
        header = f"""ply
format binary_little_endian 1.0
element vertex {num_particles}
property float x
property float y
property float z
end_header
"""
        f.write(str.encode(header))
        f.write(position.tobytes())


def _rotation_matrix_to_quaternion_batch(R: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of rotation matrices to unit quaternions (w, x, y, z).
    R: (N, 3, 3) rotation matrices
    return: (N, 4) normalized quaternions with w >= 0 for sign consistency
    """
    assert R.dim() == 3 and R.shape[1:] == (3, 3)
    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]
    trace = m00 + m11 + m22

    q = torch.zeros((R.shape[0], 4), device=R.device, dtype=R.dtype)

    cond0 = trace > 0.0
    s0 = torch.sqrt(torch.clamp(trace[cond0] + 1.0, min=1e-12)) * 2.0
    q[cond0, 0] = 0.25 * s0
    q[cond0, 1] = (m21[cond0] - m12[cond0]) / s0
    q[cond0, 2] = (m02[cond0] - m20[cond0]) / s0
    q[cond0, 3] = (m10[cond0] - m01[cond0]) / s0

    cond1 = ~cond0 & (m00 >= m11) & (m00 >= m22)
    s1 = torch.sqrt(torch.clamp(1.0 + m00[cond1] - m11[cond1] - m22[cond1], min=1e-12)) * 2.0
    q[cond1, 0] = (m21[cond1] - m12[cond1]) / s1
    q[cond1, 1] = 0.25 * s1
    q[cond1, 2] = (m01[cond1] + m10[cond1]) / s1
    q[cond1, 3] = (m02[cond1] + m20[cond1]) / s1

    cond2 = ~cond0 & ~cond1 & (m11 >= m22)
    s2 = torch.sqrt(torch.clamp(1.0 + m11[cond2] - m00[cond2] - m22[cond2], min=1e-12)) * 2.0
    q[cond2, 0] = (m02[cond2] - m20[cond2]) / s2
    q[cond2, 1] = (m01[cond2] + m10[cond2]) / s2
    q[cond2, 2] = 0.25 * s2
    q[cond2, 3] = (m12[cond2] + m21[cond2]) / s2

    cond3 = ~cond0 & ~cond1 & ~cond2
    s3 = torch.sqrt(torch.clamp(1.0 + m22[cond3] - m00[cond3] - m11[cond3], min=1e-12)) * 2.0
    q[cond3, 0] = (m10[cond3] - m01[cond3]) / s3
    q[cond3, 1] = (m02[cond3] + m20[cond3]) / s3
    q[cond3, 2] = (m12[cond3] + m21[cond3]) / s3
    q[cond3, 3] = 0.25 * s3

    # Normalize
    q = q / torch.clamp(torch.linalg.norm(q, dim=1, keepdim=True), min=1e-12)
    # Enforce a consistent hemisphere (w >= 0) to avoid sign flips
    flip = q[:, 0] < 0
    q[flip] = -q[flip]
    return q


def compute_log_scales_and_quats_from_L(L: torch.Tensor):
    """
    Compute log-scales and rotation quaternions from a batch of 3x3 matrices L using SVD.
    Returns:
    - log_scales: (N, 3)
    - quats: (N, 4)
    """
    # Batched SVD
    U, S, Vh = torch.linalg.svd(L, full_matrices=False)
    # Enforce a proper rotation (det(U) >= 0) without branching
    detU = torch.det(U)
    fix = torch.where(detU < 0, torch.tensor(-1.0, device=U.device, dtype=U.dtype), torch.tensor(1.0, device=U.device, dtype=U.dtype))
    D = torch.stack([torch.ones_like(fix), torch.ones_like(fix), fix], dim=1)  # (N,3)
    U = torch.matmul(U, torch.diag_embed(D))
    S = torch.clamp(S, min=1e-12)
    log_scales = torch.log(S)
    quats = _rotation_matrix_to_quaternion_batch(U)
    return log_scales, quats


def compute_log_scales_and_quats_from_F_and_base_chol(F: torch.Tensor, base_chol: torch.Tensor):
    """
    Compute log-scales and rotation quaternions for the simulated subset from deformation gradient F
    and precomputed base covariance Cholesky factor base_chol.
    """
    L = torch.bmm(F, base_chol)  # (N,3,3)
    return compute_log_scales_and_quats_from_L(L)


def compute_log_scales_and_quats_from_cov_upper(cov_upper: torch.Tensor):
    """
    Compute log-scales and rotation quaternions from an SPD covariance upper form using eigendecomposition.
    """
    N = cov_upper.shape[0]
    if N == 0:
        return cov_upper.new_zeros((0, 3)), cov_upper.new_zeros((0, 4))
    cov = get_mat_from_upper(cov_upper)  # (N,3,3)
    cov = torch.nan_to_num(cov, nan=0.0, posinf=1e6, neginf=-1e6)
    cov = 0.5 * (cov + cov.transpose(1, 2))
    I = torch.eye(3, device=cov.device, dtype=cov.dtype).unsqueeze(0).repeat(N, 1, 1)
    cov = cov + 1e-6 * I
    cov_cpu = cov.double().cpu()
    evals_cpu, evecs_cpu = torch.linalg.eigh(cov_cpu)
    evals = evals_cpu.to(device=cov.device, dtype=cov.dtype)
    evecs = evecs_cpu.to(device=cov.device, dtype=cov.dtype)
    evals = torch.clamp(torch.nan_to_num(evals, nan=1e-12, posinf=1e6, neginf=1e-12), min=1e-12)
    log_scales = 0.5 * torch.log(evals)
    quats = _rotation_matrix_to_quaternion_batch(evecs)
    return log_scales, quats


def export_gaussians_ply_direct(
    template_gaussians: GaussianModel,
    pos: torch.Tensor,
    shs: torch.Tensor,
    opacity: torch.Tensor,
    log_scales: torch.Tensor,
    quats: torch.Tensor,
    filepath: str,
):
    """
    Export a PLY with precomputed parameters:
    - pos: (N,3)
    - shs: (N, SH, 3)
    - opacity: (N,1) in [0,1]
    - log_scales: (N,3)
    - quats: (N,4) unit quaternions (w,x,y,z)
    """
    gm = GaussianModel(template_gaussians.max_sh_degree)
    gm._xyz = pos.detach()
    gm._features_dc = shs[:, 0:1, :].detach()
    gm._features_rest = shs[:, 1:, :].detach()
    gm._scaling = log_scales.detach()
    gm._rotation = quats.detach()
    gm._opacity = inverse_sigmoid(torch.clamp(opacity, 1e-6, 1 - 1e-6)).detach()
    gm.save_ply(filepath)
