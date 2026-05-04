#!/usr/bin/env python3
"""Render gaussian splats using gsplat for proper splatting quality."""
import sys
import struct
import numpy as np
import torch
from pathlib import Path
from PIL import Image


def load_ply_gaussians(ply_path: str):
    """Load 3DGS PLY format: xyz, SH dc, opacity, scales, rotations."""
    from plyfile import PlyData
    ply = PlyData.read(ply_path)
    v = ply['vertex']
    n = len(v)
    print(f"Loaded {n:,} gaussians from {ply_path}")

    xyz = np.stack([v['x'], v['y'], v['z']], axis=-1).astype(np.float32)

    # SH dc coefficients -> RGB via SH2RGB: color = 0.5 + C0 * sh_dc
    C0 = 0.28209479177387814  # 1/(2*sqrt(pi))
    r = 0.5 + C0 * v['f_dc_0'].astype(np.float32)
    g = 0.5 + C0 * v['f_dc_1'].astype(np.float32)
    b = 0.5 + C0 * v['f_dc_2'].astype(np.float32)
    colors = np.stack([r, g, b], axis=-1)
    colors = np.clip(colors, 0, 1)

    opacity = v['opacity'].astype(np.float32)
    # sigmoid activation
    opacity = 1.0 / (1.0 + np.exp(-opacity))

    scales = np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=-1).astype(np.float32)
    scales = np.exp(scales)

    quats = np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], axis=-1).astype(np.float32)
    # normalize
    quats = quats / (np.linalg.norm(quats, axis=-1, keepdims=True) + 1e-8)

    return {
        'means': torch.tensor(xyz, device='cuda'),
        'colors': torch.tensor(colors, device='cuda'),
        'opacities': torch.tensor(opacity, device='cuda'),
        'scales': torch.tensor(scales, device='cuda'),
        'quats': torch.tensor(quats, device='cuda'),
    }


def make_camera(W, H, fx, fy, cx, cy, c2w):
    """Build camera intrinsics (K) and extrinsics (viewmat = w2c)."""
    viewmat = torch.linalg.inv(c2w).unsqueeze(0)  # [1, 4, 4]
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                     dtype=torch.float32, device='cuda').unsqueeze(0)  # [1, 3, 3]
    return viewmat, K


def look_at_matrix(eye, target, up=None):
    """Build a camera-to-world matrix looking from eye to target."""
    if up is None:
        up = np.array([0, 0, 1], dtype=np.float32)  # Z-up
    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    fwd = target - eye
    fwd = fwd / (np.linalg.norm(fwd) + 1e-8)
    right = np.cross(fwd, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    new_up = np.cross(right, fwd)

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = new_up
    c2w[:3, 2] = -fwd  # OpenGL convention: camera looks along -Z
    c2w[:3, 3] = eye
    return torch.tensor(c2w, device='cuda')


def render_view(gaussians, W, H, fx, fy, cx, cy, c2w, bg_color=(1, 1, 1)):
    """Render one view using gsplat."""
    from gsplat import rasterization

    viewmat, K = make_camera(W, H, fx, fy, cx, cy, c2w)

    means = gaussians['means']
    quats = gaussians['quats']
    scales = gaussians['scales']
    opacities = gaussians['opacities']
    colors = gaussians['colors']

    renders, alphas, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat,
        Ks=K,
        width=W,
        height=H,
        sh_degree=None,
    )

    img = renders[0].clamp(0, 1)
    alpha = alphas[0]
    bg = torch.tensor(bg_color, dtype=torch.float32, device='cuda').view(1, 1, 3)
    img = img * alpha + bg * (1 - alpha)
    return img.detach().cpu().numpy(), alpha.detach().cpu().numpy()


def main():
    ply_path = sys.argv[1] if len(sys.argv) > 1 else "output/marble_test/scene.ply"
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "output/marble_test/renders")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading gaussians...")
    gaussians = load_ply_gaussians(ply_path)

    # Analyze scene bounds
    means = gaussians['means'].cpu().numpy()
    print(f"Scene bounds: min={means.min(0)}, max={means.max(0)}")
    print(f"Scene center: {means.mean(0)}")

    W, H = 1296, 968
    cx, cy = W / 2, H / 2
    up = np.array([0, 1, 0], dtype=np.float32)  # Y-up
    eye = np.array([0, 0, 0], dtype=np.float32)

    # FOV 55 gives correct object size. But camera looks too far down:
    # towels should be at top edge, floor/trashcan visible at bottom.
    # Reduce downward tilt (less negative Y in target).
    views = [
        ('tilt_a', (-3, -1, -10), 55),
        ('tilt_b', (-3, -2, -10), 55),
        ('tilt_c', (-3, -1.5, -10), 55),
        ('tilt_d', (-3, -1, -10), 50),
        ('tilt_e', (-3, -2, -10), 50),
        ('tilt_f', (-2.5, -1.5, -10), 55),
        ('tilt_g', (-3, -0.5, -10), 55),
        ('tilt_h', (-3, 0, -10), 55),
    ]

    for name, target, hfov in views:
        fx = fy = (W / 2) / np.tan(np.radians(hfov / 2))
        target = np.array(target, dtype=np.float32)
        print(f"\nRendering {name}: target={target} hfov={hfov} fx={fx:.0f}px")
        c2w = look_at_matrix(eye, target, up)
        img, alpha = render_view(gaussians, W, H, fx, fy, cx, cy, c2w)
        img_uint8 = (img * 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(str(out_dir / f"gs_{name}.png"))
        print(f"  Saved gs_{name}.png, alpha coverage: {alpha.mean():.2%}")

    print("\nDone!")


if __name__ == "__main__":
    main()
