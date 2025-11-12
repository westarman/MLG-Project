import os
import json
import taichi as ti
import numpy as np
import argparse
import utils  # for python to find engine.mpm_solver
from engine.mpm_solver import MPMSolver
from math import isfinite

parser = argparse.ArgumentParser()
parser.add_argument('--out-dir', type=str, default='out_data')
parser.add_argument('--save-data', action='store_true')
parser.add_argument('--show-gui', action='store_true')
parser.add_argument('--memory-fraction', type=float, default=0.9, help='Taichi CUDA device memory fraction (0~1).')
parser.add_argument('--n-sims', type=int, default=3, help='How many simulations to run')
parser.add_argument('--seed', type=int, default=None, help='Random seed (for reproducibility)')
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

# --- globals
res = (70, 70)
dt = 8e-3
n_frames = 70
gravity = (0, -120.0)
damping = 0.99
dim = 2

rng = np.random.default_rng(args.seed)

# --- random cube sampler (keeps away from edges and near the top in y)
MARGIN = 0.05
SIZE_MIN, SIZE_MAX = 0.15, 0.25  # around 0.2 +/- 0.05
Y_TOP_MIN, Y_TOP_MAX = 0.60, 0.85  # bias higher so water can fall

def sample_cube_params():
    sx = float(rng.uniform(SIZE_MIN, SIZE_MAX))
    sy = float(rng.uniform(SIZE_MIN, SIZE_MAX))
    # x lower_corner
    x_low = MARGIN
    x_high = 1.0 - MARGIN - sx
    x0 = float(rng.uniform(x_low, max(x_low, x_high)))
    # y lower_corner (biased high)
    y_high_cap = 1.0 - MARGIN - sy
    y_range_low = min(Y_TOP_MIN, y_high_cap)
    y_range_high = max(Y_TOP_MIN, min(Y_TOP_MAX, y_high_cap))
    if y_range_high < y_range_low:  # if size makes top range invalid, fallback to margin-safe range
        y_range_low = MARGIN
        y_range_high = max(MARGIN, y_high_cap)
    y0 = float(rng.uniform(y_range_low, max(y_range_low, y_range_high)))
    return [x0, y0], [sx, sy]

# --- fixed solid cube (same place for all simulations)
SOLID_LOWER_CORNER = [0.75, 0.70]   # safely inside domain with MARGIN
SOLID_SIZE         = [0.15, 0.15]   # solid cube size
SOLID_MATERIAL     = MPMSolver.material_elastic  # "solid" cube

# define kernels (can be defined before init/reset)
@ti.kernel
def apply_velocity_damping_kernel(f: ti.f32, v: ti.template()):
    for i in v:
        v[i] = v[i] * f

def run_one_sim(water_lower_corner, water_cube_size, show_gui=False):
    # Recreate GUI after a reset (if requested)
    gui = None
    colors = np.array([0x068587, 0xED553B, 0xEEEEF0, 0xFFFF00], dtype=np.uint32)
    if show_gui:
        gui = ti.GUI("2D Water + Solid", res=400, background_color=0x0D1B2A)

    # fresh solver for this simulation
    mpm = MPMSolver(res=res)
    mpm.set_gravity(gravity)

    # Water cube (random each sim)
    mpm.add_cube(lower_corner=water_lower_corner, cube_size=water_cube_size,
                 material=MPMSolver.material_water, sample_density=2)

    # Solid cube (fixed across sims)
    mpm.add_cube(lower_corner=SOLID_LOWER_CORNER, cube_size=SOLID_SIZE,
                 material=SOLID_MATERIAL, sample_density=2)

    positions_by_frame = []
    velocities_by_frame = []
    materials_by_frame = []

    if gui is not None:
        p0 = mpm.particle_info()
        gui.circles(p0['position'], radius=2.5, color=colors[p0['material']])
        gui.show()

    for _ in range(n_frames):
        mpm.step(dt)
        apply_velocity_damping_kernel(damping, mpm.v)
        p = mpm.particle_info()

        if gui is not None:
            gui.circles(p['position'], radius=2.5, color=colors[p['material']])
            gui.show()

        positions_by_frame.append(np.copy(p['position']))
        velocities_by_frame.append(np.copy(p['velocity']))
        materials_by_frame.append(np.copy(p['material']))

    pos_arr = np.stack(positions_by_frame, axis=0).astype(np.float32)  # (T,N,2)
    vel_arr = np.stack(velocities_by_frame, axis=0).astype(np.float32) # (T,N,2)
    mat_arr = np.stack(materials_by_frame, axis=0).astype(np.int32)    # (T,N)
    return pos_arr, vel_arr, mat_arr

# -------- offsets + stats
offset = {}
part_type_offset = 0          # number of int64s written so far
pos_element_offset = 0        # number of float32 elements written so far
global_group_index = 0

# Online stats accumulators (Welford)
def welford_update(state, x):
    # state = (n, mean, M2)
    n, mean, M2 = state
    for row in x:
        n += 1
        delta = row - mean
        mean = mean + delta / n
        M2 = M2 + delta * (row - mean)
    return (n, mean, M2)

def welford_finalize(state):
    n, mean, M2 = state
    if n < 2:
        return mean, np.zeros_like(mean)
    var = M2 / (n - 1)
    return mean, np.sqrt(np.maximum(var, 0))

vel_state = (0, np.zeros(dim, dtype=np.float64), np.zeros(dim, dtype=np.float64))
acc_state = (0, np.zeros(dim, dtype=np.float64), np.zeros(dim, dtype=np.float64))
bounds_min = np.full(dim, np.inf, dtype=np.float64)
bounds_max = np.full(dim, -np.inf, dtype=np.float64)

# -------- open binary files once; append per simulation
pt_path = os.path.join(args.out_dir, 'train_particle_type.dat')
pos_path = os.path.join(args.out_dir, 'train_position.dat')
off_path = os.path.join(args.out_dir, 'train_offset.json')
meta_path = os.path.join(args.out_dir, 'metadata.json')

if args.save_data:
    # truncate existing files
    open(pt_path, 'wb').close()
    open(pos_path, 'wb').close()

# ---- loop sims; reset Taichi between runs to free GPU memory ----
for sim_idx in range(args.n_sims):
    # fresh Taichi runtime each sim
    if sim_idx == 0:
        ti.init(arch=ti.cuda, device_memory_fraction=args.memory_fraction, default_fp=ti.f32)
    else:
        ti.sync()
        ti.reset()
        ti.init(arch=ti.cuda, device_memory_fraction=args.memory_fraction, default_fp=ti.f32)

    water_lower_corner, water_cube_size = sample_cube_params()
    print(
        f"Running simulation {sim_idx}: "
        f"water.lower_corner={water_lower_corner}, water.size={water_cube_size}; "
        f"solid.lower_corner={SOLID_LOWER_CORNER}, solid.size={SOLID_SIZE}"
    )

    pos_arr, vel_arr, mat_arr = run_one_sim(water_lower_corner, water_cube_size, show_gui=args.show_gui)

    # ---- stats update (online; tiny memory) ----
    vel_state = welford_update(vel_state, vel_arr.reshape(-1, dim))
    if vel_arr.shape[0] > 1:
        acc = (vel_arr[1:] - vel_arr[:-1]) / dt  # (T-1,N,2)
        acc_state = welford_update(acc_state, acc.reshape(-1, dim))
    # bounds
    flat_pos = pos_arr.reshape(-1, dim)
    bounds_min = np.minimum(bounds_min, flat_pos.min(axis=0))
    bounds_max = np.maximum(bounds_max, flat_pos.max(axis=0))

    # ---- group by material (first frame) and APPEND to binaries ----
    if args.save_data:
        with open(pt_path, 'ab') as f_pt, open(pos_path, 'ab') as f_pos:
            materials0 = mat_arr[0]
            unique_mats = np.unique(materials0)

            for mat in unique_mats:
                indices = np.where(materials0 == mat)[0]
                n_group = int(indices.size)
                if n_group == 0:
                    continue

                # particle types (int64) and positions (float32)
                ptypes = materials0[indices].astype(np.int64)               # (n_group,)
                grp_pos = pos_arr[:, indices, :].astype(np.float32)         # (T, n_group, 2)

                # write immediately (append)
                ptypes.tofile(f_pt)
                grp_pos.tofile(f_pos)

                # choose params based on material (water vs solid)
                if mat == MPMSolver.material_water:
                    obj_name = "water"
                    lc = [float(water_lower_corner[0]), float(water_lower_corner[1])]
                    sz = [float(water_cube_size[0]), float(water_cube_size[1])]
                else:
                    obj_name = "solid"
                    lc = [float(SOLID_LOWER_CORNER[0]), float(SOLID_LOWER_CORNER[1])]
                    sz = [float(SOLID_SIZE[0]), float(SOLID_SIZE[1])]

                # record offset entry
                T = int(pos_arr.shape[0])
                key = str(global_group_index)
                offset[key] = {
                    "sim_index": int(sim_idx),
                    "particle_type": {"offset": int(part_type_offset), "shape": [n_group]},
                    "position": {"offset": int(pos_element_offset), "shape": [T, n_group, dim]},
                    "object": obj_name,
                    "params": {
                        "lower_corner": lc,
                        "cube_size": sz
                    }
                }

                # advance running offsets
                part_type_offset += n_group
                pos_element_offset += T * n_group * dim
                global_group_index += 1

        # persist offsets after each sim (crash-safe)
        with open(off_path, 'w') as f:
            json.dump(offset, f, indent=2)

# -------- write metadata at the end (uses online stats) --------
if args.save_data:
    vel_mean, vel_std = welford_finalize(vel_state)
    acc_mean, acc_std = welford_finalize(acc_state)
    overall_min = np.where(np.isfinite(bounds_min), bounds_min, 0.0).tolist()
    overall_max = np.where(np.isfinite(bounds_max), bounds_max, 0.0).tolist()

    metadata = {
        "bounds": [[overall_min[i], overall_max[i]] for i in range(dim)],
        "sequence_length": int(n_frames),
        "default_connectivity_radius": 0.015,
        "dim": int(dim),
        "dt": float(dt),
        "vel_mean": vel_mean.astype(float).tolist(),
        "vel_std": vel_std.astype(float).tolist(),
        "acc_mean": acc_mean.astype(float).tolist(),
        "acc_std": acc_std.astype(float).tolist(),
        "num_sims": int(args.n_sims),
        "note": "Multiple sims concatenated; binaries appended per sim; see train_offset.json for per-group offsets and params."
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved aggregated data to {args.out_dir}/")
else:
    print("Skipping save (run with --save-data to write files).")
