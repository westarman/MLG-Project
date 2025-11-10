import os
import json
import taichi as ti
import numpy as np
import utils
from engine.mpm_solver import MPMSolver
import argparse
import time


ti.init(arch=ti.cuda)  # Use GPU if available; fallback to CPU if not

# ------------------------
# Parse arguments for data saving
# ------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--out-dir', type=str, default='out_data')
parser.add_argument('--save-data', action='store_true')
args = parser.parse_args()

if args.save_data:
    os.makedirs(args.out_dir, exist_ok=True)
    positions_by_frame = []
    velocities_by_frame = []
    materials_by_frame = []

# ------------------------
# Simulation parameters
# ------------------------
# frames - 1000, less particles - 482 particles (we have 1100), 

res = (70, 70)
dt = 8e-3
n_frames = 1000#70


# Make the simulation domain a bit smaller (smaller boundary box)
mpm = MPMSolver(res=res) #, size=0.8

# ------------------------
# Add a rectangle/square of water
# ------------------------
mpm.add_cube(lower_corner=[0.3, 0.7],   # top-left position
             cube_size=[0.2, 0.2],      # width x height
             material=MPMSolver.material_water,
             sample_density=2) 

print("Number of particles:", mpm.n_particles)

# ------------------------
# GUI
# ------------------------
# Smaller GUI window and changed background color
gui = ti.GUI("2D Water Drop", res=400, background_color=0x0D1B2A)
colors = np.array([0x068587, 0xED553B, 0xEEEEF0, 0xFFFF00], dtype=np.uint32)

# Increase gravity to reduce high splashes and add a small velocity damping
mpm.set_gravity((0, -120.0))

# Small damping kernel to softly reduce particle velocities each frame
@ti.kernel
def apply_velocity_damping(f: ti.f32):
    for i in mpm.v:
        mpm.v[i] = mpm.v[i] * f

# ------------------------
# Simulation loop
# ------------------------
# Show initial state and wait
particles = mpm.particle_info()
gui.circles(particles['position'],
           radius=2.5,
           color=colors[particles['material']])
gui.show()
#time.sleep(10)  # wait 10 seconds to allow for window capture

for frame in range(n_frames):
    mpm.step(dt)
    # gentle damping to reduce extreme splashes
    apply_velocity_damping(0.99)
    particles = mpm.particle_info()
    # Render particles slightly bigger for visibility
    gui.circles(particles['position'],
                radius=2.5,
                color=colors[particles['material']])
    
    gui.show()
    
    if args.save_data:
        positions_by_frame.append(np.copy(particles['position']))
        velocities_by_frame.append(np.copy(particles['velocity']))
        materials_by_frame.append(np.copy(particles['material']))

# After simulation: save data if requested
if args.save_data:
    # Stack arrays: shapes (n_frames, n_particles, dim)
    pos_arr = np.stack(positions_by_frame, axis=0).astype(np.float32)
    vel_arr = np.stack(velocities_by_frame, axis=0).astype(np.float32)
    mat_arr = np.stack(materials_by_frame, axis=0).astype(np.int32)

    n_frames_saved, n_particles, dim = pos_arr.shape

    if args.save_data:
        # Compute statistics for metadata
        vel_mean = np.mean(vel_arr.reshape(-1, dim), axis=0).tolist()
        vel_std = np.std(vel_arr.reshape(-1, dim), axis=0).tolist()
        # Accelerations: differences along time axis
        acc = (vel_arr[1:] - vel_arr[:-1]) / dt
        acc_mean = np.mean(acc.reshape(-1, dim), axis=0).tolist()
        acc_std = np.std(acc.reshape(-1, dim), axis=0).tolist()

        # Bounds from all positions
        overall_min = pos_arr.reshape(-1, dim).min(axis=0).tolist()
        overall_max = pos_arr.reshape(-1, dim).max(axis=0).tolist()
        bounds = [[overall_min[i], overall_max[i]] for i in range(dim)]

        metadata = {
            "bounds": bounds,
            "sequence_length": int(n_frames_saved),
            "default_connectivity_radius": 0.015,
            "dim": int(dim),
            "dt": float(dt),
            "vel_mean": vel_mean,
            "vel_std": vel_std,
            "acc_mean": acc_mean,
            "acc_std": acc_std,
        }

        # Group particles by material ID from first frame
        materials0 = mat_arr[0]
        unique_mats = np.unique(materials0)

        # Prepare binary data and offset metadata
        particle_type_blobs = []
        position_blobs = []
        offset = {}
        part_type_offset = 0
        pos_element_offset = 0

        for g_idx, mat in enumerate(unique_mats):
            mask = (materials0 == mat)
            indices = np.where(mask)[0]
            n_group = indices.size

            # Particle types for this group
            ptypes = materials0[indices].astype(np.int64)
            particle_type_blobs.append(ptypes)

            # Positions for this group: shape (n_frames, n_group, dim)
            grp_pos = pos_arr[:, indices, :].astype(np.float32)
            position_blobs.append(grp_pos)

            offset[str(g_idx)] = {
                "particle_type": {"offset": int(part_type_offset), "shape": [int(n_group)]},
                "position": {"offset": int(pos_element_offset), "shape": [int(n_frames_saved), int(n_group), int(dim)]}
            }

            part_type_offset += n_group
            pos_element_offset += int(n_frames_saved) * n_group * dim

        
        with open(os.path.join(args.out_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        with open(os.path.join(args.out_dir, 'train_offset.json'), 'w') as f:
            json.dump(offset, f, indent=2)

        
        with open(os.path.join(args.out_dir, 'train_particle_type.dat'), 'wb') as f:
            for arr in particle_type_blobs:
                arr.tofile(f)
        with open(os.path.join(args.out_dir, 'train_position.dat'), 'wb') as f:
            for arr in position_blobs:
                arr.tofile(f)

        print(f"Saved data to {args.out_dir}/")
        print("Files: metadata.json, train_offset.json, train_particle_type.dat, train_position.dat")