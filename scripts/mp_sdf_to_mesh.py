import sys
from pathlib import Path
import trimesh
import struct
import numpy as np
import marching_cubes as mc
import torch
import os
import psutil


def visualize_sdf(sdf, output_path, level=0.75):
    vertices, triangles = mc.marching_cubes_color(sdf, np.ones(list(sdf.shape) + [3], dtype=sdf.dtype) * 0.5, level)
    mc.export_obj(vertices, triangles, output_path)


def sparse_to_dense_np(locs, values, dimx, dimy, dimz, default_val):
    nf_values = 1 if len(values.shape) == 1 else values.shape[1]
    dense = np.zeros([dimz, dimy, dimx, nf_values], dtype=values.dtype)
    dense.fill(default_val)
    dense[locs[:, 0], locs[:, 1], locs[:, 2], :] = values
    if nf_values > 1:
        return dense.reshape([dimz, dimy, dimx, nf_values])
    return dense.reshape([dimz, dimy, dimx])


def load_sdf(file_path):
    fin = open(file_path, 'rb')
    dimx = np.fromfile(fin, np.uint64, 1)[0]
    dimy = np.fromfile(fin, np.uint64, 1)[0]
    dimz = np.fromfile(fin, np.uint64, 1)[0]
    voxelsize = np.fromfile(fin, np.float32, 1)[0]
    np.fromfile(fin, np.float32, 4*4)
    num = np.fromfile(fin, np.uint64, 1)[0]
    # print((4 * 3 * num) / (1024 * 1024 * 1024))
    locs = np.fromfile(fin, np.uint32, int(num*3))
    locs = locs.reshape([num, 3])
    locs = np.flip(locs, 1)
    sdf = np.fromfile(fin, np.float32, num)
    sdf /= voxelsize
    sdf = sparse_to_dense_np(locs, sdf[:,np.newaxis], dimx, dimy, dimz, -float('inf'))
    sdf[sdf > 8] = 8
    sdf[sdf < -8] = 8
    sdf = sdf.transpose((1, 0, 2))[:, :210, :]
    fin.close()
    return sdf


def get_sdf_size(file_path):
    fin = open(file_path, 'rb')
    dimx = struct.unpack('Q', fin.read(8))[0]
    dimy = struct.unpack('Q', fin.read(8))[0]
    fin.close()
    return dimx * dimy


def export_sdf_as_mesh(psdf, output_dir):
    sdf = load_sdf(str(psdf))
    visualize_sdf(sdf, Path(output_dir) / (psdf.name.split('.')[0] + '.obj'))


def visualize_plot(arr):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1)
    fig.set_size_inches(6, 3)
    n, bins, patches = plt.hist(x=np.array(arr), bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    fig.savefig(f'plot.png', dpi=600)


def get_args():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--sdf_dir", type=str, default='outputs')
    parser.add_argument("--output_dir", type=str, default='outputs')
    parser.add_argument('--num_proc', default=1, type=int)
    parser.add_argument('--proc', default=0, type=int)

    args = parser.parse_args()
    return args


def run_full_export():
    from tqdm import tqdm
    args = get_args()
    list_of_sdfs = sorted([x for x in Path(args.sdf_dir).iterdir() if x.name.endswith(".sdf")], key=lambda x: x.name)
    list_of_sdfs = [x for i, x in enumerate(list_of_sdfs) if i % args.num_proc == args.proc]
    list_of_sdfs = [x for x in list_of_sdfs if get_sdf_size(str(x)) < 1250000]
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    for psdf in tqdm(list_of_sdfs):
        export_sdf_as_mesh(psdf, args.output_dir)


def run_plot():
    from tqdm import tqdm
    sdf_dir = Path("/cluster_HDD/sorona/adai/data/matterport/mp_sdf_vox_1cm_color-complete")
    list_of_sdfs = sorted([x for x in Path(sdf_dir).iterdir() if x.name.endswith(".sdf")], key=lambda x: x.name)
    list_of_sdfs = [x for x in list_of_sdfs if get_sdf_size(str(x)) < 1250000]
    sizes = []
    for psdf in tqdm(list_of_sdfs):
        sizes.append((psdf.name, get_sdf_size(str(psdf))))

    print([x[0] for x in sorted(sizes, key=lambda x: x[1], reverse=True)[:5]])
    visualize_plot(sizes)


def run_test_mesh_export():
    from tqdm import tqdm
    sdf_dir = Path("/cluster_HDD/sorona/adai/data/matterport/mp_sdf_vox_1cm_color-complete")
    test_meshes = ['XcA2TqTSSAj_room31__0__.sdf', 'Z6MFQCViBuw_room3__0__.sdf', 'PX4nDJXEHrG_room13__0__.sdf', 'Uxmj2M2itWa_room30__0__.sdf', 'X7HyMhZNoso_room11__0__.sdf']
    list_of_sdfs = [sdf_dir / t for t in test_meshes]
    Path("output_mp").mkdir(exist_ok=True, parents=True)
    for psdf in tqdm(list_of_sdfs):
        export_sdf_as_mesh(psdf, Path("output_mp"))


if __name__ == "__main__":
    run_full_export()
