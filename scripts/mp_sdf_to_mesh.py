import sys
from pathlib import Path
import trimesh
import struct
import numpy as np
import marching_cubes as mc
import torch


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
    dimx = struct.unpack('Q', fin.read(8))[0]
    dimy = struct.unpack('Q', fin.read(8))[0]
    dimz = struct.unpack('Q', fin.read(8))[0]
    voxelsize = struct.unpack('f', fin.read(4))[0]
    world2grid = struct.unpack('f'*4*4, fin.read(4*4*4))
    world2grid = np.asarray(world2grid, dtype=np.float32).reshape([4, 4])
    num = struct.unpack('Q', fin.read(8))[0]
    locs = struct.unpack('I'*num*3, fin.read(num*3*4))
    locs = np.asarray(locs, dtype=np.int32).reshape([num, 3])
    locs = np.flip(locs, 1).copy() # convert to zyx ordering
    sdf = struct.unpack('f'*num, fin.read(num*4))
    sdf = np.asarray(sdf, dtype=np.float32)
    sdf /= voxelsize
    sdf = sparse_to_dense_np(locs, sdf[:,np.newaxis], dimx, dimy, dimz, -float('inf'))
    sdf[sdf > 8] = 8
    sdf[sdf < -8] = 8
    sdf = sdf.transpose((1, 0, 2))[:, :210, :]
    return sdf


if __name__=="__main__":
    # sdf = load_sdf("/mnt/sorona_hdd_adai/data/matterport/mp_sdf_vox_1cm_color-complete/29hnd4uzFmX_room0__0__.sdf")
    from argparse import ArgumentParser
    from tqdm import tqdm

    parser = ArgumentParser()
    parser.add_argument("--sdf_dir", type=str, default='outputs')
    parser.add_argument("--output_dir", type=str, default='outputs')
    parser.add_argument('--num_proc', default=1, type=int)
    parser.add_argument('--proc', default=0, type=int)

    args = parser.parse_args()
    list_of_sdfs = sorted([x for x in Path(args.sdf_dir).iterdir() if x.name.endswith(".sdf")], key=lambda x:x.name)
    list_of_sdfs = [x for i, x in enumerate(list_of_sdfs) if i % args.num_proc == args.proc]
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    for psdf in tqdm(list_of_sdfs):
        sdf = load_sdf(str(psdf))
        visualize_sdf(sdf, Path(args.output_dir) / (psdf.name.split('.')[0] + '.obj'))