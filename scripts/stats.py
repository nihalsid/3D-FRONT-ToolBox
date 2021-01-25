import numpy as np
from pathlib import Path
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import marching_cubes as mc
import trimesh

chunk_highres_path = Path("/cluster/moria/ysiddiqui/3DFrontDistanceFields/chunk_highres")
chunk_cat_path = Path("/cluster/moria/ysiddiqui/3DFrontDistanceFields/chunk_semantics")
scene_highres_path = Path("/cluster/moria/ysiddiqui/3DFrontDistanceFields/complete_highres")
scene_cat_path = Path("/cluster/moria/ysiddiqui/3DFrontDistanceFields/complete_semantics")
threshold = 1
voxel_resolution = 0.054167
voxel_resolution_lr = 0.43334


def to_point_list(s):
    return np.concatenate([c[:, np.newaxis] for c in np.where(s == True)], axis=1)


def visualize_probs(prob, output_path):
    prob = prob > 0.2
    point_list = to_point_list(prob)
    if point_list.shape[0] > 0:
        base_mesh = trimesh.voxel.ops.multibox(centers=point_list, pitch=1)
        base_mesh.export(output_path)


def visualize_sdf(sdf, output_path, level=0.75):
    vertices, triangles = mc.marching_cubes(sdf, level)
    mc.export_obj(vertices, triangles, output_path)


def visualize_sampled_buckets(occupancy=True):
    def find_bucket_idx(buck, occ):
        for k in buck.keys():
            if k[0] <= occ < k[1]:
                return k
    chunks = sorted(chunk_highres_path.iterdir())
    # chunks = [chunk_highres_path / (x + ".npy") for x in Path("chunks_gt_7500vox.txt").read_text().splitlines()]
    if occupancy:
        counts = np.load(f"occupancies_{threshold:.2f}.npy")
        ranges = list(range(0, 100000, 500))
    else:
        counts = np.load(f"categories_{threshold:.2f}.npy")
        ranges = list(range(0, 20, 1))
    buckets = {}
    for b in range(len(ranges) - 1):
        buckets[(ranges[b], ranges[b + 1])] = 0
    for i, x in enumerate(tqdm(chunks)):
        occupancy = counts[i]
        key = find_bucket_idx(buckets, occupancy)
        if buckets[key] < 10:
            df = np.load(x)
            visualize_sdf(df, f"visualizations/{key[0]:06d}_{buckets[key]:02d}_{x.name}.obj", 0.75 * voxel_resolution)
            buckets[key] += 1


def occupied_voxels():
    occupied_vox = []
    chunks = sorted(list(chunk_highres_path.iterdir()))
    # chunks = [chunk_highres_path / (x + ".npy") for x in Path("chunks_gt_7500vox_cat_gt_2.txt").read_text().splitlines()]
    for i, x in enumerate(tqdm(chunks)):
        occupancy = (np.load(x) < threshold * voxel_resolution).sum()
        occupied_vox.append(occupancy)
    fig, ax = plt.subplots(1)
    fig.set_size_inches(6, 3)
    n, bins, patches = plt.hist(x=np.array(occupied_vox), bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    np.save(f"occupancies_{threshold:.2f}", np.array(occupied_vox))
    fig.savefig(f'occupancy.png', dpi=600)


def category_voxels():
    categories = []
    chunks_occ = [chunk_highres_path / (x + ".npy") for x in Path("chunks_gt_7500vox_cat_gt_2.txt").read_text().splitlines()]
    chunks_cat = [chunk_cat_path / (x + ".npy") for x in Path("chunks_gt_7500vox_cat_gt_2.txt").read_text().splitlines()]
    for i in tqdm(list(range(len(chunks_occ)))):
        occupancy = np.load(chunks_occ[i]) < (threshold * voxel_resolution)
        unique_categories = np.unique(np.load(chunks_cat[i])[occupancy]).shape[0]
        categories.append(unique_categories)
    fig, ax = plt.subplots(1)
    fig.set_size_inches(6, 3)
    n, bins, patches = plt.hist(x=np.array(categories), bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    # np.save(f"categories_{threshold:.2f}", np.array(categories))
    fig.savefig(f'categories.png', dpi=600)


def create_list_of_filtered_chunks():
    valid_chunks = []
    occupancies = []
    occupancy_vox = np.load(f"occupancies_{threshold:.2f}.npy")
    chunks = sorted(list(chunk_highres_path.iterdir()))
    # chunks = [chunk_highres_path / (x + ".npy") for x in Path("chunks_gt_7500vox.txt").read_text().splitlines()]
    for i, x in enumerate(tqdm(chunks)):
        occupancy = occupancy_vox[i]  # (np.load(x) < 1 * voxel_resolution).sum()
        if occupancy > 10000:
            valid_chunks.append(x.name.split('.')[0])
            occupancies.append(occupancy)
    Path("chunks_gt_10000vox.txt").write_text("\n".join(valid_chunks))


def create_list_of_category_voxel_chunks():
    valid_chunks = []
    occupancy_vox = np.load(f"occupancies_{threshold:.2f}.npy")
    category_vox = np.load(f"categories_{threshold:.2f}.npy")
    chunks = [chunk_highres_path / (x + ".npy") for x in Path("chunks_gt_7500vox.txt").read_text().splitlines()]
    for i, x in enumerate(tqdm(chunks)):
        if category_vox[i] > 2:
            valid_chunks.append(x.name.split('.')[0])
        elif category_vox[i] == 2:
            if occupancy_vox[i] > 15000:
                valid_chunks.append(x.name.split('.')[0])
    Path("chunks_gt_7500vox_cat_gt_2.txt").write_text("\n".join(valid_chunks))


def calculate_scene_occupancies():
    scenes = list(set(["__".join(x.split("__")[:-2]) for x in Path("chunks_gt_10vox.txt").read_text().splitlines()]))
    Path("scenes_gt_10vox.txt").write_text("\n".join(scenes))
    scenes_highres = [scene_highres_path / (x + ".npy") for x in Path("scenes_gt_10vox.txt").read_text().splitlines()]
    scenes_categories = [scene_cat_path / (x + ".npy") for x in Path("scenes_gt_10vox.txt").read_text().splitlines()]
    occupancies = []
    categories = []
    for i, x in enumerate(tqdm(scenes_highres)):
        occupancy = np.load(scenes_highres[i]) < (threshold * voxel_resolution)
        unique_categories = np.unique(np.load(scenes_categories[i])[occupancy]).shape[0]
        occupancies.append(occupancy.sum())
        categories.append(unique_categories)
    np.save(f"scene_occupancies_{threshold:.2f}", np.array(occupancies))
    np.save(f"scene_categories_{threshold:.2f}", np.array(categories))


def scene_histograms():
    occupancy_vox = np.load(f"scene_occupancies_{threshold:.2f}.npy")
    category_vox = np.load(f"scene_categories_{threshold:.2f}.npy")
    num_bins = 20
    fig, ax = plt.subplots(1)
    fig.set_size_inches(6, 3)
    n, bins, patches = plt.hist(x=np.array(category_vox), bins=num_bins, color='#0504aa', alpha=0.7, rwidth=0.85)
    fig.savefig(f'scene_categories.png', dpi=600)

    num_bins = 'auto'  # [0, 3.5e4, 7e4, 14e4, 21e4, 28e4, 35e4]
    fig, ax = plt.subplots(1)
    fig.set_size_inches(6, 3)
    n, bins, patches = plt.hist(x=np.array(occupancy_vox), bins=num_bins, color='#0504aa', alpha=0.7, rwidth=0.85)
    fig.savefig(f'scene_occupancy.png', dpi=600)


def split_histogram(split):
    scenes_highres = [scene_highres_path / (x + ".npy") for x in Path(f"scenes_{split}.txt").read_text().splitlines()]
    scenes_categories = [scene_cat_path / (x + ".npy") for x in Path(f"scenes_{split}.txt").read_text().splitlines()]
    occupancy_vox = []
    category_vox = []
    for i, x in enumerate(tqdm(scenes_highres)):
        occupancy = np.load(scenes_highres[i]) < (threshold * voxel_resolution)
        unique_categories = np.unique(np.load(scenes_categories[i])[occupancy]).shape[0]
        occupancy_vox.append(occupancy.sum())
        category_vox.append(unique_categories)

    fig, ax = plt.subplots(1)
    fig.set_size_inches(6, 3)
    n, bins, patches = plt.hist(x=np.array(category_vox), bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    fig.savefig(f'scene_{split}_categories.png', dpi=600)

    fig, ax = plt.subplots(1)
    fig.set_size_inches(6, 3)
    n, bins, patches = plt.hist(x=np.array(occupancy_vox), bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    fig.savefig(f'scene_{split}_occupancy.png', dpi=600)


def sample_splits():
    scenes_highres = [x for x in Path("scenes_gt_10vox.txt").read_text().splitlines()]
    occupancy_vox = np.load(f"scene_occupancies_{threshold:.2f}.npy")
    buckets = [3.5e4, 7e4, 14e4]
    bin_0, bin_1, bin_2, bin_3 = [[], [], [], []]
    for i, x in enumerate(tqdm(scenes_highres)):
        if 0 < occupancy_vox[i] <= buckets[0]:
            bin_0.append(x)
        elif buckets[0] < occupancy_vox[i] <= buckets[1]:
            bin_1.append(x)
        elif buckets[1] < occupancy_vox[i] <= buckets[2]:
            bin_2.append(x)
        elif buckets[2] < occupancy_vox[i]:
            bin_3.append(x)
    train, val, test = [[], [], []]
    for bin in [bin_0, bin_1, bin_2, bin_3]:
        train.extend(random.sample(bin, int(len(bin) * 0.7)))
        remaining = [x for x in bin if x not in train]
        val.extend(random.sample(remaining, int(len(bin) * 0.15)))
        test.extend([x for x in remaining if x not in val])
    print(len(train), len(val), len(test))
    for s in [("train", train), ("val", val), ("test", test)]:
        Path(f"scenes_{s[0]}.txt").write_text("\n".join(s[1]))


def create_chunk_splits():
    chunks = Path("chunks_gt_10000vox.txt").read_text().splitlines()
    for split in tqdm(["train", "val", "test"]):
        split_scenes = Path(f"scenes_{split}.txt").read_text().splitlines()
        split_chunks = [c for c in chunks if "__".join(c.split("__")[:2]) in split_scenes]
        Path(split + ".txt").write_text("\n".join(split_chunks))


def create_visualization_split():
    def find_bucket_idx(buck, occ):
        for k in buck.keys():
            if k[0] <= occ < k[1]:
                return k
    ranges = list(range(0, 200000, 2000))
    for split in ["train", "val"]:
        vis_list = []
        buckets = {}
        for b in range(len(ranges) - 1):
            buckets[(ranges[b], ranges[b + 1])] = 0
        split_chunks = Path(f"{split}.txt").read_text().splitlines()
        random.shuffle(split_chunks)
        for i, c in enumerate(tqdm(split_chunks)):
            df = np.load(chunk_highres_path / (c + ".npy"))
            key = find_bucket_idx(buckets, (df < threshold * voxel_resolution).sum())
            if buckets[key] < 5:
                vis_list.append(c)
                buckets[key] += 1
        Path(f"{split}_vis.txt").write_text("\n".join(vis_list))


def visualize_split():
    split_chunks = Path(f"val_vis.txt").read_text().splitlines()
    for i, c in enumerate(tqdm(split_chunks)):
        df = np.load(chunk_highres_path / (c + ".npy"))
        visualize_sdf(df, f"visualizations/{c}.obj", 0.75 * voxel_resolution)


def remove_library_chunks():
    for split in ["train.txt", "val.txt", "test.txt", "train_eval.txt", "val_vis.txt", "train_vis.txt"]:
        split_chunks = [x for x in Path(split).read_text().splitlines() if "Library" not in x]
        Path(split).write_text("\n".join(split_chunks))


def adapt_splits_from_chunks(src_chunk_dir=Path("stats_main")):
    current_chunks = Path("chunks_gt_10000vox.txt").read_text().splitlines()
    for idx, split in enumerate(["train.txt", "val.txt", "test.txt", "train_eval.txt", "val_vis.txt", "train_vis.txt"]):
        src_split_chunks = (src_chunk_dir / split).read_text().splitlines()
        src_split_scenes = set(["__".join(x.split("__")[:-2]) for x in src_split_chunks])
        split_chunks = [x for x in current_chunks if "__".join(x.split("__")[:2]) in src_split_scenes]
        if idx > 2:
            split_chunks = random.sample(split_chunks, int(0.35 * len(split_chunks)))
        print(split, len(split_chunks))
        Path(split).write_text("\n".join(split_chunks))


def visualize_all_chunks_from_scene(path_to_lr, path_to_hr, scene):
    chunks_lr = [x for x in path_to_lr.iterdir() if x.name.startswith(scene)]
    chunks_hr = [x for x in path_to_hr.iterdir() if x.name.startswith(scene)]
    for c in chunks_hr:
        visualize_sdf(np.load(c), f"visualizations/{c.name.split('.')[0]}_target.obj", threshold * voxel_resolution)
    for c in chunks_lr:
        visualize_probs((np.load(c) < 0.5 * voxel_resolution_lr), f"visualizations/{c.name.split('.')[0]}_cond.obj")


if __name__ == "__main__":
    visualize_all_chunks_from_scene(Path("/cluster/moria/ysiddiqui/repatch/data/sdf_008/3DFront"), Path("/cluster/moria/ysiddiqui/repatch/data/sdf_064/3DFront"), '00004f89-9aa5-43c2-ae3c-129586be8aaa__MasterBedroom-5863')
