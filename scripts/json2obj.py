import argparse
import json
import math
import os
import sys
from pathlib import Path
from shutil import copyfile

import igl
import numpy as np
import trimesh
from tqdm import tqdm

from scripts.generate_random_color import generate_new_color, generate_color_from_id


def split_path(paths):
    filepath, tempfilename = os.path.split(paths)
    filename, extension = os.path.splitext(tempfilename)
    return filepath, filename, extension


def write_obj_with_tex(savepath, vert, face, vtex, ftcoor, imgpath=None):
    filepath2, filename, extension = split_path(savepath)
    with open(savepath, 'w') as fid:
        fid.write('mtllib ' + filename + '.mtl\n')
        fid.write('usemtl a\n')
        for v in vert:
            fid.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for vt in vtex:
            fid.write('vt %f %f\n' % (vt[0], vt[1]))
        face = face + 1
        ftcoor = ftcoor + 1
        for f, ft in zip(face, ftcoor):
            fid.write('f %d/%d %d/%d %d/%d\n' % (f[0], ft[0], f[1], ft[1], f[2], ft[2]))
    filepath, filename2, extension = split_path(imgpath)
    if os.path.exists(imgpath) and not os.path.exists(filepath2 + '/' + filename + extension):
        copyfile(imgpath, filepath2 + '/' + filename + extension)
    if imgpath is not None:
        with open(filepath2 + '/' + filename + '.mtl', 'w') as fid:
            fid.write('newmtl a\n')
            fid.write('map_Kd ' + filename + extension)


def rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def search_category_mapping(category_mapping, category):
    for c in category_mapping:
        if c["category"] == category or category in c["category"].split(" / "):
            return c
    return None


def list_all_mesh_categories(root_json_path, files, root_future_path, save_path):
    mesh_type = []
    for m in tqdm(files):
        with open(root_json_path + '/' + m, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for mm in data['mesh']:
                mesh_type.append(mm['type'])

    mesh_type = list(set(mesh_type))
    print(len(mesh_type))
    print(mesh_type)


def process_file_list(root_json_path, files, root_future_path, save_path):
    category_mapping = json.loads(Path("3d-front-analyzer/resources/category_mapping.json").read_text())
    current_colors = []

    for c in category_mapping:
        # c['color'] = generate_new_color(current_colors)
        c['color'] = generate_color_from_id(c['id'])
        current_colors.append(c['color'])

    for m in files:
        with open(root_json_path + '/' + m, 'r', encoding='utf-8') as f:
            data = json.load(f)
            model_jid = []
            model_uid = []
            model_bbox = []
            model_category = []

            mesh_uid = []
            mesh_xyz = []
            mesh_faces = []
            mesh_type = []

            if not os.path.exists(save_path + '/' + m[:-5]):
                os.mkdir(save_path + '/' + m[:-5])

            for ff in data['furniture']:
                if 'valid' in ff and ff['valid']:
                    model_uid.append(ff['uid'])
                    model_jid.append(ff['jid'])
                    model_bbox.append(ff['bbox'])
                    model_category.append(ff['category'])

            for mm in data['mesh']:
                mesh_uid.append(mm['uid'])
                mesh_xyz.append(np.reshape(mm['xyz'], [-1, 3]))
                mesh_faces.append(np.reshape(mm['faces'], [-1, 3]))
                mesh_type.append(mm['type'])

            scene = data['scene']
            room = scene['room']

            for r in room:
                room_id = r['instanceid']
                meshes = []
                if not os.path.exists(save_path + '/' + m[:-5] + '/' + room_id):
                    os.mkdir(save_path + '/' + m[:-5] + '/' + room_id)
                children = r['children']
                number = 1
                for c in children:
                    ref = c['ref']
                    type = 'f'
                    try:
                        idx = model_uid.index(ref)
                        if model_category[idx] in ('Ceiling Lamp', "Pendant Lamp"):
                            continue
                        if os.path.exists(root_future_path + '/' + model_jid[idx]):
                            v, vt, _, faces, ftc, _ = igl.read_obj(root_future_path + '/' + model_jid[idx] + '/raw_model.obj')
                            color = search_category_mapping(category_mapping, model_category[idx])["color"]
                            # bbox = np.max(v, axis=0) - np.min(v, axis=0)
                            # s = bbox / model_bbox[idx]
                            # v = v / s
                            # print(model_category[idx])
                    except:
                        try:
                            idx = mesh_uid.index(ref)
                        except:
                            continue
                        # if not ('Baseboard' in mesh_type[idx] or 'Hole' in mesh_type[idx]):
                        #     continue
                        if 'Ceiling' in mesh_type[idx] or 'LightBand' in mesh_type[idx] or 'SlabSide' in mesh_type[idx] or 'WallOuter' in mesh_type[idx] or 'WallBottom' in mesh_type[idx] or 'WallTop' in mesh_type[idx] or 'Back' in mesh_type[idx] or 'Front' in mesh_type[idx]:
                            continue
                        # else:
                        #     print(mesh_type[idx])
                        v = mesh_xyz[idx]
                        faces = mesh_faces[idx]
                        color = search_category_mapping(category_mapping, mesh_type[idx])["color"]
                        type = 'm'

                    pos = c['pos']
                    rot = c['rot']
                    scale = c['scale']
                    v = v.astype(np.float64) * scale
                    ref = [0, 0, 1]
                    axis = np.cross(ref, rot[1:])
                    theta = np.arccos(np.dot(ref, rot[1:])) * 2
                    if np.sum(axis) != 0 and not math.isnan(theta):
                        R = rotation_matrix(axis, theta)
                        v = np.transpose(v)
                        v = np.matmul(R, v)
                        v = np.transpose(v)

                    v = v + pos
                    vertex_colors = np.zeros_like(v)
                    for axis_idx in range(3):
                        vertex_colors[:, axis_idx] = color[axis_idx]
                    if type == 'f':
                        write_obj_with_tex(save_path + '/' + m[:-5] + '/' + room_id + '/' + str(number) + '_' + model_jid[idx] + '.obj', v, faces, vt, ftc, root_future_path + '/' + model_jid[idx] + '/texture.png')
                        meshes.append(trimesh.Trimesh(v, faces, vertex_colors=vertex_colors))
                        number = number + 1
                    else:
                        meshes.append(trimesh.Trimesh(v, faces, vertex_colors=vertex_colors))

                if len(meshes) > 0:
                    temp = trimesh.util.concatenate(meshes)
                    temp.export(save_path + '/' + m[:-5] + '/' + room_id + '/mesh.obj')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--future_path',
        default='./3D-FUTURE-model',
        help='path to 3D FUTURE'
    )
    parser.add_argument(
        '--json_path',
        default='./3D-FRONT',
        help='path to 3D FRONT'
    )

    parser.add_argument(
        '--save_path',
        default='./outputs',
        help='path to save result dir'
    )

    args = parser.parse_args()

    # files = os.listdir(args.json_path)
    # files = ["6d8db384-1df1-46a5-91c6-e34a48275c2c.json", "2be2628f-bec8-4217-9660-805b1c8a1baa.json"]
    files = ["2be2628f-bec8-4217-9660-805b1c8a1baa.json"]

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    process_file_list(args.json_path, files, args.future_path, args.save_path)
