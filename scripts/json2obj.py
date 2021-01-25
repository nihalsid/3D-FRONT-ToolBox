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
from intersections import slice_mesh_plane
from generate_random_color import generate_new_color, generate_color_from_id


proc_id = None


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
    room_types = []
    for m in tqdm(files):
        with open(root_json_path + '/' + m, 'r', encoding='utf-8') as f:
            data = json.load(f)
            scene = data['scene']
            room = scene['room']
            for mm in data['mesh']:
                mesh_type.append(mm['type'])
            for r in room:
                room_id = r['instanceid']
                room_types.append(room_id.split('-')[0])
    mesh_type = list(set(mesh_type))
    room_types = list(set(room_types))
    print(len(room_types), room_types)
    print(len(mesh_type), mesh_type)


def process_file_list(root_json_path, files, root_future_path, save_path):
    global proc_id
    category_mapping = json.loads(Path("3d-front-analyzer/resources/category_mapping.json").read_text())
    current_colors = []
    valid_room_types = ['DiningRoom', 'Bedroom', 'LivingDiningRoom', 'KidsRoom', 'MasterBedroom', 'LivingRoom', 'ElderlyRoom', 'SecondBedroom', 'Library', 'NannyRoom']

    for c in category_mapping:
        # c['color'] = generate_new_color(current_colors)
        c['color'] = generate_color_from_id(c['id'])
        current_colors.append(c['color'])

    for m in tqdm(files):
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
                room_type = room_id.split('-')[0]
                instance_ctr = 0
                if room_type not in valid_room_types:
                    continue
                meshes = []
                struct_meshes = []
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
                            # color = generate_color_from_id(instance_ctr)
                            instance_ctr += 1
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
                        # color = generate_color_from_id(instance_ctr)
                        instance_ctr += 1
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
                    if v.shape[0] == 0:
                        continue
                    if type == 'f':
                        # write_obj_with_tex(save_path + '/' + m[:-5] + '/' + room_id + '/' + str(number) + '_' + model_jid[idx] + '.obj', v, faces, vt, ftc, root_future_path + '/' + model_jid[idx] + '/texture.png')
                        try:
                            appended_mesh = trimesh.Trimesh(v, faces, vertex_colors=vertex_colors)
                            #scale
                            bbox = appended_mesh.bounding_box.bounds
                            scale = bbox[1][1] - bbox[0][1]
                            if scale > 2.6:
                                origin = bbox[0]
                                appended_mesh.apply_translation(-(bbox[0] + bbox[1]) / 2)
                                appended_mesh.apply_scale(2.6 / scale)
                                appended_mesh.apply_translation(origin - appended_mesh.bounding_box.bounds[0])
                            # ground
                            bbox = appended_mesh.bounding_box.bounds
                            y_bounds = bbox[0][1]
                            if y_bounds < 0:
                                appended_mesh.apply_translation(np.array([0, -y_bounds, 0]))
                            meshes.append(appended_mesh)
                            number = number + 1
                        except IndexError:
                            continue
                    else:
                        try:
                            appended_mesh = trimesh.Trimesh(v, faces, vertex_colors=vertex_colors)
                            y_bounds = appended_mesh.bounding_box.bounds[0][1]
                            if y_bounds < 0:
                                appended_mesh.apply_translation(np.array([0, -y_bounds, 0]))
                            meshes.append(appended_mesh)
                            struct_meshes.append(appended_mesh)
                            # if mesh_type[idx] in ["CustomizedFurniture", "Cabinet", "CustomizedFixedFurniture"]:
                            #     meshes.append(appended_mesh)
                        except IndexError:
                            continue

                if len(meshes) > 0:
                    # temp_structs = trimesh.util.concatenate(struct_meshes)
                    # temp_structs = trimesh.util.concatenate(meshes)
                    temp = trimesh.util.concatenate(meshes)
                    bbox = temp.bounding_box.bounds
                    # translate to origin
                    temp.apply_translation(-bbox[0])

                    for _m in meshes:
                        _m.apply_translation(-bbox[0])

                    # temporary save
                    # bbox = temp.bounding_box.bounds
                    # temp.apply_translation(-bbox[0])
                    # temp.export(save_path + '/' + m[:-5] + '/' + room_id + '/mesh_before.obj')

                    # crop
                    bbox = temp.bounding_box.bounds
                    scaled_extents = np.array([(bbox[1] - bbox[0])[0] * 1.125, 2.6025, (bbox[1] - bbox[0])[2] * 1.125])
                    box = trimesh.creation.box(extents=scaled_extents)
                    box.apply_translation(scaled_extents / 2)
                    temp = slice_mesh_plane(mesh=temp, plane_normal=-box.facets_normal, plane_origin=box.facets_origin)

                    for i, _m in enumerate(meshes):
                        meshes[i] = slice_mesh_plane(mesh=_m, plane_normal=-box.facets_normal, plane_origin=box.facets_origin)

                    # recenter
                    bbox = temp.bounding_box.bounds
                    loc = (bbox[0] + bbox[1]) / 2
                    scale = (bbox[1] - bbox[0])[1]
                    temp.apply_translation(-loc)
                    temp.apply_scale(2.6 / scale)

                    for i, _m in enumerate(meshes):
                        _m.apply_translation(-loc)
                        _m.apply_scale(2.6 / scale)

                    # start at 0
                    bbox = temp.bounding_box.bounds
                    temp.apply_translation(-bbox[0])

                    for i, _m in enumerate(meshes):
                        _m.apply_translation(-bbox[0])
                        _m.export(save_path + '/' + m[:-5] + '/' + room_id + '/' + f'{i:04d}' + '.obj')

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

    parser.add_argument(
        '--num_proc',
        default=1,
        type=int
    )

    parser.add_argument(
        '--proc',
        default=0,
        type=int
    )

    args = parser.parse_args()

    files = os.listdir(args.json_path)
    #/cluster/gondor/ysiddiqui/3DFrontDistanceFields/complete_highres/1a82a427-49cb-4394-b9d7-8ff02c42e18f__MasterBedroom-5463.npy

    # bad_files = Path("bad_meshes.txt").read_text().splitlines()
    # files = [f.split('/')[0]+'.json' for f in bad_files]
    # files = ["0003d406-5f27-4bbf-94cd-1cff7c310ba1.json"]
    # files = ["6d8db384-1df1-46a5-91c6-e34a48275c2c.json", "2be2628f-bec8-4217-9660-805b1c8a1baa.json"]
    # files = ["2be2628f-bec8-4217-9660-805b1c8a1baa.json", "1a82a427-49cb-4394-b9d7-8ff02c42e18f.json"]
    # files = ['6d8db384-1df1-46a5-91c6-e34a48275c2c.json', 'c33366ef-4801-4764-8ad5-ebbf2e36337a.json', 'fd0e8518-9dc9-4922-a4b9-dc00c825bd21.json', 'fd0e8518-9dc9-4922-a4b9-dc00c825bd21.json', 'fd0e8518-9dc9-4922-a4b9-dc00c825bd21.json',
    #  'fe717e28-bb7e-4705-a176-b78780ffd7ad.json', '1652d2f7-4a27-402e-8f22-0b0625256e8e.json', '1652d2f7-4a27-402e-8f22-0b0625256e8e.json', '7ed12290-2536-4fff-92d2-0f32525f949b.json', '2a001497-73d9-4172-a89a-90ce19d94ed2.json',
    #  '6bde9708-fd42-4cf1-bdf1-a291639a71cd.json', 'e19d78e4-6fbf-4e68-9ad9-e12d1edfabf4.json', '73ccd93b-b2eb-4456-9a64-816006c825f9.json']
    files = [x for i, x in enumerate(files) if i % args.num_proc == args.proc]
    proc_id = args.proc
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    process_file_list(args.json_path, files, args.future_path, args.save_path)
