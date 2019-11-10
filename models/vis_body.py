from human_body_prior.tools.visualization_tools import *
from human_body_prior.tools.omni_tools import colors
from smpl_np import MeshViewer
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import trimesh, os, cv2, sys
from tqdm import tqdm
import numpy as np
import pyrender



obj_dir = '/home/ICT2000/yxiu/Pictures/CVPR2020/samples_multi_mats_infer/'
img_dir = '/home/ICT2000/yxiu/Pictures/CVPR2020/samples_multi_mats_infer_rgb/'
imw, imh = 150, 256
mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
mv.set_background_color(colors['white'])
# imw, imh = 1000, 1000

parts_num = 17

# for _,_,obj_paths in os.walk(obj_dir):
#     for obj_path in tqdm(obj_paths):
#         if 'ply' in obj_path:
#             obj = trimesh.load_mesh(os.path.join(obj_dir, obj_path))
#             mv.set_meshes([obj], 'static')

#             camera_pose = np.eye(4)
#             #right leg
#             camera_pose[:3,3] = np.array([-1.0, 0.0, 0.0])
#             camera_pose[:3,:3] = R.from_euler('zyx', [-90, -85, 0], degrees=True).as_dcm()
#             #right arm
#             camera_pose[:3,3] = np.array([-1.0, 0.0, 0.4])
#             camera_pose[:3,:3] = R.from_euler('zyx', [-90, -70, 0], degrees=True).as_dcm()
#             #left arm
#             camera_pose[:3,3] = np.array([0.0, -1.0, 0.4])
#             camera_pose[:3,:3] = R.from_euler('zyx', [0, 0, 70], degrees=True).as_dcm()
#             #left leg
#             camera_pose[:3,3] = np.array([1.0, -0.1, 0.0])
#             camera_pose[:3,:3] = R.from_euler('zyx', [90, 85, 5], degrees=True).as_dcm()

#             # for angle_x in tqdm(range(-180,180,90)):
#             #     for angle_y in range(-180,180,90):
#             #         for angle_z in range(-180,180,90):
#             #             for x in np.linspace(-1.0,1.0,5):
#             #                 for y in np.linspace(-1.0,1.0,5):
#             #                     for z in np.linspace(-1.0,1.0,5):

#             # camera_pose[:3,3] = np.array([x, y, z])
#             # camera_pose[:3,:3] = R.from_euler('zyx', [angle_z, angle_y, angle_x], degrees=True).as_dcm()
#             mv.update_camera(camera_pose)
#             # cv2.imwrite(os.path.join(img_dir, obj_path[:-4]+"anglex_%d_angley_%d_anglez_%d_x_%f_y_%f_z_%f.jpg"%(angle_x, angle_y, angle_z, x, y, z)), mv.render())
#             cv2.imwrite(os.path.join(img_dir, obj_path[:-4]+".jpg"), mv.render())

#             break 
#         break 
#     break


# fig, axes = plt.subplots(nrows=parts_num, ncols=std_num)

# for _,_,obj_paths in os.walk(obj_dir):
#     for obj_path in tqdm(obj_paths):
#         if 'ply' in obj_path:
#             obj = trimesh.load_mesh(os.path.join(obj_dir, obj_path))
#             part_id = int(obj_path.split("_")[3])
#             std_id = int(obj_path.split("_")[5])+5
#             mv.set_meshes([obj], 'static')
#             axes[part_id, std_id].imshow(mv.render())
#             axes[part_id, std_id].axis('off')

# fig.tight_layout()
# plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
# plt.savefig("/home/ICT2000/yxiu/Pictures/CVPR2020/vposer.png", dpi=512)

all_parts = range(17)
not_include_parts = [0,12,16,3,6,8]
std_num = 21
include_parts = [item for item in all_parts if item not in not_include_parts]
final = np.zeros((imh*(parts_num-len(not_include_parts)), imw*std_num, 3))

#right leg
camera_rl = np.eye(4)
camera_rl[:3,3] = np.array([-1.0, 0.0, 0.0])
camera_rl[:3,:3] = R.from_euler('zyx', [-90, -85, 0], degrees=True).as_dcm()
#right arm
camera_ra = np.eye(4)
camera_ra[:3,3] = np.array([-1.0, 0.0, 0.4])
camera_ra[:3,:3] = R.from_euler('zyx', [-90, -70, 0], degrees=True).as_dcm()
#left arm
camera_la = np.eye(4)
camera_la[:3,3] = np.array([0.0, -1.0, 0.4])
camera_la[:3,:3] = R.from_euler('zyx', [0, 0, 70], degrees=True).as_dcm()
#left leg
camera_ll = np.eye(4)
camera_ll[:3,3] = np.array([1.0, -0.1, 0.0])
camera_ll[:3,:3] = R.from_euler('zyx', [90, 85, 5], degrees=True).as_dcm()


cam_dict = {1:camera_ll, 2:camera_ll, 3:camera_ll, 4:camera_rl, 5:camera_rl, 
    6:camera_rl, 7:camera_ra, 8:camera_ra, 9:camera_la, 10:camera_la, 11:camera_la, 
    12:camera_la, 13:camera_ra, 14:camera_ra, 15:camera_ra, 16:camera_ra}

for _,_,obj_paths in os.walk(obj_dir):
    for obj_path in tqdm(obj_paths):
        if 'ply' in obj_path:
            obj = trimesh.load_mesh(os.path.join(obj_dir, obj_path))
            part_id = int(obj_path.split("_")[3])
            if part_id in include_parts:
                mv.update_camera(cam_dict[part_id])
                part_id = include_parts.index(part_id)
                std_id = int(obj_path.split("_")[5])+10
                mv.set_meshes([obj], 'static')
                final[part_id*imh:(part_id+1)*imh, std_id*imw:(std_id+1)*imw] = mv.render()

cv2.imwrite("/home/ICT2000/yxiu/Pictures/CVPR2020/vposer_new.png", final)


# obj_dir = '/home/ICT2000/yxiu/Pictures/CVPR2020/middle_samples_infer/'
# imw, imh = 150, 256
# mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
# mv.set_background_color(colors['black'])
# mid_num = 10

# final = np.zeros((imh, imw*(mid_num+2), 3))

# for _,_,obj_paths in os.walk(obj_dir):
#     for obj_path in tqdm(obj_paths):
#         if 'ply' in obj_path and "ORIG" not in obj_path:
#             obj = trimesh.load_mesh(os.path.join(obj_dir, obj_path))
#             if "INFER" not in obj_path:
#                 mid_id = int(obj_path.split("_")[7])
#                 mv.set_meshes([obj], 'static')
#                 final[:, (mid_id+1)*imw:(mid_id+2)*imw] = mv.render()
#             elif "start_recover" in obj_path:
#                 mv.set_meshes([obj], 'static')
#                 final[:, :imw] = mv.render()
#             elif "end_recover" in obj_path:
#                 mv.set_meshes([obj], 'static')
#                 final[:,-imw:] = mv.render()

                
# cv2.imwrite("/home/ICT2000/yxiu/Pictures/CVPR2020/interpolation.png", final)



