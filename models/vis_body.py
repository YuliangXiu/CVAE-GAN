from human_body_prior.tools.visualization_tools import *
from human_body_prior.tools.omni_tools import colors
from smpl_np import MeshViewer
import matplotlib.pyplot as plt
import trimesh, os, cv2, sys
from tqdm import tqdm
import numpy as np
import pyrender



obj_dir = '/home/ICT2000/yxiu/Pictures/CVPR2020/samples_multi_infer/'
img_dir = '/home/ICT2000/yxiu/Pictures/CVPR2020/samples_multi_infer_rgb/'
imw, imh = 150, 256
mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
mv.set_background_color(colors['white'])
parts_num = 17
std_num = 11

# for _,_,obj_paths in os.walk(obj_dir):
#     for obj_path in tqdm(obj_paths):
#         if 'ply' in obj_path:
#             obj = trimesh.load_mesh(os.path.join(obj_dir, obj_path))
#             mv.set_meshes([obj], 'static')
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

# all_parts = range(17)
# not_include_parts = [0,12,16,3,6]
# include_parts = [item for item in all_parts if item not in not_include_parts]

# final = np.zeros((imh*(parts_num-len(not_include_parts)), imw*std_num, 3))

# for _,_,obj_paths in os.walk(obj_dir):
#     for obj_path in tqdm(obj_paths):
#         if 'ply' in obj_path:
#             obj = trimesh.load_mesh(os.path.join(obj_dir, obj_path))
#             part_id = int(obj_path.split("_")[3])
#             if part_id not in not_include_parts:
#                 part_id = include_parts.index(part_id)
#                 std_id = int(obj_path.split("_")[5])+5
#                 mv.set_meshes([obj], 'static')
#                 final[part_id*imh:(part_id+1)*imh, std_id*imw:(std_id+1)*imw] = mv.render()

# cv2.imwrite("/home/ICT2000/yxiu/Pictures/CVPR2020/vposer.png", final)


obj_dir = '/home/ICT2000/yxiu/Pictures/CVPR2020/middle_samples_infer/'
imw, imh = 150, 256
mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
mv.set_background_color(colors['black'])
mid_num = 10

final = np.zeros((imh, imw*(mid_num+2), 3))

for _,_,obj_paths in os.walk(obj_dir):
    for obj_path in tqdm(obj_paths):
        if 'ply' in obj_path and "ORIG" not in obj_path:
            obj = trimesh.load_mesh(os.path.join(obj_dir, obj_path))
            if "INFER" not in obj_path:
                mid_id = int(obj_path.split("_")[7])
                mv.set_meshes([obj], 'static')
                final[:, (mid_id+1)*imw:(mid_id+2)*imw] = mv.render()
            elif "start_recover" in obj_path:
                mv.set_meshes([obj], 'static')
                final[:, :imw] = mv.render()
            elif "end_recover" in obj_path:
                mv.set_meshes([obj], 'static')
                final[:,-imw:] = mv.render()

                
cv2.imwrite("/home/ICT2000/yxiu/Pictures/CVPR2020/interpolation.png", final)



