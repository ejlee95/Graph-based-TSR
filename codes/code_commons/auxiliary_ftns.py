import functools
import cv2 
import numpy as np 
from global_constants import *
import os 

drawthickline = True


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    result = [x.name for x in local_device_protos if x.device_type == 'GPU']
    if len(result) > 0: return result
    return [x.name for x in local_device_protos]


def tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()), 1)


def draw_contour( img, landmarks, start_idx, end_idx, c ):
    thickness = 1#np.random.randint(0,3) % 2+ 1
    if drawthickline is True:
        thickness = 2
    for j in range(start_idx,end_idx):
        pt1 = (landmarks[j,0], landmarks[j,1])
        pt2 = (landmarks[(j+1),0], landmarks[(j+1),1])
        cv2.line( img, pt1, pt2, color=c,thickness=thickness)
    pt1 = (landmarks[start_idx,0], landmarks[start_idx,1])
    pt2 = (landmarks[end_idx,0], landmarks[end_idx,1])
    cv2.line( img, pt1, pt2, color=c, thickness=thickness)            


def draw_junctions( junction_map, base_img = None  ):
    import get_colors_with_names
    
    if base_img is None:
        junction_img = np.zeros( shape=(HEATMAP_HEIGHT, HEATMAP_WIDTH, 3), dtype = np.float32 )
    else:
        junction_img = base_img
        junction_img = junction_img.astype( np.float32 )
        junction_img = cv2.resize( junction_img, (HEATMAP_WIDTH, HEATMAP_HEIGHT ))


    for i in range( HEATMAP_CHANNEL ):
        color = get_colors_with_names.get_color_with_name( COMMON_COLOR_NAMES[i] )
        junction_img[:,:,0] = junction_img[:,:,0] + color[2] * junction_map[...,i]
        junction_img[:,:,1] = junction_img[:,:,1] + color[1] * junction_map[...,i]
        junction_img[:,:,2] = junction_img[:,:,2] + color[0] * junction_map[...,i]
    

    junction_img = np.clip( junction_img, 0, 255 )
    junction_img = junction_img.astype( np.uint8 )
    return junction_img


def draw_junctions_32f( junction_map, base_img = None  ):
    import get_colors_with_names

    rst = []    

    for i in range( junction_map.shape[0]):    
        
        if base_img is not None:
            junction_img = draw_junctions( junction_map[i], base_img[i] )
        else:
            junction_img = draw_junctions( junction_map[i] )
        junction_img = junction_img.astype( np.float32 ) / 255.0
        junction_img = cv2.resize( junction_img, (IMAGE_WIDTH,IMAGE_WIDTH))

        rst.append( junction_img )

    rst = np.array( rst )
    return rst
