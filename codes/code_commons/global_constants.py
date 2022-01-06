import numpy as np 
import tensorflow as tf 

IMAGE_WIDTH = 512
IMAGE_HEIGHT = IMAGE_WIDTH 
IMAGE_CHANNEL = 3
STANDARD_CHARACTER_HEIGHT = 20 

HEATMAP_WIDTH = 256
HEATMAP_HEIGHT = 256
HEATMAP_CHANNEL = 9 #+2
AFFINITY_CHANNEL = 2

QUANTIZE = False 

COMMON_COLOR_NAMES = ['maroon', 'red', 'brown', 'green', 'limegreen', 'magenta', 'purple', 'violet', 'blue', 'white', 'white' ]


label_to_vector_label = np.array([
    [1,1,0,0], 
    [1,1,1,0], 
    [0,1,1,0], 
    [1,1,0,1],
    [1,1,1,1],
    [0,1,1,1],
    [1,0,0,1],
    [1,0,1,1],
    [0,0,1,1],
    [1,0,1,0],    
    [0,1,0,1],
    [0,0,0,0]],
    dtype = np.int32 )


vector_label_to_label = np.ones( shape=[2,2,2,2], dtype = np.int32 ) * (-10)
for idx in range( label_to_vector_label.shape[0] ):
    vector_label_to_label[ label_to_vector_label[idx][0], label_to_vector_label[idx][1], \
                          label_to_vector_label[idx][2], label_to_vector_label[idx][3]] = idx
vector_label_to_label[0,0,0,0] = -1



DATA_FIELD_NAMES = [ "image", "affinity", "heatmap" ]
DATA_FIELD_TYPES = [ np.float32, np.float32, np.float32 ]
DATA_FIELD_TF_TYPES = [ tf.float32, tf.float32, tf.float32 ]
DATA_FIELD_SHAPES = [ (IMAGE_HEIGHT,IMAGE_WIDTH,3), (HEATMAP_HEIGHT,HEATMAP_WIDTH,AFFINITY_CHANNEL), (HEATMAP_HEIGHT,HEATMAP_WIDTH,HEATMAP_CHANNEL)]

