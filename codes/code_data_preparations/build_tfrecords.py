import json
import cv2
import os, sys 
import numpy as np 
import skimage.transform
from glob import glob
import skimage.transform
import tensorflow as tf 
from tqdm import tqdm

MAIN_DIR = os.getcwd()
COMMON_DIR = os.path.join(MAIN_DIR, "code_commons")
sys.path.append( COMMON_DIR )
COMMON_DIR = os.path.join(MAIN_DIR, "code_training")
sys.path.append( COMMON_DIR )

import tfrecord_utils 
from tfrecord_utils import _bytes_feature

from global_constants import *  
from train_sample_generator import *

tfrecords_filename = os.path.join(MAIN_DIR, "validation.tfrecords")


if __name__ == "__main__":
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    datadir = ['test/validation/']
    data_generator = TrainDataGenerator( datadir)
    dataset = data_generator.dataset

    num_files = len(dataset)

#    import ipdb
#    ipdb.set_trace()


    for i in tqdm.tqdm( range(num_files), desc=f"jpg_json to tfrecord conversion", total=num_files):
        jpgPath, annotations = dataset[i]
        img = cv2.imread(jpgPath)
        H, W, _ = img.shape
        scale =  np.float(STANDARD_CHARACTER_HEIGHT)/annotations["character_height"]
        H = int(scale*H)
        W = int(scale*W)
        numH = np.int(np.ceil(H/IMAGE_HEIGHT))
        numW = np.int(np.ceil(W/IMAGE_WIDTH))

#        while True:
        for h in range(numH):
            for w in range(numW):
                data_dict, validity = data_generator.generate_data( jpgPath, annotations, w, h )   
                if validity is False:
                    print(jpgPath, w, h)
                    continue
#                if validity is True:
#                    break

                feature = {}
                data_string_dict = {}
                for name in DATA_FIELD_NAMES:
                    data_string_dict[name] = data_dict[name].tostring() 
                    if DATA_FIELD_TYPES is np.uint8:
                        feature[name] = _bytes_feature( data_string_dict[name] )
                    else:
                        feature[name] = _bytes_feature( tf.compat.as_bytes( data_string_dict[name] ) )
        
        
                features = tf.train.Features(feature=feature)
                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())

    writer.close()
