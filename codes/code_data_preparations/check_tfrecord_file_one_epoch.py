import numpy as np 
import tensorflow as tf 
import cv2
import os, sys
import math
from skimage.draw import circle 


MAIN_DIR = os.getcwd()
COMMON_DIR = os.path.join(MAIN_DIR, "code_commons")
sys.path.append( COMMON_DIR )
import tfrecord_utils 
from global_constants import * 


DATA_DIR = os.path.join( MAIN_DIR, "data")
tfrecords_filename = 'validation.tfrecords'


def main():
    validation_set = [tfrecords_filename]

    batch_size_tensor = tf.placeholder( tf.int32, shape=() )
    validation_batch_size = 1

    data_dict = tfrecord_utils.make_batch(validation_set, validation_batch_size, shuffle=False, num_epochs=1 )

    directory_name = os.path.join( DATA_DIR, "sample_images" )
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)

    width = height = IMAGE_WIDTH
    channels = 3 
    nx = 10
    ny = 6
    num_images = nx * ny # I want {num_images} images
 
    with tf.Session() as sess:
        sess.run( tf.global_variables_initializer() )

        img_samples = []
        for idx in range(num_images):
            _data_dict = sess.run( data_dict)
            _images = _data_dict["image"] / np.sqrt(2.0) + 0.5
            img_samples.append( _images[0,...] )


        img_samples = np.array( img_samples )
        img_samples = img_samples.reshape(ny,nx,height,width,channels).transpose(0,2,1,3,4).reshape(height*ny,width*nx,channels)            


        cv2.imshow( "img", img_samples[:,:,::-1] )
        key = cv2.waitKey(0)



if __name__ == "__main__":
    main()