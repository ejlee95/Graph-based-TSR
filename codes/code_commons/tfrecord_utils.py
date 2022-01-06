import numpy as np 
import tensorflow as tf 
import cv2
import glob, os
from global_constants import *

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def parser(serialized_example):
    features_dict = {}
    for name in DATA_FIELD_NAMES:
        features_dict[name] =  tf.FixedLenFeature([], tf.string)

    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features=features_dict)

    data_dict = {}
    for idx,name in enumerate(DATA_FIELD_NAMES):
        data_dict[name] = tf.decode_raw(features[name], DATA_FIELD_TF_TYPES[idx] )
        data_dict[name] = tf.reshape( data_dict[name], DATA_FIELD_SHAPES[idx] )

    return data_dict 




def make_batch(filenames, const_batch_size, shuffle = True, num_epochs = 1, MIN_QUEUE_EXAMPLES = 1000 ):
    """Read the images and labels from 'filenames'."""

    for file in filenames:
        num_samples = sum(1 for _ in tf.python_io.tf_record_iterator(file))    
        print( num_samples, "samples in ", file )

    for filename in filenames:
        if os.path.exists(filename) is False:
            raise Exception("no such file" + filename)

    # Repeat infinitely.
    if num_epochs == 1:
        dataset = tf.data.TFRecordDataset(filenames)
    else:
        dataset = tf.data.TFRecordDataset(filenames).repeat()

    # Parse records.
    dataset = dataset.map( parser, num_parallel_calls=10)

    # Potentially shuffle records.
    if shuffle is True:
        dataset = dataset.shuffle(buffer_size=MIN_QUEUE_EXAMPLES)

    # Batch it up.
    dataset = dataset.batch(const_batch_size)
    iterator = dataset.make_one_shot_iterator()

    return  iterator.get_next()


def get_available_cpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    result = [x.name for x in local_device_protos if x.device_type == 'CPU']
    if len(result) > 0: return result
    return [x.name for x in local_device_protos]
