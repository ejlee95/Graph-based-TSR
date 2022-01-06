import numpy as np
import sharedmem
import multiprocessing
import cv2
import os
import json
import sys
import time

from scipy.stats import truncnorm
#from cypress import utils
import scipy.io
from tqdm import tqdm

sys.path.append( os.path.join(os.getcwd(), "code_commons") )
from global_constants import *


def queue_worker(gen_obj, batch_size, batch_dict, queue_in, queue_out):
    g = gen_obj()
    try:
        while True:
            idx = queue_in.get()
            for i in range(batch_size):

                sample_dict = g.__next__()

                if sys.platform == 'win32':
                    assert False
                else:
                    for name in DATA_FIELD_NAMES:
                        batch_dict[name][idx,i, ... ] = sample_dict[name]

            queue_out.put(idx)
    except EOFError:
        return

def generate_batch(gen_obj, batch_size=64, num_processes=4 ):
    #image_size = gen_obj.image_size
    #channels = gen_obj.channels
    #num_landmarks = NUM_LANDMARKS

    num_slots = num_processes * 1

    batch_pool_dict = {}
    if sys.platform == 'win32':
        assert False
    else:
        for idx, name in enumerate(DATA_FIELD_NAMES):
            batch_pool_dict[name]  = sharedmem.empty((num_slots, batch_size,) + DATA_FIELD_SHAPES[idx], dtype=DATA_FIELD_TYPES[idx] )


    manager = multiprocessing.Manager()
    queue_in = manager.Queue()
    queue_out = manager.Queue()

    processes = []
    for i in range(num_processes):
        processes.append(
            multiprocessing.Process(target=queue_worker, args=(
                gen_obj, batch_size, batch_pool_dict, queue_in, queue_out
            )))


    for p in processes:
        p.start()

    for i in range(num_slots):
        queue_in.put(i)

    try:
        while True:
            idx = queue_out.get()
            #   TODO: Copy?
            if sys.platform == 'win32':
                assert False
                images_data = np.array(images[idx * (images_batch): (idx+1) * (images_batch)], dtype='float32')
                landmarks_data = np.array(landmarks[idx * (landmark_batch): (idx+1) * (landmark_batch)], dtype=landmark_data_type)
                yield images_data.reshape(images_shape), landmarks_data.reshape(landmarks_shape)
            else:
                yield_list = []
                for i, name in enumerate(DATA_FIELD_NAMES):
                    yield_list.append( np.array( batch_pool_dict[ DATA_FIELD_NAMES[i] ] [idx,...], dtype = DATA_FIELD_TYPES[i] ) )
                yield yield_list

            queue_in.put(idx)

    except EOFError:
        for p in processes:
            p.join()


