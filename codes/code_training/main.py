import numpy as np
import tensorflow as tf
import os, sys
import cv2


sys.path.append( os.path.join(os.getcwd(), "code_commons") )
from global_constants import *
from auxiliary_ftns import *

sys.path.append(os.path.join(os.getcwd(), "code_training"))
import set_default_training_options
import training_help_ftns

from model_test import *
from junctionnet import JAnet
import os

MAIN_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(MAIN_DIR, "experiments")

TRAINING_FILE_PATHS = [  ]

FLAGS = set_default_training_options.get_flags(MAIN_DIR)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)


def main(args):
    #############################################################################
    # Configuration
    #############################################################################
    if sys.version_info[0] != 3:
        raise Exception(f"ERROR: You must use Python 3.7 "
                                        f" but you are running Python {sys.version_info[0]}")

    # Prints Tensorflow version
    print(f"This code was developed and tested on TensorFlow 1.13.0. " f"Your TensorFlow version: {tf.__version__}.")


   # Defines {FLAGS.train_dir}, maybe based on {FLAGS.experiment_dir}
    if not FLAGS.experiment_name:
        raise Exception("You need to specify an --experiment_name or --train_dir.")

    FLAGS.experiment_name = experiment_name = FLAGS.experiment_name
    FLAGS.train_dir = train_dir = (FLAGS.train_dir or os.path.join(OUTPUT_DIR, experiment_name))

    print("###########################")
    print(f"experiment_name: {experiment_name}")
    print(f"train_dir: {train_dir}")
    print(f"batch size: {FLAGS.batch_size}")
    print(f"mode: {FLAGS.mode}")
    print(f"data_dir: {FLAGS.data_dir}")
    print(f"validation tfrecord: {FLAGS.validation_dataset_file_path}")
    print(f"num_samples_per_learning_rate_half_decay: {FLAGS.num_samples_per_learning_rate_half_decay}")
    print("###########################")

    if not os.path.exists(train_dir):
        os.mkdir(train_dir)

    janet = JAnet( FLAGS )

    cpu_mode = False

    if cpu_mode is False:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
    else:
        config = tf.ConfigProto(device_count = {'GPU': 0})

    if FLAGS.mode == "train":
        janet.train_initialize( FLAGS.data_dir, cpu_mode = cpu_mode  )
        if len(FLAGS.validation_dataset_file_path) != 0:
            janet.add_validation_env( FLAGS.validation_dataset_file_path )

        with tf.Session(config=config) as sess:
            training_help_ftns.initialize_model(sess, janet, train_dir )
            janet.train(sess)

    elif FLAGS.mode == 'validation':
        janet.runttime_initialize( )
        janet.add_validation_env( FLAGS.validation_dataset_file_path )
        with tf.Session(config=config) as sess:
            training_help_ftns.initialize_model(sess, janet, train_dir, expect_exists=True)
            janet.evaluate_validation_loss( sess )

    elif FLAGS.mode == 'test':
        janet.runttime_initialize()

        with tf.Session(config=config) as sess:
            ckpt_basename = training_help_ftns.initialize_model(sess, janet, train_dir, expect_exists=True)

            run_test( sess, janet, FLAGS, 'folder', srcname = [FLAGS.test_data_dir+"/*.jpg", FLAGS.test_data_dir+"/*.png"], wait_time = 0, save_result_image=True, num_extra_candidates=FLAGS.nextra, niter_th=FLAGS.niter, save_intermediate_optimization=FLAGS.save_inter, test_scan=FLAGS.test_scan, with_detection=FLAGS.with_detection)
            os._exit(0)



    os._exit(0)

if __name__ == '__main__':
  tf.app.run()
