import tensorflow as tf
import numpy as np
import os, sys
import time
import tqdm
import heatmap.base_net as net

sys.path.append( os.path.join(os.getcwd(), "code_commons") )
from config import th_conf
from global_constants import *
from tfrecord_utils import *
import auxiliary_ftns

from train_sample_generator import *
import train_data_provider

import sharedmem
import threading

from tqdm import tqdm
from auxiliary_ftns import *

import tensorflow.contrib.slim as slim


import tensorflow as tf
import tensorflow.contrib.layers as tcl
from tensorflow.contrib.framework import arg_scope

from tensorflow.contrib import slim
from tensorflow.contrib.layers import xavier_initializer

import random


class JAnet(object):
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS


        self.max_feature_depth = 128 # 256
        self.hourglass_depth = 3 # 4

        if self.FLAGS.save_best:
            self.min_val_loss = 100000

    def runttime_initialize(self, add_saver = True ):
        self.input = tf.placeholder( tf.float32, [None, None, None, IMAGE_CHANNEL])
        self.output = self.build_heatmap_graph( self.input, reuse = False, is_training = False )

        if QUANTIZE is True:
            tf.contrib.quantize.create_eval_graph()

        if add_saver is True:
            self.add_saver()



    def add_validation_env(self,  validation_set ):
        self.num_validation_samples = 0
        if os.path.splitext(self.FLAGS.validation_dataset_file_path[0])[1] == '.tfrecords':
            print('initializing_tfrecord_for_validation')

            with tf.variable_scope('tfrecord_open'):
                self.validation_set = validation_set
                validation_batch_size = 1

                num_validation_sizes = []
                num_validation_samples = 0
                for file in validation_set:
                    num_samples = sum(1 for _ in tf.python_io.tf_record_iterator(file))
                    num_validation_samples +=  num_samples
                    num_validation_sizes = num_validation_sizes + [num_samples]
                print(f"validation samples = {num_validation_samples}")
                self.num_validation_samples = num_validation_samples

                self.validation_data_dict = make_batch(validation_set, validation_batch_size, \
                     shuffle = False, num_epochs = 80000, MIN_QUEUE_EXAMPLES = 10 )

        self.val_image = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
        self.val_gt_affinity = tf.placeholder(tf.float32, [None, HEATMAP_HEIGHT, HEATMAP_WIDTH, 2])
        self.val_gt_heatmap = tf.placeholder(tf.float32, [None, HEATMAP_HEIGHT, HEATMAP_WIDTH, HEATMAP_CHANNEL])

        self.val_gt_maps = tf.concat([self.val_gt_heatmap, self.val_gt_affinity], axis=-1)
        self.val_heatmaps = self.build_heatmap_graph(self.val_image, reuse=True, is_training=False)

        gpus = get_available_gpus()

        with tf.variable_scope('validation_stuff'):
            self.validation_heatmap_loss = self.heatmap_loss(self.val_heatmaps, self.val_gt_maps)
            self.add_summary_per_gpu_validation(0, self.validation_heatmap_loss[0], self.val_image, self.val_gt_heatmap, self.val_gt_affinity, self.val_heatmaps[self.FLAGS.NUM_STACKS-1]  )
            # self.l2_loss = square_sum_error = tf.reduce_mean( tf.square( depth_map - self.val_output) )   #
        self.val_summaries = tf.summary.merge_all(scope='validation_stuff')


    def evaluate_validation_loss(self,sess, step):
        print( 'validation_started')

        if self.num_validation_samples == 0:
            return
        tic = time.time()
        validation_default_batch_size = 1
        loss_sum = 0.0

        random_i = random.randint(0, self.num_validation_samples)

        for i in tqdm( range(self.num_validation_samples), total=self.num_validation_samples, leave=False):
            _data_dict = sess.run( self.validation_data_dict  )
            if i == random_i:
                summary_data_dict = _data_dict

            _loss, step_loss= sess.run(self.validation_heatmap_loss, feed_dict = {self.val_image: _data_dict["image"], self.val_gt_affinity: _data_dict["affinity"], self.val_gt_heatmap: _data_dict["heatmap"]} )
            loss_sum = loss_sum + _loss

        val_summary = sess.run(self.val_summaries, feed_dict = {self.val_image: summary_data_dict["image"], self.val_gt_affinity: summary_data_dict["affinity"], self.val_gt_heatmap: summary_data_dict["heatmap"]})
        self.writer.add_summary(val_summary, global_step=step)


        toc = time.time()
        print(f'elapsed={toc-tic}sec')
        avg_loss = np.sqrt(loss_sum/self.num_validation_samples)
        print( "avg loss = ", avg_loss )
        return avg_loss



    def heatmap_regression(self,inp_img, num_output_channel,
            stack=2, max_feature_depth=128, hourglass_depth=4,
            conv=net.dsconv,
            reuse=False,  scope=None): #net.dsconv,
        if scope is None:
            scope_name = 'heatmap'
        else:
            scope_name = "heatmap_%s" % ( scope)

        # tf.logging.info("making var scope %s" % scope_name)

        with tf.variable_scope(scope_name, reuse=reuse):
            f = max_feature_depth
            f2 = f // 2
            f4 = f // 4

            #print(f"====> {scope_name} inputs: {inp_img.get_shape()}")

            conv1 = conv(inp_img, [7, 7, 3, f4], 'conv1', stride=[1,2,2,1])
            conv2 = conv(conv1, [3, 3, f4, f2], 'conv2')
            #pool1 = net.pool(conv2, 'pool1')
            pool1 = conv2
            conv2b = conv(pool1, [3, 3, f2, f2], 'conv2b')
            conv3 = conv(conv2b, [3, 3, f2, f2], 'conv3')
            conv4 = conv(conv3, [3, 3, f2, f], 'conv4')

            inter = conv4

            preds = []
            for i in range(stack):
                # Hourglass
                hg = net.hourglass(conv, inter, hourglass_depth, f, i, f2)
                #print(f'----> hg_{i}: {hg.get_shape()}')
                # Final output
                conv5 = conv(hg, [3, 3, f, f], 'conv5_%d' % i)
                conv6 = conv(conv5, [1, 1, f, f], 'conv6_%d' % i)
                #print(f"message_passing: {message_passing}")

                pred = conv(conv6, [1, 1, f, num_output_channel], 'out_%d' % i, dorelu=False)

                preds += [pred] #original
                # pred_sigmoid = tf.nn.sigmoid(pred)
                # preds += [ pred_sigmoid ] #variant

                # Residual link across hourglasses
                if i < stack-1:
                    """
                    predx = preds[-1]

                    f_before = int(inter.get_shape()[-1])
                    inter = tf.concat( [predx, additional_channels[i]], axis = -1 )
                    f_after =  int(inter.get_shape()[-1]  )
                    inter = conv(inter, [1, 1, f_after, f_before], 'channel_%d' % i, dorelu = False )

                    inter = inter + conv(conv6, [1, 1, f, f], 'tmp_%d' % i, dorelu=False) + conv(predx, [1, 1, num_output_channel, f], 'tmp_out_%d'%i, dorelu = False)

                    predx = preds[-1]
                    if i == self.FLAGS.NUM_STACKS - 2:
                        print( 'REGRESSION FUSION at %s-th output'%(i) )
                        predx, f_before = self.fusion( conv, predx, additional_channels[i], i)
                    else:
                        f_before = int(predx.get_shape()[-1])

                    assert num_output_channel == f_before

                    inter = inter + conv(conv6, [1, 1, f, f], 'tmp_%d' % i, dorelu=False) + conv(predx, [1, 1, f_before, f], 'tmp_out_%d'%i, dorelu = False)
                    """
                    inter = inter + conv(conv6, [1, 1, f, f], 'tmp_%d' % i, dorelu=False)  + conv(preds[-1], [1, 1, num_output_channel, f], 'tmp_out_%d'%i, dorelu = False)


            #print(f"====> {scope_name} outputs: {stack}x{preds[0].get_shape()}")
            return preds


    def build_heatmap_graph( self, image_slice, reuse, is_training  ):

        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            heatmaps = self.heatmap_regression(
                                image_slice,
                                HEATMAP_CHANNEL + AFFINITY_CHANNEL,
                                max_feature_depth=self.max_feature_depth,
                                hourglass_depth=self.hourglass_depth,
                                reuse=reuse,
                                stack=self.FLAGS.NUM_STACKS )

        return heatmaps


    def heatmap_loss(self, heatmaps, gt_heatmaps): #, gt_junction_labels):
        """
        Loss function definition
        """

        """
        heatmap_shape = heatmaps[0].get_shape()
        batch_size = heatmap_shape[0].value

        assert heatmap_shape[1] == heatmap_shape[2]
        num_landmarks = heatmap_shape[3].value
        total_loss = tf.zeros((), dtype='float32')
        step_losses = []

        assert num_landmarks == INNER_NUM_LANDMARKS
        """


        total_loss = tf.zeros((), dtype='float32')
        step_losses = []
        for idx, heatmap in enumerate(heatmaps):
            step_loss = tf.reduce_mean( tf.square( gt_heatmaps - heatmap) )
            step_losses.append(step_loss)

            total_loss += step_loss

        # cross-entropy loss
#        gt_junction_one_hot = tf.one_hot(gt_junction_labels, depth=10)
#        junction_type_loss = tf.nn.softmax_cross_entropy_with_logits_v2(gt_junction_one_hot, heatmaps)

        return total_loss, step_losses


    def train_initialize(self, datadir = None, cpu_mode = False ):

        assert IMAGE_WIDTH == IMAGE_HEIGHT

        if len(datadir) == 1 and os.path.splitext( datadir[0] )[-1] == ".tfrecords":
            batch_data_dict = make_batch(datadir, self.FLAGS.batch_size, shuffle=True, num_epochs=self.FLAGS.num_epochs) #10000 )

        else:
            ########################## data pipeline ####################
            tic = time.time()
            data_generator = TrainDataGenerator( datadir )
            #data_generator()
            toc = time.time()
            print("###########################")
            print(f'data loading={toc-tic}sec')

            batch_generator = train_data_provider.generate_batch(data_generator,
                batch_size=self.FLAGS.batch_size,
                num_processes=self.FLAGS.num_preprocessing_processes)

            lock = threading.Lock()
            def generate_batch():
                with lock:
                    batch_data_list = batch_generator.__next__()
                return batch_data_list
            ################################################################

            batch_list = tf.py_func(generate_batch, [], DATA_FIELD_TYPES, stateful=True)

            batch_data_dict = {}

            for idx, name in enumerate(DATA_FIELD_NAMES):
                batch_data_dict[name] = batch_list[idx]
                batch_data_dict[name].set_shape((self.FLAGS.batch_size,) +  DATA_FIELD_SHAPES[idx] )
                print( DATA_FIELD_NAMES[idx] + ":", batch_data_dict[name].shape, batch_data_dict[name].dtype )


        self.batch_data_dict = batch_data_dict

        if cpu_mode is True:
            gpus = ['/device:CPU:0']
            print(gpus)
        else:
            gpus = get_available_gpus()
            print(gpus)

        num_gpus = len(gpus)
        assert(self.FLAGS.batch_size % num_gpus == 0)
        batch_slice = self.FLAGS.batch_size // num_gpus

        tower_losses = []

        for idx_gpu, gpu in enumerate(gpus):
            print( gpu )
            with tf.device(gpu):
                image_slice = batch_data_dict["image"][batch_slice*idx_gpu:batch_slice*(idx_gpu+1),...]
                affinity_slice = batch_data_dict["affinity"][batch_slice*idx_gpu:batch_slice*(idx_gpu+1),...]                # dimension = INNER_NUM_LANDMARKS
                heatmap_slice = batch_data_dict["heatmap"][batch_slice*idx_gpu:batch_slice*(idx_gpu+1),...]

                heatmaps = self.build_heatmap_graph( image_slice, reuse=(idx_gpu>0), is_training = True)

                for heatmap in heatmaps:
                    print(heatmap.shape )

                with tf.variable_scope('hgnet_loss'):
                    gt_slice = tf.concat( [heatmap_slice,affinity_slice], axis = -1)
                    loss, step_losses = self.heatmap_loss(heatmaps, gt_slice )

                with tf.variable_scope('janet_loss'):
                    tower_losses.append( loss )

                self.add_summary_per_gpu( idx_gpu, loss, image_slice, heatmap_slice, affinity_slice, heatmaps[self.FLAGS.NUM_STACKS-1]  )

        if QUANTIZE is True:
            self.add_quant_training_graph()
        self.print_var_status( var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES ) )


        self.loss = tf.reduce_mean(tower_losses)
        self.add_gradient()
        self.add_summary()
        self.add_saver()


    @staticmethod
    def convert_to_color_image( x ):
        x = tf.expand_dims( x, axis=-1 )
        x = tf.tile( x, [1,1,1,3])
        return x


    def add_summary_per_gpu(self, gpu_idx, loss, image_slice, heatmap_slice, affinity_slice, heatmaps ):
        num_summary_images = self.FLAGS.num_summary_images


        tf.summary.histogram("input_histogram", image_slice)


        with tf.variable_scope( 'summary_%s'%(gpu_idx) ):
            tf.summary.scalar("heatmap_loss_%s_th_gpu"%(gpu_idx), loss )

            image_slice = ( ( image_slice ) / np.sqrt(2.0) + 0.5 )

            """
            depth_map = RTnet.convert_to_color_image( depth_map )
            gt_depth_map = RTnet.convert_to_color_image( gt_depth_map )

            images_with_lines = tf.py_func( draw_landmarks, \
                [ image_slice[:num_summary_images], landmarks_slice[:num_summary_images] ], tf.float32 )
            """

            images_with_gtjunctions = tf.py_func( draw_junctions_32f, \
                [ heatmap_slice[:num_summary_images,:,:,:HEATMAP_CHANNEL], 40.*image_slice[:num_summary_images,:,:,:HEATMAP_CHANNEL]  ], tf.float32 )


            zero_plane = tf.zeros( shape=(num_summary_images, HEATMAP_HEIGHT, HEATMAP_WIDTH,1), dtype=tf.float32)
            images_with_gt_affinity = tf.concat( [affinity_slice[:num_summary_images], zero_plane], axis = -1 )
            images_with_gt_affinity = tf.image.resize_bilinear( images_with_gt_affinity, (IMAGE_HEIGHT,IMAGE_WIDTH) )



            images_with_est_junctions = tf.py_func( draw_junctions_32f, \
                [ heatmaps[:num_summary_images,:,:,:HEATMAP_CHANNEL], 40.*image_slice[:num_summary_images,:,:,:HEATMAP_CHANNEL] ], tf.float32 )


            images_with_est_affinity = tf.concat( [heatmaps[:num_summary_images,:,:,HEATMAP_CHANNEL:], zero_plane], axis = -1 )
            images_with_est_affinity = tf.image.resize_bilinear( images_with_est_affinity, (IMAGE_HEIGHT,IMAGE_WIDTH) )



            input_prediction = 255.0*tf.concat(
                [image_slice[:num_summary_images], images_with_gtjunctions[:num_summary_images], images_with_gt_affinity[:num_summary_images], images_with_est_junctions[:num_summary_images], images_with_est_affinity[:num_summary_images] ], axis=2)
            input_prediction = tf.clip_by_value( input_prediction, 0.0, 255.0 )

            input_prediction = tf.image.resize_bilinear( input_prediction, (256,256*5) )


            tf.summary.image("input_prediction gpu:%s"%(gpu_idx), input_prediction, max_outputs=num_summary_images)

    def add_summary_per_gpu_validation(self, gpu_idx, loss, image_slice, heatmap_slice, affinity_slice, heatmaps ):
        num_summary_images = 1 #self.FLAGS.num_summary_images
        # tf.summary.histogram("input_histogram_validation", image_slice)
        with tf.variable_scope( 'summary_validation_%s'%(gpu_idx) ):
            tf.summary.scalar("heatmap_loss_validation_%s_th_gpu"%(gpu_idx), loss )

            image_slice = ( ( image_slice ) / np.sqrt(2.0) + 0.5 )

            """
            depth_map = RTnet.convert_to_color_image( depth_map )
            gt_depth_map = RTnet.convert_to_color_image( gt_depth_map )

            images_with_lines = tf.py_func( draw_landmarks, \
                [ image_slice[:num_summary_images], landmarks_slice[:num_summary_images] ], tf.float32 )
            """

            images_with_gtjunctions = tf.py_func( draw_junctions_32f, \
                [ heatmap_slice[:num_summary_images,:,:,:HEATMAP_CHANNEL], 40.*image_slice[:num_summary_images,:,:,:HEATMAP_CHANNEL]  ], tf.float32 )


            zero_plane = tf.zeros( shape=(num_summary_images, HEATMAP_HEIGHT, HEATMAP_WIDTH,1), dtype=tf.float32)
            images_with_gt_affinity = tf.concat( [affinity_slice[:num_summary_images], zero_plane], axis = -1 )
            images_with_gt_affinity = tf.image.resize_bilinear( images_with_gt_affinity, (IMAGE_HEIGHT,IMAGE_WIDTH) )



            images_with_est_junctions = tf.py_func( draw_junctions_32f, \
                [ heatmaps[:num_summary_images,:,:,:HEATMAP_CHANNEL], 40.*image_slice[:num_summary_images,:,:,:HEATMAP_CHANNEL] ], tf.float32 )


            images_with_est_affinity = tf.concat( [heatmaps[:num_summary_images,:,:,HEATMAP_CHANNEL:], zero_plane], axis = -1 )
            images_with_est_affinity = tf.image.resize_bilinear( images_with_est_affinity, (IMAGE_HEIGHT,IMAGE_WIDTH) )



            input_prediction = 255.0*tf.concat(
                [image_slice[:num_summary_images], images_with_gtjunctions[:num_summary_images], images_with_gt_affinity[:num_summary_images], images_with_est_junctions[:num_summary_images], images_with_est_affinity[:num_summary_images] ], axis=2)
            input_prediction = tf.clip_by_value( input_prediction, 0.0, 255.0 )

            input_prediction = tf.image.resize_bilinear( input_prediction, (256,256*5) )


            tf.summary.image("input_prediction_validation gpu:%s"%(gpu_idx), input_prediction, max_outputs=num_summary_images)

    def add_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.FLAGS.keep)
        if self.FLAGS.save_best:
            self.best_saver = tf.train.Saver(max_to_keep=self.FLAGS.keep)

    def print_var_status(self,var_list):
        print("Following tensors will be updated:")
        sum_tensor_size = 0
        for v in var_list:
            cur_tensor_size = auxiliary_ftns.tensor_size(v)
            print(f"{v.name} with the size of {cur_tensor_size}")
            sum_tensor_size += cur_tensor_size
        print(f"total size = {sum_tensor_size} ({sum_tensor_size})")

    def add_gradient(self ):
        print("add gradient")
        with tf.variable_scope("gradients"):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            self.global_step = tf.Variable( 0, trainable=False)

            self.learning_rate = tf.train.exponential_decay(
                learning_rate = self.FLAGS.learning_rate,
                global_step = self.global_step,
                decay_steps = self.FLAGS.num_samples_per_learning_rate_half_decay / self.FLAGS.batch_size,
                decay_rate = 0.5)


            optimizer = tf.train.AdamOptimizer(self.learning_rate)

            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.minimize(self.loss,global_step=self.global_step )


    def add_summary(self):
        with tf.variable_scope('summary'):
            tf.summary.scalar("loss", self.loss)
            self.summaries = tf.summary.merge_all()


    def occasional_jobs( self, sess, global_step, save=False):
        ckpt_filename = os.path.join( self.FLAGS.train_dir, "myckpt")

        # if not self.FLAGS.save_best:
        if (global_step % self.FLAGS.save_every == 0 ) or save:
                save_path = self.saver.save(sess, ckpt_filename, global_step=global_step)
                tqdm.write( "saved at" + save_path )

        if global_step % self.FLAGS.eval_every == 0:
            eval_loss = self.evaluate_validation_loss(sess, global_step)
            print("evaluation loss:", eval_loss)
            summary = tf.Summary()
            with tf.variable_scope("validation"):
                summary.value.add(tag="validation_loss",simple_value=eval_loss)
                self.writer.add_summary(summary, global_step=global_step)

            if self.FLAGS.save_best:
                if self.min_val_loss > eval_loss:
                    self.min_val_loss = eval_loss
                    best_ckpt_filename = os.path.join(self.FLAGS.train_dir, "bestckpt")
                    save_path = self.best_saver.save(sess, best_ckpt_filename, global_step=global_step)
                    tqdm.write( "saved at " + save_path)


        # write examples to the examples directory
        """
        if  global_step % self.FLAGS.save_examples_every == 0:
            print("save examples - nothing done")
        """


    def train(self, sess):
        self.writer = tf.summary.FileWriter( self.FLAGS.train_dir, sess.graph)

        exp_loss = None
        counter = 0

        print("train starting")

        print_every = 1000
        save = False
        try:
            while True:
                for iter in tqdm( range( print_every ), leave=False ):

                    output_feed = {
                        "train_op": self.train_op,
                        "global_step": self.global_step,
                        "learning_rate": self.learning_rate,
                        "loss": self.loss
                    }

                    if iter % self.FLAGS.summary_every == 0:
                        output_feed["summaries"] = self.summaries

                    _results = sess.run( output_feed )


                    global_step = _results["global_step"]
                    learning_rate  = _results["learning_rate"]

                    if iter % self.FLAGS.summary_every == 0:
                        self.writer.add_summary( _results["summaries"], global_step=global_step )

                    cur_loss = _results["loss"]

                    if not exp_loss:  # first iter
                        exp_loss = cur_loss
                    else:
                        exp_loss = 0.99 * exp_loss + 0.01 * cur_loss

                    self.occasional_jobs( sess, global_step )

                if True:
                    print( f"global_step = {global_step}, learning_rate = {learning_rate:.6f}")
                    print( f"loss = {exp_loss:0.4f}" )
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            save = True
        finally:
            self.occasional_jobs(sess, global_step, save=save)

            sys.stdout.flush()



