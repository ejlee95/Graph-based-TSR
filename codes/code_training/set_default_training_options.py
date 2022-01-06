import tensorflow as tf

#############################################################################
# Initialization
#############################################################################

def get_flags(MAIN_DIR):

    tf.app.flags.DEFINE_integer('batch_size', 16,  'Minibatch size')
    tf.app.flags.DEFINE_integer('NUM_STACKS', 2,  'Number of hourglass stacks')

    tf.app.flags.DEFINE_integer("summary_every", 10,
                                "How many iterations to do per TensorBoard "
                                "summary write.")
    tf.app.flags.DEFINE_integer("save_examples_every", 1000,
                                "Sets how often to compute target predictions"
                                "and write them to the experiments directory.")
    tf.app.flags.DEFINE_integer("save_every", 1000,
                                "How many iterations to do per save.")
    tf.app.flags.DEFINE_integer("eval_every", 5000,
                                "How many iterations to do per calculating the "
                                "loss on the test set. This operation "
                                "is time-consuming, so should not be done often.")
    tf.app.flags.DEFINE_boolean('save_best', False, 'save ckpt if validation loss declined')

    tf.app.flags.DEFINE_integer(
            'num_preprocessing_processes', 4,
            'The number of processes to create batches.')


    tf.app.flags.DEFINE_integer(
        'num_samples_per_learning_rate_half_decay', 1600000,
        'Number of samples for half decaying learning rate')

    tf.app.flags.DEFINE_float(
        'learning_rate', 0.001,
        'Initial learning rate')



    # Control flags
    tf.app.flags.DEFINE_string("gpu", "0",
                                "Sets which GPU to use, if you have multiple.")
    tf.app.flags.DEFINE_string("mode", "test",
                            "Options: {train,eval,predict}.")
    tf.app.flags.DEFINE_string("experiment_name",  "default_name",  # "regression_result_317k",
                            "Creates a dir with this name in the output/ "
                            "directory, to which checkpoints and logs related "
                            "to this experiment will be saved.")
    tf.app.flags.DEFINE_integer("keep", 3,
                                "How many checkpoints to keep. None means keep "
                                "all. These files are storage-consuming so should "
                                "not be kept in aggregate.")
    tf.app.flags.DEFINE_integer("print_every", 500,
                                "How many iterations to do per print.")


    # TensorBoard
    tf.app.flags.DEFINE_integer("num_summary_images", 2,
                                "How many images to write to summary.")


    # Training

    tf.app.flags.DEFINE_integer("num_epochs", 1000,
                                "Sets the number of epochs to train. None means "
                                "train indefinitely.")
    tf.app.flags.DEFINE_string("train_dir", "",
                            "Sets the dir to which checkpoints and logs will "
                            "be saved. Defaults to "
                            "output/{experiment_name}.")

    tf.app.flags.DEFINE_string("bicubic", "on",
                            "bicubic interpolation")


    tf.app.flags.DEFINE_list("data_dir", "",
                            "dataset dir list")

    tf.app.flags.DEFINE_list("validation_dataset_file_path", "",
                            "validation tfrecords list")

    """
    tf.app.flags.DEFINE_float("lr_decay_every", 30000,
                            "Sets the intervals at which to do learning"
                            "rate decay (cuts by 1/2). Setting to 0 means"
                            "no decay")
    tf.app.flags.DEFINE_boolean("data_augmentation", True,
                                "Sets whether or not to perform data augmentation")
    """
    # Test
    tf.app.flags.DEFINE_string("test_data_dir", "/media/jane/D/ispl/Researches/Layout_analysis/table/data/ctdar19/ICDAR2019_cTDaR-master/test/TRACKB2/modern/bordered/img",
                               "testset dir path")
    tf.app.flags.DEFINE_string("test_name", "", "test result dir name")
    tf.app.flags.DEFINE_integer("niter", 2, "only for test, number of the iteration of the optimization")
    tf.app.flags.DEFINE_integer("nextra", 3, "only for test, number of the extra nodes")
    tf.app.flags.DEFINE_boolean("save_inter", False, "save intermediate results(True) or not(False)")
    tf.app.flags.DEFINE_boolean("test_scan", False, "True: scan, False: pdf-based image, to use different evaluation function")
    tf.app.flags.DEFINE_boolean("with_detection", False, "")

    return tf.app.flags.FLAGS
