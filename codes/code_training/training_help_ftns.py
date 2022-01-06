import tensorflow as tf 
import os 
#############################################################################
# Initialization                                                             
#############################################################################


def initialize_model(sess, 
                     model, 
                     train_dir, 
                     expect_exists=False, 
                     init_op = True):

    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""

    init_op = tf.group( [tf.global_variables_initializer(), tf.local_variables_initializer()] )
    pathname = None
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        print(f"Reading model parameters from {ckpt.model_checkpoint_path}")

        if init_op is True:
            sess.run( init_op )
        model.saver.restore(sess, ckpt.model_checkpoint_path)
        
        pathname = os.path.basename(ckpt.model_checkpoint_path)
        return ckpt.model_checkpoint_path, pathname  

    else:
        if expect_exists:
            raise Exception(f"There is no saved checkpoint at {train_dir}")
        else:
            print(f"There is no saved checkpoint at {train_dir}. Creating model with fresh parameters.")
            sess.run( init_op )

        return None, pathname 
    

