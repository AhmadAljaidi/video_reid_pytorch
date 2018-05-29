from __future__ import print_function
import os
import pprint
import tensorflow as tf
from human_reid_learner import reid_Learner

# server directory /home/amohamma/UOIT/Human_re_id/iLIDS-VID/i-LIDS-VID/sequences
# laptop directory C:/Users/k-any/Uoit/Human-Re-id/Dataset/iLIDS-VID/i-LIDS-VID/sequences
flags = tf.app.flags
flags.DEFINE_string("directory", "C:/Users/k-any/Uoit/Human-Re-id/Dataset/iLIDS-VID/i-LIDS-VID/sequences", "Directory to the dataset")
flags.DEFINE_string("opt_flow_dir", "./optical_flow_dir", "Directory to the optical flow dataset")
flags.DEFINE_string("train_test_split_dir", "./dataset", "Directory to the train/test dataset")
flags.DEFINE_string("dataset_name", "dataset_1_train1.txt", "Train dataset name")
flags.DEFINE_string("exp_name", "exp_1", "Experiment name")
flags.DEFINE_string("checkpoint_dir", "./checkpoints", "Directory name to save the checkpoints")
flags.DEFINE_string("logs_path", "logs", "Tensorboard log path")
flags.DEFINE_bool("continue_train", False, "Resume training")
flags.DEFINE_string("init_checkpoint_file", "hnRiD_98000", "checkpoint file")
flags.DEFINE_bool("use_opt_flow", False, "Use optical flow")
flags.DEFINE_integer("batch_size", 1, "The size of of a sample batch")
flags.DEFINE_integer("image_width",  48, "Image width, must same as prepare_dataset")
flags.DEFINE_integer("image_height", 64, "Image height, must same as prepare_dataset")
flags.DEFINE_integer("hidden_size", 128, "Hidden units in RNN")
flags.DEFINE_integer("sequence_length", 16, "RNN time steps")
flags.DEFINE_integer("nPersons", 2, "Total number of people")
flags.DEFINE_integer("margin", 2, "Hinge loss margin")
flags.DEFINE_integer("start_step", 0,    "Starting training step")
flags.DEFINE_integer("max_steps", 500, "Maximum number of epochs")
flags.DEFINE_string("pooling", 'temporal', "max or temporal")
flags.DEFINE_float("drop", 0.6,    " Dropout probability")
flags.DEFINE_float("l2", 0.0005,   " Weight Decay")
flags.DEFINE_float("l_rate", 0.0001, " learning rate")
flags.DEFINE_float("clip_grad", 5.0, "Magnitude of clip on the RNN gradient")
flags.DEFINE_integer("reduce_l_rate_itr", 5000, "Reduce learning rate after this many iterations")
flags.DEFINE_integer("summary_freq", 20, "Logging every summary_freq iterations")
flags.DEFINE_integer("valid_freq",   25, "Logging every valid_freq epoch")
flags.DEFINE_integer("save_latest_freq", 25, \
                       "Save the latest model every save_latest_freq epoch")
FLAGS = flags.FLAGS

def main(_):
    pp = pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    hrl = reid_Learner()
    hrl.train(FLAGS)


if __name__ == '__main__':
    tf.app.run()
