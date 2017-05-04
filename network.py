import tensorflow as tf
import tensorflow.contrib.layers as layers

"""
Network definitions for various DQN/RL models

* Create new models like the one in dqn_base: same input parameters
"""

# DQN as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
def dqn_base(img_in, num_actions, scope, reuse=False, regularize=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        if regularize:
            collect_var = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.REGULARIZATION_LOSSES]
        else:
            collect_var = [tf.GraphKeys.GLOBAL_VARIABLES]
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu, variables_collections=collect_var)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu, variables_collections=collect_var)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu, variables_collections=collect_var)
        out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=512,         activation_fn=tf.nn.relu, variables_collections=collect_var)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None, variables_collections=collect_var)
        return out


def dqn_fullyconnected(tensor_in, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = tensor_in
        out = layers.flatten(out)
        out = layers.fully_connected(out, num_outputs=1024, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=2048, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=1024, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out

