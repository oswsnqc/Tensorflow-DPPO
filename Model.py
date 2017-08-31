import tensorflow as tf
from RL.Others.distributions import make_pdtype
from RL.Others import tf_util as U


class Model(object):
    def FC(self, scope, s, action_space):
        with tf.variable_scope(scope):
            pdtype = make_pdtype(action_space)
            initializer = U.normc_initializer(0.01)
            s = tf.expand_dims(s, axis=1)
            l1 = tf.layers.dense(s, 16, tf.nn.relu, kernel_initializer=initializer)
            predv = tf.layers.dense(l1, 1, kernel_initializer=initializer)
            logits = tf.layers.dense(l1, pdtype.param_shape()[0], kernel_initializer=initializer)
            pd = pdtype.pdfromflat(logits)

        para = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        return predv, pd, para

