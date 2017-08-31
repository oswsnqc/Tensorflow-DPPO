# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from RL.Model import Model
from RL.Others import tf_util as U

class PPO(object):
    def __init__(self, scope, parameter_dict, env, workerLists = None):
        LEARNING_RATE = parameter_dict['LEARNING_RATE']
        CLIP_PARAM = parameter_dict['CLIP_PARAM']
        ENTCOEFF = parameter_dict['ENTCOEFF']
        VCOEFF = parameter_dict['VCOEFF']
        NUM_WORKERS = parameter_dict['NUM_WORKERS']
        observation_space = env.observation_space
        action_space = env.action_space
        model = Model()
        self.s = tf.placeholder(tf.float32, [None, observation_space.shape[0]])
        self.l_mul = tf.placeholder(name='learningRate_multiplier', dtype=tf.float32, shape=[])
        CLIP_PARAM = CLIP_PARAM * self.l_mul
        Optimizer = tf.train.AdamOptimizer(LEARNING_RATE * self.l_mul)
        self.pipredv, piPD, self.pipara = model.FC(scope=scope + 'pi', state=self.s, action_space=action_space)
        oldpredv, oldpiPD, self.oldpipara = model.FC(scope=scope + 'oldpi', state=self.s, action_space=action_space)
        self.ca = piPD.sample()

        if scope != "Chief":
            self.a = tf.placeholder(tf.int32, [None, 1])
            self.adv = tf.placeholder(tf.float32, [None, 1])
            self.estimatedReturn = tf.placeholder(tf.float32, [None, 1])
            kloldnew = oldpiPD.kl(piPD)
            ent = piPD.entropy()
            meanent = U.mean(ent)
            ratio = tf.exp(piPD.logp(self.a) - oldpiPD.logp(self.a))
            surr1 = ratio * self.adv
            surr2 = U.clip(ratio, 1.0 - CLIP_PARAM, 1.0 + CLIP_PARAM) * self.adv
            self.policyLoss = - U.mean(tf.minimum(surr1, surr2))
            self.entropyLoss = (-ENTCOEFF) * meanent
            vfloss1 = tf.square(self.pipredv - self.estimatedReturn)
            vpredclipped = oldpredv + tf.clip_by_value(self.pipredv - oldpredv, -CLIP_PARAM, CLIP_PARAM)
            vfloss2 = tf.square(vpredclipped - self.estimatedReturn)
            self.valueLoss = VCOEFF * U.mean(tf.maximum(vfloss1, vfloss2))
            self.total_loss = self.policyLoss + self.entropyLoss + self.valueLoss
            tf.summary.scalar("entropyLoss", self.entropyLoss)
            tf.summary.scalar("policyLoss", self.policyLoss)
            tf.summary.scalar("valueLoss", self.valueLoss)
            tf.summary.scalar("total_loss", self.total_loss)
            self.summary_op = tf.summary.merge_all()
            self.gradient = Optimizer.compute_gradients(self.total_loss, self.pipara)
            self.sync_pis = [oldp.assign(p) for p, oldp in zip(self.pipara, self.oldpipara)]
        else:
            gradientList = []
            for i in range(NUM_WORKERS):
                gradientList.append(workerLists[i].ppo.gradient)
            averaged_gradients = self._average_gradients(gradientList)
            self.train = Optimizer.apply_gradients(zip(averaged_gradients, self.pipara))

    def _average_gradients(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            grad = tf.concat(grads, axis=0)
            grad = tf.reduce_mean(grad, 0)
            average_grads.append(grad)
        return average_grads