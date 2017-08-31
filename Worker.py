# -*- coding: utf-8 -*-
from collections import deque
import gym
import numpy as np
import random
from RL.PPO import PPO

class Worker(object):
    def __init__(self, scope, parameter_dict, SESS, MEMORY_DICT, COORD):
        self.env = gym.make(parameter_dict['GAME'])
        self.ppo = PPO(scope, parameter_dict, self.env)
        self.COORD = COORD
        self.MEMORY_DICT = MEMORY_DICT
        self.name = scope
        self.sess = SESS
        self.CUR_EP = 0
        self.EPOCH_MAX = parameter_dict['EPOCH_MAX']
        self.MAX_EPOCH_STEPS = parameter_dict['MAX_EPOCH_STEPS']
        self.MAX_AC_EXP_RATE = parameter_dict['MAX_AC_EXP_RATE']
        self.MIN_AC_EXP_RATE = parameter_dict['MIN_AC_EXP_RATE']

        self.AC_EXP_EPOCH = parameter_dict['AC_EXP_PERCENTAGE'] * parameter_dict['EPOCH_MAX']
        self.SCHEDULE = parameter_dict['SCHEDULE']
        self.GAMMA = parameter_dict['GAMMA']
        self.LAM = parameter_dict['LAM']
        self.ENV_SAMPLE_ITERATIONS = parameter_dict['ENV_SAMPLE_ITERATIONS']
        self.LOG_FILE_PATH = parameter_dict['LOG_FILE_PATH']

    def work(self, PUSH_EVENT, UPDATE_EVENT, log_writer):
        while not self.COORD.should_stop():

            buffer_s, buffer_a, buffer_r = deque(), deque(), deque()
            buffer_predv, buffer_done, buffer_epr = deque(), deque(), deque()

            s = self.env.reset()
            EPR = 0
            index = 0

            while index < self.MAX_EPOCH_STEPS:
                if not PUSH_EVENT.is_set():
                    PUSH_EVENT.wait()
                    self.sess.run(self.ppo.sync_pis)
                    buffer_s, buffer_a, buffer_r = deque(), deque(), deque()
                    buffer_predv, buffer_done, buffer_epr = deque(), deque(), deque()
                    s = self.env.reset()
                    EPR = 0
                    index = 0
                else:
                    action, pred_v = self.act(s)
                    s_, r, done, _ = self.env.step(action)

                    buffer_s.append(s)
                    buffer_a.append(action)
                    buffer_r.append(r)
                    buffer_predv.append(pred_v)
                    buffer_done.append(done)

                    s = s_
                    EPR += r
                    index += 1

                    if done:
                        buffer_epr.append(EPR)
                        s = self.env.reset()
                        EPR = 0
            self.CUR_EP += 1

            print self.name, "finished iterator", self.CUR_EP, "\n"

            buffer_s = list(buffer_s)
            buffer_a = list(buffer_a)
            buffer_r = list(buffer_r)
            buffer_predv = list(buffer_predv)
            buffer_done = list(buffer_done)
            buffer_epr = list(buffer_epr)

            if self.SCHEDULE == 'constant':
                self.current_learningRate = 1.0
            elif self.SCHEDULE == 'linear':
                self.current_learningRate = max(1.0 - float(self.CUR_EP) / self.EPOCH_MAX, 0)

            buffer_done = np.append(buffer_done, 0)
            buffer_predv_tmp = np.append(buffer_predv, pred_v * (1 - done))
            T = len(buffer_r)
            buffer_adv = np.empty(T, 'float32')
            lastgaelam = 0
            for t in reversed(range(T)):
                nonterminal = 1 - buffer_done[t + 1]
                delta = buffer_r[t] + self.GAMMA * buffer_predv_tmp[t + 1] * nonterminal - buffer_predv_tmp[t]
                buffer_adv[t] = lastgaelam = delta + self.GAMMA * self.LAM * nonterminal * lastgaelam
            buffer_etr = np.add(buffer_adv.tolist(), buffer_predv)
            buffer_adv = (buffer_adv - buffer_adv.mean()) / buffer_adv.std()

            buffer_s, buffer_a = np.vstack(buffer_s), np.vstack(buffer_a)
            buffer_etr, buffer_adv = np.vstack(buffer_etr), np.vstack(buffer_adv)

            batchs = deque()
            batchs.append(buffer_s)
            batchs.append(buffer_a)
            batchs.append(buffer_adv)
            batchs.append(buffer_etr)
            batchs.append(self.current_learningRate)

            feed_dict = {
                self.ppo.s: buffer_s,
                self.ppo.action: buffer_a,
                self.ppo.advantage: buffer_adv,
                self.ppo.estimatedReturn: buffer_etr,
                self.ppo.l_mul: self.current_learningRate,
            }

            if (self.name == 'Worker_N0'):
                summary = self.sess.run(self.ppo.summary_op, feed_dict)
                log_writer.add_summary(summary, self.CUR_EP)


            queryItem = [self.ppo.policyLoss, self.ppo.valueLoss, self.ppo.entropyLoss, self.ppo.total_loss]
            policyLoss, valueLoss, entropyLoss, totalLoss = self.sess.run(queryItem, feed_dict)

            buffer_epr = np.array(buffer_epr)
            score = buffer_epr.mean() / buffer_epr.std()

            logs = deque()
            logs.append(score)
            logs.append(buffer_epr.min())
            logs.append(buffer_epr.max())
            logs.append(buffer_epr.mean())
            logs.append(policyLoss)
            logs.append(valueLoss)
            logs.append(entropyLoss)
            logs.append(totalLoss)
            logs.append(self.CUR_EP)
            batchs.append(list(logs))

            if (len(buffer_epr) > 0):
                self.MEMORY_DICT[self.name].append(list(batchs))

            UPDATE_EVENT.set()

    def act(self, s):
        if (self.CUR_EP >= self.AC_EXP_EPOCH):
            cur_exp_rate = self.MIN_AC_EXP_RATE
        else:
            cur_exp_rate = self.MAX_AC_EXP_RATE + self.CUR_EP * (self.MIN_AC_EXP_RATE - self.MAX_AC_EXP_RATE) / self.AC_EXP_EPOCH

        action, pred_v = self.sess.run([self.ppo.ca, self.ppo.pipredv], {self.ppo.s: [s]})

        action_space = self.env.action_space
        if random.random() < cur_exp_rate:
            self.returnAction = random.randint(0, action_space.n - 1)
        else:
            self.returnAction = action[0][0]
        return self.returnAction, pred_v[0][0][0]