# -*- coding: utf-8 -*-
from collections import deque
import gym
import numpy as np
import sys
from RL.PPO import PPO

class Chief(object):
    def __init__(self, scope, parameter_dict, SESS, MEMORY_DICT, COORD, workers):
        env = gym.make(parameter_dict['GAME'])
        self.ppo = PPO(scope, parameter_dict, env, workers)
        self.sess = SESS
        self.MEMORY_DICT = MEMORY_DICT
        self.workers = workers
        self.UPDATE_STEPS = parameter_dict['UPDATE_STEPS']
        self.COORD = COORD
        self.NUM_WORKERS = parameter_dict['NUM_WORKERS']

    def check(self, PUSH_EVENT, UPDATE_EVENT):
        while not self.COORD.should_stop():
            UPDATE_EVENT.wait()
            min_data_size, _ = self._get_data_size()
            if min_data_size >= 1:
                PUSH_EVENT.clear()
                self._train()
                self._update_local_workers_weight()
                PUSH_EVENT.set()
            UPDATE_EVENT.clear()

    def _train(self):
        data_stack = deque()
        _, max_data_size= self._get_data_size()
        while max_data_size > 0:
            for key, value in self.MEMORY_DICT.items():
                if len(value) > 0:
                    value = list(value)
                    tmp = deque()
                    tmp.append(value[0][0])  # buffer_states
                    tmp.append(value[0][1])  # buffer_actions
                    tmp.append(value[0][2])  # buffer_advantage
                    tmp.append(value[0][3])  # buffer_estimatedReturn
                    tmp.append(value[0][4])  # current_learningRate
                    tmp.append(value[0][5])  # buffer_score

                    self.MEMORY_DICT[key].popleft()
                    tmp = list(tmp)
                    data_stack.append(tmp)
            _, max_data_size = self._get_data_size()

        data_stack = list(data_stack)
        data_stack = reversed(sorted(data_stack,key=lambda x: x[5][2]))
        data_stack = list(data_stack)

        feed_dict = {}

        learningRate_multiplier = data_stack[0][4]
        feed_dict[self.ppo.learningRate_multiplier] = learningRate_multiplier
        for i in range(self.NUM_WORKERS):
            feed_dict[self.workers[i].ppo.state] = data_stack[i][0]
            feed_dict[self.workers[i].ppo.action] = data_stack[i][1]
            feed_dict[self.workers[i].ppo.advantage] = data_stack[i][2]
            feed_dict[self.workers[i].ppo.estimatedReturn] = data_stack[i][3]
            feed_dict[self.workers[i].ppo.learningRate_multiplier] = learningRate_multiplier
        [self.sess.run(self.ppo.train, feed_dict=feed_dict) for _ in range(self.UPDATE_STEPS)]
        self._logs_writer(data_stack)

    def _update_local_workers_weight(self):
        for worker in self.workers:
            update_weight = [localp.assign(chiefp) for chiefp, localp in zip(self.ppo.piNetParameters, worker.ppo.piNetParameters)]
            self.sess.run(update_weight)

    def _get_data_size(self):
        min_data_size = sys.maxint
        max_data_size = -1
        for key, value in self.MEMORY_DICT.items():
            min_data_size = min(min_data_size, len(value))
            max_data_size = max(max_data_size, len(value))
        return min_data_size, max_data_size

    def _logs_writer(self, data_stack):
        logs = []
        for item in data_stack:
            logs.append(item[5])

        localsteps = logs[0][8]
        if localsteps >= 500:
            self.COORD.request_stop()

    def act(self, state):
        state = np.array([state])
        action = self.sess.run(self.ppo.ca, {self.ppo.state: state})
        return action[0][0]