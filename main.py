# -*- coding: utf-8 -*-
import os
import threading
import time, multiprocessing
from collections import deque
import gym
import tensorflow as tf
from RL.Chief import Chief
from RL.Worker import Worker

if __name__ == "__main__":
    parameter_dict = {
        'GAME': 'CartPole-v0',
        'LEARNING_RATE': 2e-5,
        'ENTCOEFF': 0.01,
        'VCOEFF': 0.5,
        'CLIP_PARAM': 0.2,
        'GAMMA': 0.99,
        'LAM': 0.95,
        'SCHEDULE': 'linear',
        'MAX_AC_EXP_RATE': 0.4,
        'MIN_AC_EXP_RATE': 0.15,
        'AC_EXP_PERCENTAGE': 1,
        'UPDATE_STEPS': 4,
        'MAX_EPOCH_STEPS': 100,
        'EPOCH_MAX': 500,
        'NUM_WORKERS': multiprocessing.cpu_count(),
        'LOG_FILE_PATH': '/log'
    }

    SESS = tf.Session()
    COORD = tf.train.Coordinator()

    PUSH_EVENT, UPDATE_EVENT = threading.Event(), threading.Event()
    PUSH_EVENT.clear()
    UPDATE_EVENT.clear()

    MEMORY_DICT = {}
    workers = []
    for i in range(parameter_dict['NUM_WORKERS']):
        i_name = 'Worker_N%i' % i
        MEMORY_DICT[i_name] = deque()
        workers.append(Worker(i_name, parameter_dict, SESS, MEMORY_DICT, COORD))
    chief = Chief('Chief', parameter_dict, SESS, MEMORY_DICT, COORD, workers)
    SESS.run(tf.global_variables_initializer())
    log_writer = tf.summary.FileWriter(parameter_dict['LOG_FILE_PATH'], graph=tf.get_default_graph())

    for worker in workers:
        SESS.run([localp.assign(chiefp) for chiefp, localp in zip(chief.ppo.pipara, worker.ppo.pipara)])
        SESS.run([localp.assign(chiefp) for chiefp, localp in zip(chief.ppo.oldpipara, worker.ppo.oldpipara)])

    start_time = time.time()
    threads = []
    for worker in workers:
        job = lambda : worker.work(PUSH_EVENT, UPDATE_EVENT, log_writer)
        t = threading.Thread(target=job)
        t.start()
        threads.append(t)
    PUSH_EVENT.set()
    threads.append(threading.Thread(target=chief.check(PUSH_EVENT, UPDATE_EVENT)))
    threads[-1].start()
    COORD.join(threads)

    print "TRAINING FINISHED."
    print "Train time elapsed:", time.time() - start_time, "seconds"

    env = gym.make(parameter_dict['GAME'])
    s = env.reset()
    epr = 0
    while True:
        env.render()
        a = chief.act(s)
        s_, r, done, _ = env.step(a)
        epr += r
        s = s_
        if done:
            print epr
            state = env.reset()
            epr = 0