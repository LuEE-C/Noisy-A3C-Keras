# Initial implementation from https://jaromiru.com/2017/02/16/lets-make-an-a3c-theory/
# Reworked with strong inspiration from https://github.com/Kaixhin/NoisyNet-A3C


import numpy as np

import tensorflow as tf

import gym, time, random, multiprocessing, threading

from keras.models import Model
from keras.layers.advanced_activations import PReLU
from keras.layers import Dense, Input, Dropout, GRU, BatchNormalization
from keras import backend as K
from NoisyDense import NoisyDense

# -- constants
ENV = 'CartPole-v1'

# FRAMES_RUN_TIME = 200000
RENDER = False
RUN_TIME = 180
THREADS = 8
OPTIMIZERS = 3
THREAD_DELAY = 0.001
SIGMA_INIT=0.02

GAMMA = 0.99

N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN

MIN_BATCH = 32

LOSS_V = .5  # v loss coefficient
LOSS_ENTROPY = .01  # entropy coefficient


# ---------
class Brain:
    train_queue = [[], [], [], [], []]  # s, a, r, s', s' terminal mask
    lock_queue = threading.Lock()

    def __init__(self):
        self.session = tf.Session()
        K.set_session(self.session)
        K.set_learning_phase(1)
        K.manual_variable_initialization(True)

        self.model = self._build_model()
        self.graph = self._build_graph(self.model)

        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()

        self.default_graph.finalize()  # avoid modifications

    def _build_model(self):

        l_input = Input(batch_shape=(None, NUM_STATE))
        x = Dense(128)(l_input)
        x = PReLU()(x)
        x = Dense(128)(x)
        x = PReLU()(x)
        out_actions = NoisyDense(NUM_ACTIONS, activation='softmax', name='out_actions', sigma_init=SIGMA_INIT)(x)
        out_value = NoisyDense(1, name='out_value', sigma_init=SIGMA_INIT)(x)

        model = Model(inputs=[l_input], outputs=[out_actions, out_value])
        model._make_predict_function()  # have to initialize before threading

        return model

    def _build_graph(self, model):
        s_t = tf.placeholder(tf.float32, shape=(None, NUM_STATE))
        a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
        r_t = tf.placeholder(tf.float32, shape=(None, 1))  # not immediate, but discounted n step reward

        p, v = model(s_t)

        log_prob = tf.log(tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10)
        advantage = r_t - v

        loss_policy = - log_prob * tf.stop_gradient(advantage)  # maximize policy
        loss_value = LOSS_V * tf.square(advantage)  # minimize value error
        entropy = LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keep_dims=True)  # maximize entropy (regularization)

        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=.99)
        minimize = optimizer.minimize(loss_total)

        return s_t, a_t, r_t, minimize

    def optimize(self):
        if len(self.train_queue[0]) < MIN_BATCH:
            time.sleep(0)  # yield
            return

        with self.lock_queue:
            if len(self.train_queue[0]) < MIN_BATCH:  # more thread could have passed without lock
                return  # we can't yield inside lock

            s, a, r, s_, s_mask = self.train_queue
            self.train_queue = [[], [], [], [], []]

        s = np.vstack(s)
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.vstack(s_)
        s_mask = np.vstack(s_mask)

        if len(s) > 5 * MIN_BATCH: print("Optimizer alert! Minimizing batch of %d" % len(s))

        v = self.predict_v(s_)
        r = r + GAMMA_N * v * s_mask  # set v to 0 where s_ is terminal state
        self.model.get_layer('out_actions').sample_noise()
        self.model.get_layer('out_value').sample_noise()
        # print(self.model.get_layer('out_value').get_weights()[1])
        s_t, a_t, r_t, minimize = self.graph
        self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: r})

    def print_average_weight(self):
        print(np.mean(self.model.get_layer('out_actions').get_weights()[1]))

    def train_push(self, s, a, r, s_):
        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)

            if s_ is None:
                self.train_queue[3].append(NONE_STATE)
                self.train_queue[4].append(0.)
            else:
                self.train_queue[3].append(s_)
                self.train_queue[4].append(1.)

    def predict(self, s):
        with self.default_graph.as_default():
            #self.model.get_layer('out_actions').sample_noise()
            p, v = self.model.predict(s)
            return p, v

    def predict_p(self, s):
        with self.default_graph.as_default():
            #self.model.get_layer('out_actions').sample_noise()
            p, v = self.model.predict(s)
            return p

    def predict_v(self, s):
        with self.default_graph.as_default():
            #self.model.get_layer('out_actions').sample_noise()
            p, v = self.model.predict(s)
            return v


# ---------
frames = 0


class Agent:
    def __init__(self):

        self.memory = []  # used for n_step return
        self.R = 0.


    def act(self, s):
        global frames;
        frames = frames + 1
        s = np.array([s])
        p = brain.predict_p(s)[0]

        # a = np.argmax(p)
        a = np.random.choice(NUM_ACTIONS, p=p)

        return a

    def train(self, s, a, r, s_):
        def get_sample(memory, n):
            s, a, _, _ = memory[0]
            _, _, _, s_ = memory[n - 1]

            return s, a, self.R, s_

        a_cats = np.zeros(NUM_ACTIONS)  # turn action into one-hot representation
        a_cats[a] = 1

        self.memory.append((s, a_cats, r, s_))

        self.R = (self.R + r * GAMMA_N) / GAMMA

        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                brain.train_push(s, a, r, s_)

                self.R = (self.R - self.memory[0][2]) / GAMMA
                self.memory.pop(0)

            self.R = 0

        if len(self.memory) >= N_STEP_RETURN:
            s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
            brain.train_push(s, a, r, s_)

            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)

            # possible edge case - if an episode ends in <N steps, the computation is incorrect


# ---------
class Environment(threading.Thread):
    stop_signal = False

    def __init__(self, render=False):
        threading.Thread.__init__(self)

        self.render = render
        self.env = gym.make(ENV)
        self.agent = Agent()

    def runEpisode(self):
        s = self.env.reset()

        R = 0
        while True:
            time.sleep(THREAD_DELAY)  # yield

            if self.render: self.env.render()

            a = self.agent.act(s)
            s_, r, done, info = self.env.step(a)

            if done:  # terminal state
                s_ = None

            self.agent.train(s, a, r, s_)

            s = s_
            R += r

            if done or self.stop_signal:
                break

        print("Total R:", R)

    def run(self):
        while not self.stop_signal:
            self.runEpisode()

    def stop(self):
        self.stop_signal = True


# ---------
class Optimizer(threading.Thread):
    stop_signal = False

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while not self.stop_signal:
            brain.optimize()

    def stop(self):
        self.stop_signal = True


if __name__ == '__main__':
    # -- main
    env_test = Environment(render=RENDER)
    NUM_STATE = env_test.env.observation_space.shape[0]
    NUM_ACTIONS = env_test.env.action_space
    print(env_test.env.action_space, env_test.env.observation_space)
    print(NUM_ACTIONS, NUM_STATE)
    NUM_ACTIONS = 2
    NONE_STATE = np.zeros(NUM_STATE)

    brain = Brain()  # brain is global in A3C

    envs = [Environment() for i in range(THREADS)]
    opts = [Optimizer() for i in range(OPTIMIZERS)]

    for o in opts:
        o.start()

    for e in envs:
        e.start()

    time.sleep(RUN_TIME)

    for e in envs:
        e.stop()
    for e in envs:
        e.join()

    for o in opts:
        o.stop()
    for o in opts:
        o.join()

    print("Training finished")
    env_test.run()