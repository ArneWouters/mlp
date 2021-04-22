# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Python Deep CFR example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import six

import tensorflow.compat.v1 as tf
import numpy as np

from open_spiel.python import policy
from open_spiel.python.algorithms import deep_cfr
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import exploitability
from open_spiel.python.bots import uniform_random
import pyspiel

# Temporarily disable TF2 behavior until we update the code.
tf.disable_v2_behavior()

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_episodes", 10000, "Number of training episodes")
flags.DEFINE_integer("num_iterations", 50, "Number of iterations")
flags.DEFINE_integer("num_traversals", 40, "Number of traversals/games")
flags.DEFINE_string("game_name", "universal_poker", "Name of the game")


def export_model(sess):
    saver = tf.train.Saver()
    save_path = saver.save(sess, "tmp/model.ckpt")
    logging.info("Model saved in path: %s" % save_path)


def restore_model(sess):
    saver = tf.train.Saver()
    saver.restore(sess, "tmp/model.ckpt")
    logging.info("Model restored.")


def main(unused_argv):
    logging.info("Loading %s", FLAGS.game_name)
    game = pyspiel.load_game(FLAGS.game_name)
    with tf.Session() as sess:
        deep_cfr_solver = deep_cfr.DeepCFRSolver(
            sess,
            game,
            policy_network_layers=(16,),
            advantage_network_layers=(16,),
            num_iterations=FLAGS.num_iterations,
            num_traversals=FLAGS.num_traversals,
            learning_rate=1e-3,
            batch_size_advantage=128,
            batch_size_strategy=1024,
            memory_capacity=1e7,
            policy_network_train_steps=400,
            advantage_network_train_steps=20,
            reinitialize_advantage_networks=False)
        sess.run(tf.global_variables_initializer())

        for i in range(FLAGS.num_episodes):
            logging.info("Iteration: {}".format(i))
            deep_cfr_solver.solve()
            export_model(sess)


if __name__ == "__main__":
    app.run(main)
