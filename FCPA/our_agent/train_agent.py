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
import tensorflow as tf

from open_spiel.python.algorithms import deep_cfr_tf2
import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_iterations", 25, "Number of iterations")
flags.DEFINE_integer("num_traversals", 100, "Number of traversals/games")


def main(unused_argv):
    fcpa_game_string = pyspiel.hunl_game_string("fcpa")
    game = pyspiel.load_game(fcpa_game_string)

    deep_cfr_solver = deep_cfr_tf2.DeepCFRSolver(
        game,
        policy_network_layers=(64, 128, 128, 64),
        advantage_network_layers=(64, 64, 64, 64),
        num_iterations=FLAGS.num_iterations,
        num_traversals=FLAGS.num_traversals,
        learning_rate=1e-3,
        batch_size_advantage=2048,
        batch_size_strategy=2048,
        memory_capacity=1e4,
        policy_network_train_steps=500,
        advantage_network_train_steps=100,
        reinitialize_advantage_networks=True,
        infer_device="gpu",
        train_device="gpu")

    # load model
    deep_cfr_solver._policy_network = tf.keras.models.load_model("saved_model/my_model2")

    for i in range(300):
        logging.info("Iteration: {}".format(i))
        deep_cfr_solver.solve()
        deep_cfr_solver.save_policy_network("saved_model/my_model2")


if __name__ == "__main__":
    app.run(main)
