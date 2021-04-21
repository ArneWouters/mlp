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

"""Example use of the NeuRD algorithm on Kuhn Poker.

This NeuRD implementation does not use an entropy bonus to ensure that the
current joint policy approaches an equilibrium in zero-sum games, but it
tracks the exact tabular average so that the average policy approaches an
equilibrium (assuming the policy networks train well).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf
import numpy as np

from open_spiel.python.algorithms import neurd, rcfr
import pyspiel
import time

tf.enable_eager_execution()

FLAGS = flags.FLAGS

flags.DEFINE_integer("iterations", 1001, "Number of iterations")
flags.DEFINE_string("game", "universal_poker", "Name of the game")
flags.DEFINE_integer("players", 2, "Number of players")
flags.DEFINE_integer("print_freq", 20, "How often to print the exploitability")
flags.DEFINE_integer("num_hidden_layers", 1,
                     "The number of hidden layers in the policy model.")
flags.DEFINE_integer("num_hidden_units", 13,
                     "The number of hidden layers in the policy model.")
flags.DEFINE_integer(
    "num_hidden_factors", 8,
    "The number of factors in each hidden layer in the policy model.")
flags.DEFINE_boolean(
    "use_skip_connections", True,
    "Whether or not to use skip connections in the policy model.")
flags.DEFINE_integer("batch_size", 100, "The policy model training batch size.")
flags.DEFINE_float(
    "threshold", 2.,
    "Logits of the policy model will be discouraged from growing beyond "
    "`threshold`.")
flags.DEFINE_float("step_size", 1.0, "Policy model step size.")
flags.DEFINE_boolean(
    "autoencode", False,
    "Whether or not to augment the policy model with outputs that attempt to "
    "reproduce the model inputs. The policy model is updated online so "
    "training with the reproduction error as an auxiliary task helps to keep "
    "the model stable in the absence of an entropy bonus.")


def export_layers(game, models):
    timestamp = round(time.time())

    for i in range(len(models)):
        model = models[i]
        keras_model = tf.keras.Sequential(
            [tf.keras.Input(shape=(rcfr.num_features(game),))] + model.layers
        )
        keras_model.save("output/exported_model" + str(i) + "_" + str(timestamp) + ".h5", save_format="h5")


def import_layers(game, models):
    reconstructed_models = [
        tf.keras.models.load_model("aaaaaaaaa.h5", compile=False),
        tf.keras.models.load_model("bbbbbbbbb.h5", compile=False)
    ]

    for i in range(len(models)):
        m = models[i]
        imported_layers = reconstructed_models[i].layers
        for j in range(len(m.layers)):
            m.layers[j].set_weights(imported_layers[j].get_weights())

        # Construct variables for all layers by exercising the network.
        x = tf.zeros([1, rcfr.num_features(game)])
        for layer in m.layers:
            x = layer(x)


def main(_):
    game = pyspiel.load_game(FLAGS.game,
                             {"numPlayers": pyspiel.GameParameter(FLAGS.players)})

    models = []
    for _ in range(game.num_players()):
        models.append(
            neurd.DeepNeurdModel(
                game,
                num_hidden_layers=FLAGS.num_hidden_layers,
                num_hidden_units=FLAGS.num_hidden_units,
                num_hidden_factors=FLAGS.num_hidden_factors,
                use_skip_connections=FLAGS.use_skip_connections,
                autoencode=FLAGS.autoencode))

    # import_layers(game, models)
    solver = neurd.CounterfactualNeurdSolver(game, models)
    expl = np.array([])

    def _train(model, data):
        neurd.train(
            model,
            data,
            batch_size=FLAGS.batch_size,
            step_size=FLAGS.step_size,
            threshold=FLAGS.threshold,
            autoencoder_loss=(tf.compat.v1.losses.huber_loss
                              if FLAGS.autoencode else None))

    for i in range(FLAGS.iterations):
        solver.evaluate_and_update_policy(_train)
        logging.info("Iteration %s", i)
        if i % FLAGS.print_freq == 0:
            export_layers(game, models)
            conv = pyspiel.exploitability(game, solver.average_policy())
            expl = np.append(expl, conv)
            logging.info("Iteration {} exploitability {}".format(i, conv))
            np.savetxt("output/fcpa_expl.csv", expl, delimiter=",")


if __name__ == "__main__":
    app.run(main)
