# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Example impala agent for interacting with DeepMind Fast Mapping Tasks."""

from absl import app
from absl import flags
from absl import logging
from dm_env import specs
import dm_fast_mapping
import numpy as np

import threading
from typing import List
from examples.impala import actor as actor_lib
from examples.impala import agent as agent_lib
from examples.impala import haiku_nets
from examples.impala import learner as learner_lib
from examples.impala import util
import jax
import optax

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'docker_image_name', None,
    'Name of the Docker image that contains the Fast Language Learning Tasks. '
    'If None, uses the default dm_fast_mapping name')

flags.DEFINE_integer('seed', 123, 'Environment seed.')
flags.DEFINE_string('level_name', 'architecture_comparison/fast_map_three_objs',
                    'Name of task to run.')


class RandomAgent(object):
  """Basic random agent for DeepMind Fast Language Fast Language Learning Tasks."""

  def __init__(self, action_spec):
    self.action_spec = action_spec

  def act(self):
    action = {}

    for name, spec in self.action_spec.items():
      # Uniformly sample BoundedArray actions.
      if isinstance(spec, specs.BoundedArray):
        action[name] = np.random.uniform(spec.minimum, spec.maximum, spec.shape)
      else:
        action[name] = spec.generate_value()
    return action

ACTION_REPEAT = 1
BATCH_SIZE = 2
DISCOUNT_FACTOR = 0.99
MAX_ENV_FRAMES = 20000
NUM_ACTORS = 2
UNROLL_LENGTH = 20

FRAMES_PER_ITER = ACTION_REPEAT * BATCH_SIZE * UNROLL_LENGTH

def run_actor(actor: actor_lib.Actor, stop_signal: List[bool]):
  """Runs an actor to produce num_trajectories trajectories."""
  while not stop_signal[0]:
    frame_count, params = actor.pull_params()
    actor.unroll_and_push(frame_count, params)

def main(_):
  env_settings = dm_fast_mapping.EnvironmentSettings(
      seed=FLAGS.seed, level_name=FLAGS.level_name)
  with dm_fast_mapping.load_from_docker(
      name=FLAGS.docker_image_name, settings=env_settings) as env:
    # agent = RandomAgent(env.action_spec())
    # Create the networks to optimize (online) and target networks.
    num_actions = len(env.action_spec().values())
    agent = agent_lib.Agent(num_actions, env.observation_spec(),
                            haiku_nets.CatchNet)  # TODO: CatchNetの代わりにこの環境に適したネットワークを実装する

    max_updates = MAX_ENV_FRAMES / FRAMES_PER_ITER
    opt = optax.rmsprop(5e-3, decay=0.99, eps=1e-7)

    # Construct the agent.
    learner = learner_lib.Learner(
      agent,
      jax.random.PRNGKey(428),
      opt,
      BATCH_SIZE,
      DISCOUNT_FACTOR,
      FRAMES_PER_ITER,
      max_abs_reward=1.,
      logger=util.AbslLogger(),  # Provide your own logger here.
  )

# Construct the actors on different threads.
  # stop_signal in a list so the reference is shared.
  actor_threads = []
  stop_signal = [False]
  for i in range(NUM_ACTORS):
    actor = actor_lib.Actor(
        agent,
        env,
        UNROLL_LENGTH,
        learner,
        rng_seed=i,
        logger=util.AbslLogger(),  # Provide your own logger here.
    )
    args = (actor, stop_signal)
    actor_threads.append(threading.Thread(target=run_actor, args=args))

  # Start the actors and learner.
  for t in actor_threads:
    t.start()
  learner.run(int(max_updates))

  # Stop.
  stop_signal[0] = True
  for t in actor_threads:
    t.join()

    # timestep = env.reset()
    # score = 0
    # while not timestep.last():
    #   action = agent.act()
    #   timestep = env.step(action)

    #   if timestep.reward:
    #     score += timestep.reward
    #     logging.info('Total score: %1.1f, reward: %1.1f', score,
    #                  timestep.reward)


if __name__ == '__main__':
  app.run(main)
