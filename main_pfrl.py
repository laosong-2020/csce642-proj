import argparse
import os
import sys

#import gym
import numpy as np
import torch.optim as optim
from gym import spaces

import pfrl
from pfrl import experiments, explorers
from pfrl import nn as pnn
from pfrl import q_functions, replay_buffers, utils
from pfrl.agents.dqn import DQN

from sumo_rl import SumoEnvironment
def make_env(test=False):
    env = SumoEnvironment(
        net_file='nets/RESCO/cologne1.net.xml',
        route_file='nets/RESCO/cologne1.rou.xml',
        use_gui=False,
        begin_time=25200,
        num_seconds=3600,
        min_green=5,
        delta_time=5,
    )

    # seed = 0, sub_process seeds = [0]
    process_seeds = [0]
    seed = 0
    process_seed = 0
    env_seed = 2**32 - 1 - process_seed if test else process_seed
    utils.set_random_seed(env_seed)

    # Cast observations to float32 because our model uses float32
    env = pfrl.wrappers.CastObservationToFloat32(env)

    return env

def main():
    import logging

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--final-exploration-steps", type=int, default=10**4)
    parser.add_argument("--start-epsilon", type=float, default=1.0)
    parser.add_argument("--end-epsilon", type=float, default=0.1)
    parser.add_argument("--noisy-net-sigma", type=float, default=None)
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--steps", type=int, default=10**5)
    parser.add_argument("--prioritized-replay", action="store_true")
    parser.add_argument("--replay-start-size", type=int, default=1000)
    parser.add_argument("--target-update-interval", type=int, default=10**2)
    parser.add_argument("--target-update-method", type=str, default="hard")
    parser.add_argument("--soft-update-tau", type=float, default=1e-2)
    parser.add_argument("--update-interval", type=int, default=1)
    parser.add_argument("--eval-n-runs", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=10**4)
    parser.add_argument("--n-hidden-channels", type=int, default=100)
    parser.add_argument("--n-hidden-layers", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--render-train", action="store_true")
    parser.add_argument("--render-eval", action="store_true")
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--reward-scale-factor", type=float, default=1e-3)
    parser.add_argument(
        "--actor-learner",
        action="store_true",
        default=False,
        help="Enable asynchronous sampling with asynchronous actor(s)",
    )  # NOQA
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help=(
            "The number of environments for sampling (only effective with"
            " --actor-learner enabled)"
        ),
    )  # NOQA
    args = parser.parse_args()

    # Set a random seed used in PFRL
    utils.set_random_seed(0)

    args.outdir = experiments.prepare_output_dir(args, "results", argv=sys.argv)
    print("Output files are saved in {}".format(args.outdir))

    
    env = make_env(test=False)
    timestep_limit = 72
    obs_space = env.observation_space
    obs_size = obs_space.low.size
    action_space = env.action_space

    if isinstance(action_space, spaces.Box):
        action_size = action_space.low.size
        # Use NAF to apply DQN to continuous action spaces
        q_func = q_functions.FCQuadraticStateQFunction(
            obs_size,
            action_size,
            n_hidden_channels=args.n_hidden_channels,
            n_hidden_layers=args.n_hidden_layers,
            action_space=action_space,
        )
        # Use the Ornstein-Uhlenbeck process for exploration
        ou_sigma = (action_space.high - action_space.low) * 0.2
        explorer = explorers.AdditiveOU(sigma=ou_sigma)
    else:
        n_actions = action_space.n
        q_func = q_functions.FCStateQFunctionWithDiscreteAction(
            obs_size,
            n_actions,
            n_hidden_channels=args.n_hidden_channels,
            n_hidden_layers=args.n_hidden_layers,
        )
        # Use epsilon-greedy for exploration
        explorer = explorers.LinearDecayEpsilonGreedy(
            args.start_epsilon,
            args.end_epsilon,
            args.final_exploration_steps,
            action_space.sample,
        )

    opt = optim.Adam(q_func.parameters())

    rbuf_capacity=10000
    rbuf = replay_buffers.ReplayBuffer(rbuf_capacity)

    agent = DQN(
        q_func,
        opt,
        rbuf,
        gpu=args.gpu,
        gamma=args.gamma,
        explorer=explorer,
        replay_start_size=args.replay_start_size,
        target_update_interval=args.target_update_interval,
        update_interval=args.update_interval,
        minibatch_size=args.minibatch_size,
        target_update_method=args.target_update_method,
        soft_update_tau=args.soft_update_tau,
    )

    eval_env = make_env(test=True)

    experiments.train_agent_with_evaluation(
            agent=agent,
            env=env,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            eval_env=eval_env,
            train_max_episode_len=timestep_limit,
            eval_during_episode=True,
        )
    
if __name__ == "__main__":
    main()