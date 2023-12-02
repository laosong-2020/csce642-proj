import os
import multiprocessing as mp
import sys
import argparse
from tqdm import tqdm
import torch
import random
import numpy as np

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from env.env import Env
from env.map_config import map_configs
from states import drq_norm
from rewards import wait_norm
from agent.DQN import DQN, ReplayBuffer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", type=str, default="cologne1")
    ap.add_argument("--pwd", type=str, default=os.path.dirname(__file__))
    ap.add_argument("--log_dir", type=str, default=os.path.join(os.path.dirname(os.getcwd()), 'results' + os.sep))
    ap.add_argument("--gui", type=bool, default=False)
    ap.add_argument("--libsumo", type=bool, default=False)
    args = ap.parse_args()

    # libsumo and multi-process
    if args.libsumo and 'LIBSUMO_AS_TRACI' not in os.environ:
        raise EnvironmentError("Set LIBSUMO_AS_TRACI to nonempty value to enable libsumo")

    env = make_env(args)

    lr = 1e-3
    num_episodes = 200
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 2
    buffer_size = 10000
    minimal_size = 500
    batch_size = 32
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    
    #seed
    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)
    #DQN Agent
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = env.observation_space[0].shape
    action_dim = env.action_space[0].n
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device)
    #train
    return_list = []
    with tqdm(total=int(num_episodes), desc='Iteration: ') as pbar:
        for i_ep in range(int(num_episodes)):
            episode_return = 0
            state = env.reset()
            
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward

                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    agent.update(transition_dict)

            return_list.append(episode_return)
            if (i_ep + 1) %target_update == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / target_update + i_ep + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-target_update:])
                })
            pbar.update(1)
            

def make_env(args):

    # map config
    map_config = map_configs[args.map]
    num_steps_eps = int(
        (map_config['end_time'] - map_config['start_time']) / map_config['step_length']
    )
    
    route = map_config['route']
    if route is not None:
        route = os.path.join(args.pwd, route)

    # env
    env = Env(
        run_name="DQN",
        map_name=args.map,
        net=os.path.join(args.pwd, map_config['net']),
        state_fn=drq_norm,
        reward_fn=wait_norm,
        route=route,
        step_length=map_config['step_length'],
        yellow_length=map_config['yellow_length'],
        step_ratio=map_config['step_ratio'], 
        end_time=map_config['end_time'],
        max_distance=200, 
        lights=map_config['lights'], 
        gui=args.gui,
        log_dir=args.log_dir, 
        libsumo=args.libsumo, 
        warmup=map_config['warmup']
    )

    return env


if __name__ == "__main__":
    main()