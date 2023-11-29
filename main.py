import pathlib
import os
import multiprocessing as mp

import argparse

from .env.env import Env
from .env.signal_config import signal_config 
from .env.map_config import map_config
from .agent.agent_config import agent_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--agent", type=str, default="IDQN")
    ap.add_argument("--map", type=str, default="cologne3")
    ap.add_argument("--eps", type=int, default=100)
    ap.add_argument("--procs", type=int, default=1)

    ap.add_argument("--pwd", type=str, default=os.path.dirname(__file__))
    ap.add_argument("--log_dir", type=str, default=os.path.join(os.path.dirname(os.getcwd()), 'results' + os.sep))
    ap.add_argument("--gui", type=bool, default=False)
    ap.add_argument("--libsumo", type=bool, default=False)
    ap.add_argument("--tr", type=int, default=0)  # Can't multi-thread with libsumo, provide a trial number
    args = ap.parse_args()

    # libsumo and multi-process
    if args.libsumo and 'LIBSUMO_AS_TRACI' not in os.environ:
        raise EnvironmentError("Set LIBSUMO_AS_TRACI to nonempty value to enable libsumo")

    if args.procs == 1 or args.libsumo:
        run_trial(args, args.tr)
    else:
        pool = mp.Pool(processes=args.procs)
        for trial in range(1, args.trials+1):
            pool.apply_async(run_trial, args=(args, trial))
        pool.close()
        pool.join()

def run_trial(args, trial):
    # agent config
    agent_config = agent_config[args.agent]
    agent_map_config = agent_config.get(args.map)
    if agent_map_config is not None:
        agent_config = agent_map_config
    alg = agent_config['agent']

    # map config
    map_config = map_config[args.map]
    num_steps_eps = int(
        (map_config['end_time'] - map_config['start_time'] / map_config['step_length'])
    )
    route = map_config['route']
    if route is not None:
        route = os.path.join(args.pwd, route)

    # env
    env = Env(
        run_name=alg.__name__ + 'tr' + str(trial),
        map_name=args.map,
        net=os.path.join(args.pwd, map_config['net']),
        state_fn=agent_config['state'],
        reward_fn=agent_config['reward'],
        route=route,
        step_length=map_config['step_length'],
        yellow_length=map_config['yellow_length'],
        step_ratio=map_config['step_ratio'], 
        end_time=map_config['end_time'],
        max_distance=agent_config['max_distance'], 
        lights=map_config['lights'], 
        gui=args.gui,
        log_dir=args.log_dir, 
        libsumo=args.libsumo, 
        warmup=map_config['warmup']
    )

    agent_config['episodes'] = int(args.eps * 0.8)
    agent_config['steps'] = agent_config['episodes'] * num_steps_eps
    agent_config['log_dir'] = os.path.join(args.log_dir, env.connection_name)
    agent_config['num_lights'] = len(env.all_ts_ids)
    agent_config['save_freq'] = 10

    obs_act = dict()
    for key in env.obs_shape:
        obs_act[key] = [env.obs_shape[key], len(env.phases[key]) if key in env.phases else None]
    agent = alg(
        agent_config,
        obs_act,
        args.map, trial
    )

    for ep in range(args.eps):
        obs = env.reset()
        done = False
        while not done:
            act = agent.act(obs)
            obs, reward, done, info = env.step(act)
            agent.observe(obs, reward, done, info)
    env.close()

if __name__ == "__main__":
    main()