import os
import sys

import gymnasium as gym
from stable_baselines3.dqn.dqn import DQN

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci

from sumo_rl import SumoEnvironment

if __name__ == "__main__":
    env = SumoEnvironment(
        #net_file="nets/osm.net.xml",
        #route_file="nets/flow.xml",
        net_file="nets/RESCO/cologne3.net.xml",
        route_file="nets/RESCO/cologne3.rou.xml",
        out_csv_name="outputs/CollegeStation/dqn",
        single_agent=True,
        use_gui=True,
        min_green=5,
        max_green=60,
        begin_time=25000.0,
        num_seconds=100000,
    )
    
    model = DQN(
    env=env,
    policy="MlpPolicy",
    learning_rate=1e-3,
    learning_starts=0,
    buffer_size=50000,
    train_freq=1,
    target_update_interval=50,
    exploration_fraction=0.05,
    exploration_final_eps=0.01,
    verbose=1,
    )
    model.learn(total_timesteps=100000)
    