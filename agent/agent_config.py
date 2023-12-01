import .rewards as rewards
import .states as states

from .agent.DQN import DQNIndependentAgent

agent_config = {
    'IDQN': {
        'agent': DQNIndependentAgent,
        'state': states.drq_norm,
        'reward': rewards.wait_norm,
        'max_distance': 200,
        'BATCH_SIZE': 32,
        'GAMMA': 0.99,
        'EPS_START': 1.0,
        'EPS_END': 0.0,
        'EPS_DECAY': 220,
        'TARGET_UPDATE': 500
    },
}