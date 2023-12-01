import numpy as np

def wait_norm(signals):
    rewards = dict()
    for signal_id in signals:
        total_wait = 0
        for lane in signals[signal_id].lanes:
            total_wait += signals[signal_id].full_observation[lane]['total_wait']

        rewards[signal_id] = np.clip(-total_wait/224, -4, 4).astype(np.float32)
    return rewards