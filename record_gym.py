import gym
import pygame
import pygame.locals as pl
import argparse
import numpy as np
import h5py

"""

Script to Record Gym demonstrations from human playing
Extension on ALE interface for recording


"""

keys = [pl.K_SPACE, pl.K_UP, pl.K_RIGHT, pl.K_LEFT, pl.K_DOWN]

mapping = {
    # dlruf
    0b00000: 0,
    0b00001: 1,
    0b00010: 2,
    0b00100: 3,
    0b01000: 4,
    0b10000: 5,
    0b00110: 6,
    0b01010: 7,
    0b10100: 8,
    0b11000: 9,
    0b00011: 10,
    0b00101: 11,
    0b01001: 12,
    0b10001: 13,
    0b00111: 14,
    0b01011: 15,
    0b10101: 16,
    0b11001: 17
}

def keystates_to_ale_action(keystates):
    if keystates[pl.K_UP] and keystates[pl.K_DOWN]:
        keystates[pl.K_UP] = False
        keystates[pl.K_DOWN] = False
    if keystates[pl.K_LEFT] and keystates[pl.K_RIGHT]:
        keystates[pl.K_LEFT] = False
        keystates[pl.K_RIGHT] = False
    bitvec = sum(2 ** i if keystates[key] else 0 for i, key in enumerate(keys))
    assert bitvec in mapping
    return mapping[bitvec]

def update_keystates(keystates):
    events = pygame.event.get()
    for event in events:
        if hasattr(event, 'key') and event.key == pl.K_ESCAPE:
            exit(0)
        if hasattr(event, 'key') and event.key in keys:
            if event.type == pygame.KEYDOWN:
                keystates[event.key] = True
            elif event.type == pygame.KEYUP:
                keystates[event.key] = False


def record(args):
    keystates = {key: False for key in keys}

    env = gym.make(args.env_name)
    legal_actions = range(env.action_space.n)
    ale_actions = list(env.unwrapped.ale.getMinimalActionSet())
    action_map = {ale_actions[q]: legal_actions[q] for q in range(len(ale_actions))}

    obs = env.reset()
    pygame.init()
    eps = 0

    rec_obs = []; rec_act = []; rec_rew = []

    while len(rec_obs) < args.num_observations:
        env.render()
        update_keystates(keystates)
        al_act = keystates_to_ale_action(keystates)
        if al_act in action_map:
            action = action_map[al_act]
            obs, reward, done, _ = env.step(action)
        else:
            action = action_map[0]
            obs, reward, done, _ = env.step(action)

        rec_obs.append(obs)
        rec_rew.append(reward)
        rec_act.append(action)
        
        if done:
            obs = env.reset()
            eps += 1
            if eps >= args.episodes:
                break

    with h5py.File(args.output_file, 'w', libver='latest') as f:
        n = len(rec_obs)
        input_shape = np.array(rec_obs)[0].shape
        action_set = f.create_dataset('action_set', (len(legal_actions),), dtype='uint8', data=np.array(legal_actions))
        o = f.create_dataset('obs', (n, ) + input_shape, dtype='uint8', compression='lzf', data=np.array(rec_obs))
        a = f.create_dataset('actions', (n, ), dtype='uint8', data=np.array(rec_act))
        r = f.create_dataset('rewards', (n, ), dtype='uint8', data=np.array(rec_rew))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default="PongNoFrameskip-v3", help="atari task")
    parser.add_argument('--output_file', type=str, default="demonstrations.h5", help="output file")
    parser.add_argument('--episodes', type=int, default=1, help="num episodes to record")
    parser.add_argument('--num_observations', type=int, default=100, help="number of demonstrations to extract")
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    record(args)



