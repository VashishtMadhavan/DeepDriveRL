import os,argparse,sys, random
import gym
from gym import wrappers
import pickle
from collections import namedtuple
import universe
from universe.wrappers import Vision, Logger, Monitor
from universe.wrappers.experimental import CropObservations, SafeActionSpace
import numpy as np
import tensorflow as tf
from scipy.misc import imresize
from network import *
from dqn_utils import *

"""

Runs a specified DQN and uses it to collect expert demonstrations for imitation learning

"""
def setup(env, args):
   if args.task == "DuskDrive":
       env = Logger(env)
       env = Monitor(env, monitor_dir, force=True)
       env = Vision(env)
       env = CropObservations(env)
       env = SafeActionSpace(env)

   if args.model == "BaseDQN":
       args.q_func = dqn_base
   elif args.model == "DQNFullyConnected":
       args.q_func = dqn_fullyconnected
   else:
   	   raise NotImplementedError("Add support for different Q network models")
   return env

def task_input_shape(task):
    if task == "DuskDrive":
        return (128, 128, 3)
    elif task == "Torcs":
        return (64, 64, 3)
    elif task == "Torcs_novision":
        return (1, 1, 70)

def task_actions(task, env):
    if task == "DuskDrive":
        actions = env.action_space.actions
        num_actions = len(actions)
    elif task == "Torcs":
        actions = [-1, 0, 1]
        num_actions = 3
    elif task == "Torcs_novision":
        actions = [-1, 0, 1]
        num_actions = 3
    return actions, num_actions

def collect(env, session, args, replay_buffer_size=100000, frame_history_len=4):
    img_h, img_w, img_c = task_input_shape(args.task)
    input_shape = (img_h, img_w, img_c)
    actions, num_actions = task_actions(args.task, env)

    obs_t_ph = tf.placeholder(tf.uint8, [None] + list(input_shape))
    act_t_ph = tf.placeholder(tf.int32, [None])
    rew_t_ph = tf.placeholder(tf.float32, [None])
    obs_tp1_ph = tf.placeholder(tf.uint8, [None] + list(input_shape)) # action at next timestep
    done_mask_ph = tf.placeholder(tf.float32, [None]) # 0 if next state is end of episode

    obs_t_float   = tf.cast(obs_t_ph,   tf.float32) / 255.0
    obs_tp1_float = tf.cast(obs_tp1_ph, tf.float32) / 255.0

    q_net = args.q_func(obs_t_float, num_actions, scope='q_func', reuse=False)
    saver = tf.train.Saver()
    saver.restore(session, args.weights)

    if args.task == 'DuskDrive':
        last_obs = env.reset()
    else:
        last_obs = env.reset(relaunch=True)

    expert_observations = []; expert_actions = []; expert_rewards = []
    expert_done = []

    while len(expert_observations) < args.max_demonstrations:
    	if args.render:
    		env.render()

    	if args.task == "DuskDrive":
            last_obs_image = last_obs[0]
        else:
            #last_obs_image = last_obs.img
            last_obs_image = concat_obs(last_obs)

        if last_obs_image is None:
            last_obs, reward, done, info = env.step([actions[0]])
            if args.task == 'Torcs_novision':
                reward = reward_from_obs(last_obs, reward, done)
            continue

        if args.task == "DuskDrive":
            down_samp = imresize(last_obs_image, (128, 128, 3))
        elif args.task == "Torcs":
            down_samp = last_obs_image.reshape(64, 64, 3)
        elif args.task == "Torcs_novision":
            down_samp = last_obs_image.reshape(1, 1, -1)

        # getting action from donwsampled image
        encoded_obs = down_samp[np.newaxis, ...]
        q_net_eval = session.run(q_net, feed_dict={obs_t_ph: encoded_obs})
        action_idx = np.argmax(q_net_eval)

        last_obs, reward, done, info = env.step([actions[action_idx]])
        last_reward = reward[0] if args.task == "DuskDrive" else reward
        last_done = done[0] if args.task == "DuskDrive" else done

        expert_observations.append(down_samp)
        expert_actions.append(action_idx)
        expert_rewards.append(last_reward)
        expert_done.append(last_done)

        if last_done:
            if args.task == "DuskDrive":
                last_obs = env.reset()
            else:
                last_obs = env.reset(relaunch=True)

    with h5py.File(args.demonstrations_file, 'w', libver='latest') as f:
        n = len(expert_observations)
        input_shape = np.array(expert_observations)[0].shape
        task = f.create_dataset('task', data=np.string_(args.task))
        o = f.create_dataset('obs', (n, ) + input_shape, dtype='uint8', compression='lzf', data=np.squeeze(np.array(expert_observations)))
        a = f.create_dataset('actions', (n, ), dtype='uint8', data=np.array(expert_actions))
        r = f.create_dataset('rewards', (n, ), dtype='uint8', data=np.array(expert_rewards))
        d = f.create_dataset('done', (n, ), dtype='uint8', data=np.array(expert_done))


def main(args):
    args.seed = 0
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.task == "DuskDrive":
        env = gym.make('flashgames.DuskDrive-v0')
        env.configure(remotes=1)

    elif args.task == "Torcs":
        from gym_torcs import TorcsEnv
        env = TorcsEnv(vision=True, throttle=False)

    elif args.task == "Torcs_novision":
        from gym_torcs import TorcsEnv
        env = TorcsEnv(vision=False, throttle=False)
        
    sess = get_session(str(args.gpu))
    env = setup(env, args)
    collect(env, sess, args)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=4, help='gpu id')
    parser.add_argument('--demonstrations_file', default="dqn_demonstrations.h5", help="DQN demonstrations")
    parser.add_argument('--model', type=str, default="BaseDQN", help="type of network model for the Q network")
    parser.add_argument('--task', type=str, choices=['DuskDrive', 'Torcs', 'Torcs_novision'], default="DuskDrive")
    parser.add_argument('--weights', type=str, default="output_full/weights/model_7000000.ckpt", help="path to model weights")
    parser.add_argument('--max_demonstrations', type=int, default=10000, help="max number of demonstrations to collect")
    parser.add_argument('--render', action='store_true', help='If true, will call env.render()')
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    main(args)