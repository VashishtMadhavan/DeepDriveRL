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
	if not os.path.exists(args.demonstration_dir):
		os.mkdir(args.demonstration_dir)

    args.demonstrations_file = args.demonstration_dir + "demonstrations.pkl"

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


def collect(env, session, args,
	replay_buffer_size=100000,
	batch_size=32,
	frame_history_len=4):

	# SETTING UP INPUT
	if args.task == "DuskDrive":
		img_h, img_w, img_c = (128, 128, 3)
	elif args.task == "Torcs":
        img_h, img_w, img_c = (64, 64, 3)
    elif args.task == "Torcs_novision":
        img_h, img_w, img_c = (1, 1, 70)
	input_shape = (img_h, img_w, frame_history_len * img_c)


	# SETTING UP ACTIONS
	if args.task == "DuskDrive":
        actions = env.action_space.actions
        num_actions = len(actions)
    elif args.task == "Torcs":
        actions = [-1, 0, 1]
        num_actions = 3
    elif args.task == "Torcs_novision":
        actions = [-1, 0, 1]
        num_actions = 3

    obs_t_ph = tf.placeholder(tf.uint8, [None] + list(input_shape))
    act_t_ph = tf.placeholder(tf.int32, [None])
    rew_t_ph = tf.placeholder(tf.float32, [None])
    obs_tp1_ph = tf.placeholder(tf.uint8, [None] + list(input_shape)) # action at next timestep
    done_mask_ph = tf.placeholder(tf.float32, [None]) # 0 if next state is end of episode

    obs_t_float   = tf.cast(obs_t_ph,   tf.float32) / 255.0
    obs_tp1_float = tf.cast(obs_tp1_ph, tf.float32) / 255.0

    q_net = args.q_func(obs_t_float, num_actions, scope='q_func', reuse=False)

    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)
    saver = tf.train.Saver()
    if args.task == 'DuskDrive':
        last_obs = env.reset()
    else:
        last_obs = env.reset(relaunch=True)

    if args.weights:
    	saver.restore(session, args.weights)
    else:
    	raise NotImplementedError("Please load weights..")

    if args.render:
    	env.render()

    expert_dem = []; expert_act = []
    episode_dem = []; episode_actions = []

    for t in range(args.max_iters):

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
        obs_idx = replay_buffer.store_frame(down_samp)
        encoded_obs = replay_buffer.encode_recent_observation()
        encoded_obs = encoded_obs[np.newaxis, ...]
        q_net_eval = session.run(q_net, feed_dict={obs_t_ph: encoded_obs})
        action_idx = np.argmax(q_net_eval)

        episode_dem.append(encoded_obs); episode_actions.append(action_idx)

        last_obs, reward, done, info = env.step([actions[action_idx]])
        last_reward = reward[0] if args.task == "DuskDrive" else reward
        replay_buffer.store_effect(obs_idx, action_idx, last_reward, done)

        last_done = done[0] if args.task == "DuskDrive" else done

        if last_done:
            if args.task == "DuskDrive":
                last_obs = env.reset()
            else:
                last_obs = env.reset(relaunch=True)
            expert_dem.append(episode_dem)
            expert_act.append(episode_actions)
            episode_dem = []; episode_actions = []

    print("Done recording observations....")
    ret_dict = {}
    ret_dict['obs'] = np.squeeze(np.array(expert_dem))
    ret_dict['actions'] = np.array(dem_actions)
    pickle.dump(ret_dict, open(args.demonstrations_file,'w'))


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
    parser.add_argument('--model', type=str, default="BaseDQN", help="type of network model for the Q network")
    parser.add_argument('--demonstration_dir', type=str, default="demonstrations_full/", help="where to store the collected demonstrations")
    parser.add_argument('--task', type=str, choices=['DuskDrive', 'Torcs', 'Torcs_novision'], default="DuskDrive")
    parser.add_argument('--weights', type=str, default="output_full/weights/model_7000000.ckpt", help="path to model weights")
    parser.add_argument('--max_iters', type=int, default=20000, help='number of demonstrations to collect')
    parser.add_argument('--render', action='store_true', help='If true, will call env.render()')
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    main(args)