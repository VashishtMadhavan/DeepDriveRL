import os,argparse,sys, random
import gym
import itertools
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
Script for naive pre-training of DQN with demonstrations.
After that regular Q-Learning is run 
"""

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

def load_demonstrations(dem_file):
    data = h5py.File(open(dem_file))
    obs = data['obs']
    actions = data['actions']
    rewards = data['rewards']
    done = data['done']
    return obs, actions, rewards, done

def setup(env, args):
	args.lr_schedule = PiecewiseSchedule([
    (0, 1e-4),
    (args.dqn_iters / 10, 1e-4),
    (args.dqn_iters / 2,  5e-5),
    ], outside_value=5e-5)

    args.optimizer = OptimizerSpec(
    constructor=tf.train.AdamOptimizer,
    kwargs=dict(epsilon=1e-4),
    lr_schedule=args.lr_schedule
    )

    # setting up dir to store monitor output
    monitor_dir = os.path.join(args.output_dir, "monitor")
    if not os.path.exists(monitor_dir):
        os.mkdir(monitor_dir)

    # setting up dir to store model weights
    args.snapshot_dir = os.path.join(args.output_dir, "weights")
    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)

    args.log_file = args.output_dir + "train.log"

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

    args.exploration_schedule = PiecewiseSchedule(
    [
        (0, 1.0),
        (1e6, 0.1),
        (args.dqn_iters / 2, 0.01),
        ], outside_value=0.01
    )
    return env

def concat_obs(obs):
    cat = np.array([])
    for ob in obs:
        cat = np.concatenate((cat, ob.reshape(-1)), 0)
    #ob = obs.angle
    #cat = np.concatenate((cat, ob.reshape(-1)), 0)
    #ob = obs.trackPos
    #cat = np.concatenate((cat, ob.reshape(-1)), 0)
    return cat

def reward_from_obs(obs, reward, done):
    if done:
        return -100
    #return obs.speedX + sum(obs.track**2) - abs(obs.speedY) - abs(obs.speedZ)
    #print(obs.focus)
    #print(obs.track)
    #print(reward)
    return reward

def train(env, session, args,
	replay_buffer_size=100000,
	learning_freq=4,
	learning_starts=40000,
	batch_size=32,
	frame_skip=8,
    frame_history_len=4,
    grad_norm_clipping=10):

    img_h, img_w, img_c = task_input_shape(args.task)
    input_shape = (img_h, img_w, frame_history_len * img_c)
    
    actions, num_actions = task_actions(args.task, env)
    demo_obs, demo_act, demo_rew, demo_done = load_demonstrations(args.demonstrations)

    demo_buffer = ReplayBuffer(dem_obs.shape[0], frame_history_len)
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    obs_t_ph = tf.placeholder(tf.uint8, [None] + list(input_shape))
    act_t_ph = tf.placeholder(tf.int32, [None])
    rew_t_ph = tf.placeholder(tf.float32, [None])
    obs_tp1_ph = tf.placeholder(tf.uint8, [None] + list(input_shape))
    done_mask_ph = tf.placeholder(tf.float32, [None])

    obs_t_float   = tf.cast(obs_t_ph,   tf.float32) / 255.0
    obs_tp1_float = tf.cast(obs_tp1_ph, tf.float32) / 255.0

    # Double Q Learning Dynamics
    q_net = args.q_func(obs_t_float, num_actions, scope='q_func', reuse=False, regularize=True)
    q_net_tp1 = args.q_func(obs_tp1_float, num_actions, scope='q_func', reuse=True, regularize=True)
    q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func') 

    target_q_net= args.q_func(obs_tp1_float, num_actions, scope='tq_func', reuse=True)
    target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='tq_func')

    actions_mat = tf.one_hot(tf.argmax(q_net, axis=1), num_actions)
    q_val = tf.reduce_max(q_net * actions_mat, axis=1)
    
    q_actions_mat = tf.one_hot(tf.argmax(q_net_tp1, axis=1), num_actions)
    target_q_val = rew_t_ph + gamma * tf.reduce_max(target_q_net * q_actions_mat, axis=1) * done_mask_ph

    loss_q = tf.reduce_mean(tf.square(target_q_val - q_val))
    loss_l = (1. - tf.one_hot(act_t_ph, num_actions)) * 0.8
    loss_margin = tf.reduce_max(q_net + loss_l, axis=1) - q_val

    pretrain_optim = tf.train.AdamOptimizer(args.lr).minimize(loss_margin)
    pretrain_fn = minimize_and_clip(pretrain_optim, loss_margin, var_list=q_func_vars, clip_val=grad_norm_clipping)

    lr = tf.placeholder(tf.float32, (), name="learn_rate")
    optim = args.optimizer.constructor(learning_rate=lr, **args.optimizer.kwargs)
    train_fn = minimize_and_clip(optim, loss_q, var_list=q_func_vars, clip_val=grad_norm_clipping)

    update_target_fn = []
    for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name), 
                            sorted(target_q_func_vars, key=lambda v: v.name)):
        update_target_fn.append(var_target.assign(var))
    update_target_fn = tf.group(*update_target_fn)

    model_initailized = False
    mean_episode_reward      = -float('nan')
    best_mean_episode_reward = -float('inf')
    total_episode_rewards = []
    episode_reward = 0.
    num_param_updates = 0
    log_steps = 10000
    if args.task == "DuskDrive":
        last_obs = env.reset()
    else:
        last_obs = env.reset(relaunch=True)
    saver = tf.train.Saver()

    if args.weights:
    	saver.restore(session, args.weights)
    	model_initailized = True

    # Step 1: Add demonstrations to buffer
    for idx in range(demo_obs.shape[0]):
        scale_down = imresize(demo_obs[idx], (img_w, img_h, img_c))
        demo_obs_idx = demo_buffer.store_fame(scale_down)
        demo_buffer.store_effect(demo_obs_idx, demo_act[idx], demo_rew[idx], demo_done[idx])

    # Step 2: Pretrain with Imitation Learning
    for _ in range(args.pretrain_iters):
    	obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = demo_buffer.sample(batch_size)
        if not model_initailized:
            initialize_interdependent_variables(session, tf.global_variables(), 
                        {obs_t_ph: obs_batch, obs_tp1_ph: next_obs_batch})
            model_initailized = True

        train_dict ={
            obs_t_ph: obs_batch,
            act_t_ph: act_batch,
            rew_t_ph: rew_batch,
            obs_tp1_ph: next_obs_batch,
            done_mask_ph: done_mask
        }
        session.run(pretrain_fn, feed_dict=train_dict)

    # Step 3: Double DQN Training
    with open(args.log_file,'w') as lfp:
        lfp.write("Timestep, Mean Reward, Best Reward\n")

    for t in itertools.count():
        print("Iteration: %s" % str(t))
    	if args.render:
    		env.render()
    		
    	if t % args.save_period == 0:
    		saver.save(session, os.path.join(args.snapshot_dir, "model_%s.ckpt" %(str(t))))

    	if t >= args.dqn_iters:
    		break

        if args.task == "DuskDrive":
            last_obs_image = last_obs[0]
        else:
            last_obs_image = concat_obs(last_obs)

    	if last_obs_image is None:
    		last_obs, reward, done, info = env.step([actions[0]])
            if args.task == "Torcs_novision":
                reward = reward_from_obs(last_obs, reward, done)
    		continue

    	down_samp = imresize(last_obs[0], (img_w, img_h, img_c))
    	obs_idx = replay_buffer.store_frame(down_samp)
    	eps = args.exploration_schedule.value(t)

    	if np.random.rand(1) < eps:
    		action_idx = np.random.choice(num_actions)
        else:
        	encoded_obs = replay_buffer.encode_recent_observation()[np.newaxis,...]
        	q_eval = session.run(q_net, feed_dict={obs_t_ph: encoded_obs})
        	action_idx = np.argmax(q_eval)

        last_reward = 0.
        for _ in range(frame_skip):
        	last_obs, reward, done, info = env.step([actions[action_idx]])
        	last_reward += reward[0] if args.task == "DuskDrive" else reward 

        episode_reward += last_reward
        replay_buffer.store_effect(obs_idx, action_idx, last_reward, done)
        last_done = done[0] if args.task == "DuskDrive" else done

        if last_done:
            if args.task == "DuskDrive":
                last_obs = env.reset()
            else:
                last_obs = env.reset(relaunch=True)
            total_episode_rewards.append(episode_reward)
            episode_reward = 0.

        # Performing Experience Replay by sampling from the ReplayBuffer
        # and training the network with some random exploration criterion
        if t > learning_starts and t % learning_freq == 0 and replay_buffer.can_sample(batch_size):
            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size)
            train_dict = {obs_t_ph: obs_batch,
                        act_t_ph: act_batch,
                        rew_t_ph: rew_batch,
                        obs_tp1_ph: next_obs_batch,
                        done_mask_ph: done_mask,
                        lr: args.optimizer.lr_schedule.value(t)
                        }

            session.run(train_fn, feed_dict=train_dict)
            num_param_updates += 1

            if num_param_updates % target_update_freq == 0:
                session.run(update_target_fn)

            # Logging
            if len(total_episode_rewards) > 0:
                mean_episode_reward = np.mean(np.array(total_episode_rewards)[-100:])
            if len(total_episode_rewards) > 10:
                best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
            if t % log_steps == 0 and model_initialized and best_mean_episode_reward != float('-inf'):
                with open(args.log_file,'a') as lfp:
                    lfp.write("%s,%s,%s\n" %(str(t), str(mean_episode_reward), str(best_mean_episode_reward)))


def get_session(gpu_id):
	os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
	tf.reset_default_graph()
	tf_config = tf.ConfigProto(
		inter_op_parallelism_threads=1,
		intra_op_parallelism_threads=1)
	session = tf.Session(config=tf_config)
	return session


def main(args):
	args.seed = 0
	random.seed(args.seed)
	np.random.seed(args.seed)

	if not os.path.exists(args.output_dir):
		os.mkdir(args.output_dir)

	if args.task == "DuskDrive":
		env = gym.make('flashgames.DuskDrive-v0')
		env.configure(remotes=1)

	elif args.task == "Torcs":
		raise NotImplementedError()

	sess = get_session(args.gpu)
	env = setup(env, args)
	train(env, sess, args)

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', type=int, default=0, help='gpu id')
	parser.add_argument('--demonstrations', type=str, help="Data file with observations and actions")
	parser.add_argument('--model', type=str, default="BaseDQN", help="type of network model for the Q network")
	parser.add_argument('--output_dir', type=str, default="output_imit/", help="where to store all misc. training output")
	parser.add_argument('--task', type=str, choices=['DuskDrive', 'Torcs'], default="DuskDrive")
	parser.add_argument('--lr', type=float, default=0.001, help='learning rate for pretraining')
	parser.add_argument('--save_period', type=int, default=1e6, help="snapshot iters")
	parser.add_argument('--weights', type=str, default=None, help="path to model weights")
	parser.add_argument('--dqn_iters', type=int, default=10e6, help='number of timesteps to run DQN')
    parser.add_argument('--pretrain_iters', type=int, default=2000, help="number of iterations to pretrain network")
	parser.add_argument('--render', action='store_true', help='If true, will call env.render()')
	return parser.parse_args()

if __name__=="__main__":
	args = parse_args()
	main(args)
