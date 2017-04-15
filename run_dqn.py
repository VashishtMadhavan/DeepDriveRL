import os,argparse,sys, random
import gym
import itertools
from collections import namedtuple
import universe
from universe.wrappers import Vision, Logger, Monitor
from universe.wrappers.experimental import CropObservations, SafeActionSpace
import numpy as np
import tensorflow as tf
from network import *
from dqn_utils import *

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

"""
Setting up learning rate scheduler and exploration criterion
"""
def setup(env, args):
	args.lr_schedule = PiecewiseSchedule([
	(0,1e-4 * args.lr_mult),
	(args.max_iters / 10, 1e-4 * args.lr_mult),
	(args.max_iters / 2,  5e-5 * args.lr_mult),
	], outside_value=5e-5 * args.lr_mult)

	args.optimizer = OptimizerSpec(
	constructor=tf.train.AdamOptimizer,
	kwargs=dict(epsilon=1e-4),
	lr_schedule=args.lr_schedule
	)

	monitor_dir = os.path.join(args.output_dir, "monitor")
	if not os.path.exists(monitor_dir):
		os.mkdir(monitor_dir)

	env = Logger(env)
	env = Monitor(env, monitor_dir, force=True)
	env = Vision(env)
	env = CropObservations(env) #TODO: dont really know if this works see if you could keep it aournd
	env = SafeActionSpace(env)

	if args.model == "BaseDQN":
		args.q_func = dqn_base
	else:
		raise NotImplementedError("Add support for different Q network models")

	args.exploration_schedule = PiecewiseSchedule(
	[
		(0, 1.0),
		(1e6, 0.1),
		(args.max_iters / 2, 0.01),
		], outside_value=0.01
	)
	return env

#TODO: attach tensorBoard
#TODO: add support for saving weights
#TODO: add support for different Q networks
def train(env, session, args,
	replay_buffer_size=1000000,
	batch_size=32,
	gamma=0.99,
	learning_starts=50000,
	learning_freq=4,
	frame_history_len=4,
	target_update_freq=10000,
	grad_norm_clipping=10):

	# TODO: set up Torcs with same variable names as Dusk Drive and with similar format
	if args.task == "DuskDrive":
		img_h, img_w, img_c = (512, 800, 3)
		input_shape = (img_h, img_w, frame_history_len * img_c) # to account for sequence of frames
		actions = [[x] for x in env.action_space[0]]+ [[y] for y in env.action_space[1]] + [[z] for z in env.action_space[2]]
		num_actions = len(actions)
	elif args.task == "Torcs":
		raise NotImplementedError("Please implement Torcs Functionality...")

	# Placeholder Formatting
	obs_t_ph = tf.placeholder(tf.uint8, [None] + list(input_shape))
	act_t_ph = tf.placeholder(tf.int32, [None])
	rew_t_ph = tf.placeholder(tf.float32, [None])
	obs_tp1_ph = tf.placeholder(tf.uint8, [None] + list(input_shape)) # action at next timestep
	done_mask_ph = tf.placeholder(tf.float32, [None]) # 0 if next state is end of episode

	obs_t_float   = tf.cast(obs_t_ph,   tf.float32) / 255.0
	obs_tp1_float = tf.cast(obs_tp1_ph, tf.float32) / 255.0

	# Q learning dynamics
	actions_mat = tf.one_hot(act_t_ph, num_actions, off_value=0.0, on_value=1.0, axis=-1)
	q_net = args.q_func(obs_t_float, num_actions, scope='q_func', reuse=False)
	target_q_net= args.q_func(obs_tp1_float, num_actions, scope='tq_func', reuse=False)

	q_val = tf.reduce_sum(q_net * actions_mat, reduction_indices=1)
	target_q_val = rew_t_ph + gamma * tf.reduce_max(target_q_net, reduction_indices=1) * done_mask_ph
	error = tf.reduce_mean(tf.square(target_q_val - q_val))

	q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
	target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='tq_func')

	# Optimization parameters
	lr = tf.placeholder(tf.float32, (), name="learn_rate")
	opt = args.optimizer.constructor(learning_rate=lr, **args.optimizer.kwargs)
	train_fn = minimize_and_clip(opt, error, var_list=q_func_vars, clip_val=grad_norm_clipping)

	update_target_fn = []
	for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name), 
							sorted(target_q_func_vars, key=lambda v: v.name)):

		update_target_fn.append(var_target.assign(var))

	update_target_fn = tf.group(*update_target_fn)
	replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len) #TODO: check ReplayBuffer code

	###############
	# Run Env    #
	###############
	model_initialized = False
	num_param_updates = 0
	mean_episode_reward      = -float('nan')
	best_mean_episode_reward = -float('inf')
	episode_rewards = []
	log_steps = 10000

	last_obs = env.reset()
	log_file = "%s/%s" % (args.output_dir, args.log_file)

	lfp = open(log_file, 'w')
	lfp.write("Timestep\tMean Reward\tBest Mean Reward\tEpisodes\tExploration\tLearning Rate\n")
	lfp.close()

	for t in itertools.count():
		if t >= args.max_iters:
			break

		if last_obs[0] is None:
			last_obs, reward, done, info = env.step([actions[0]])
			continue


		obs_idx = replay_buffer.store_frame(last_obs[0])

		# Loading ReplayBuffer with actions that are:
		# 	1. Random with probability exploration_schedule.value(t)
		#	2. Chosen from the target DQN

		if random.random() < args.exploration_schedule.value(t) or not model_initialized:
			action_idx = random.randint(0,num_actions-1)
		else:
			encoded_obs = replay_buffer.encode_recent_observation()
			encoded_obs = encoded_obs[np.newaxis, ...]
			q_net_eval = sess.run(q_net, feed_dict={obs_t_ph: encoded_obs})
			action_idx = np.argmax(q_net_eval)

		last_obs, reward, done, info = env.step([actions[action_idx]])
		episode_rewards.append(reward[0])
		replay_buffer.store_effect(obs_idx, action_idx, reward[0], done)

		if done[0]:
			last_obs = env.reset()
			episode_rewards = []


		# Performing Experience Replay by sampling from the ReplayBuffer
		# and training the network with some random exploration criterion

		if t > learning_starts and t % learning_freq == 0 and replay_buffer.can_sample(batch_size):
			obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size)
			
			if not model_initialized:
				model_initialized = True
				initialize_interdependent_variables(session, tf.global_variables(), 
					{obs_t_ph: obs_batch, obs_tp1_ph: next_obs_batch})

			train_dict = {obs_t_ph: obs_batch,
						act_t_ph: act_batch,
						rew_t_ph: rew_batch,
						obs_tp1_ph: next_obs_batch,
						done_mask_ph: done_mask,
						learning_rate: args.optimizer.lr_schedule.value(t)
						}
			session.run(train_fn, feed_dict=train_dict)

			# Logging
			if len(episode_rewards) > 0:
				mean_episode_reward = np.mean(episode_rewards)
			if len(episode_rewards) > 100:
				best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

			if t % log_steps == 0 and model_initialized:
				if best_mean_episode_reward != -float('inf'):
					lfp = open(log_file, 'a')
					lfp.write("%s\t%s\t%s\t%s\t%s\t%s\n" %(t, 
						mean_episode_reward, 
						best_mean_episode_reward, 
						len(episode_rewards),
						args.exploration_schedule.value(t),
						args.optimizer.lr_schedule.value(t)))
					lfp.close()

	
def get_session(gpu_id):
	os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
	tf.reset_default_graph()
	tf_config = tf.ConfigProto(
		inter_op_parallelism_threads=1,
		intra_op_parallelism_threads=1)
	tf_config.gpu_options.visible_device_list=gpu_id
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
		env.configure(remotes=1) # TODO: potentially change this for different remotes
	elif args.task == "Torcs":
		raise NotImplementedError()

	sess = get_session(str(args.gpu))
	env = setup(env, args)
	train(env, sess, args)

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', type=int, default=-1, help='gpu id')
	parser.add_argument('--model', type=str, default="BaseDQN", help="type of network model for the Q network")
	parser.add_argument('--output_dir', type=str, default="output/", help="where to store all misc. training output")
	parser.add_argument('--task', type=str, choices=['DuskDrive', 'Torcs'], default="DuskDrive")
	parser.add_argument('--lr_mult', type=float, default=1.0, help='learning rate multiplier')
	parser.add_argument('--max_iters', type=int, default=2e6, help='number of timesteps to run DQN')
	parser.add_argument('--log_file', type=str, default="train.log", help="where to log DQN output")
	return parser.parse_args()


if __name__=="__main__":
	args = parse_args()
	main(args)
