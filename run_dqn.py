import os,argparse,sys, random
import gym
from gym.wrappers import Monitor
import universe
from universe.wrappers import Vision, Logger
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
	(num_iterations / 10, 1e-4 * args.lr_mult),
	(num_iterations / 2,  5e-5 * args.lr_mult),
	], outside_value=5e-5 * args.lr_mult)

	args.optimizer = OptimizerSpec(
	constructor=tf.train.AdamOptimizer,
	kwargs=dict(epsilon=1e-4),
	lr_schedule=args.lr_schedule
	)

	env = Monitor(env, args.monitor_dir)
	env = Vision(env)
	env = CropObservations(env) #TODO: dont really know if this works see if you could keep it aournd
	env = SafeActionSpace(env)
	env = Logger(env)

	args.num_timesteps = args.iters * 4

	def stop_criterion(env, max_t):
		return env.get_total_steps() >= max_t

	args.stopping = stop_criterion
	args.exploration_schedule = PiecewiseSchedule(
	[
		(0, 1.0),
		(1e6, 0.1),
		(args.iters / 2, 0.01),
		], outside_value=0.01
	)
	return env

def train(env, session, args, 
	q_func=dqn_base, 
	replay_buffer_size=1000000,
	batch_size=32,
	gamma=0.99,
	learning_starts=50000,
	learning_freq=4,
	frame_history_len=4,
	target_update_freq=10000,
	grad_norm_clipping=10):

	#TODO: potentially adapt this to include more actions...SafeActionSpace sort of restricts total possible space of actions
	img_h, img_w, img_c = (512, 800, 3)
	input_shape = (img_h, img_w, frame_history_len * img_c) # to account for sequence of frames
	actions = env.action_space[0] + env.action_space[1] + env.action_space[2]
	num_actions = len(actions)

	# Placeholder Formatting
	obs_t_ph = tf.placeholder(tf.uint8, [None] + list(input_shape))
	act_t_ph = tf.placeholder(tf.int32, [None])
	rew_t_ph = tf.placeholder(tf.float32, [None])
	obs_t2_ph = tf.placeholder(tf.uint8, [None] + list(input_shape)) # action at next timestep
	done_mask_ph = tf.placeholder(tf.float32, [None]) # 0 if next state is end of episode

	obs_t_float   = tf.cast(obs_t_ph,   tf.float32) / 255.0
	obs_tp1_float = tf.cast(obs_tp1_ph, tf.float32) / 255.0


	# Q learning dynamics
	actions_mat = tf.one_hot(act_t_ph, num_actions, off_value=0.0, on_value=1.0, axis=-1)
	q_val = q_func(obs_t_float, num_actions, scope='q_func', reuse=False)
	target_q_val = q_func(obs_tp1_float, num_actions, scope='tq_func', reuse=False)

	q_val = tf.reduce_sum(q_val * actions_mat, reduction_indices=1)
	target_q_val = rew_t_ph + gamma * tf.reduce_max(target_q_val, reduction_indices=1) * done_mask_ph
	error = tf.reduce_mean(tf.square(target_q_val - q_val))

	q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
	target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='tq_func')

	# Optimization parameters
	lr = tf.placeholder(tf.float32, (), name="learn_rate")
	opt = args.optimizer.constructor(learning_rate=lr, **args.optimizer.kwargs)
	train_fn = minimize_and_clip(optimizer, total_error, var_list=q_func_vars, clip_val=grad_norm_clipping)

	update_target_fn = []
	for var, var_target in zip(sorted(q_func_vars, 
		key=lambda v: v.name), 
		sorted(target_q_func_vars, 
		key=lambda v: v.name)):

		update_target_fn.append(var_target.assign(var))
		update_target_fn = tf.group(*update_target_fn)

	replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

	###############
    # RUN ENV     #
    ###############
    model_initialized = False
    num_param_updates = 0
    mean_episode_reward      = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    log_steps = 10000

    plot_iters = []
    plot_mean_rewards = []
    plot_best_rewards = []






	

	







def get_session(gpu_id):
	os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    tf_config.gpu_options.visible_device_list=gpu_id
    session = tf.Session(config=tf_config)
    return session


def main(args):
	args.seed = 0
	args.output_dir = 'experiment_%s' %(args.exp_name)

	if not os.path.exists(args.output_dir):
		os.mkdir(args.output_dir)

	if args.task == "DuskDrive":
		env = gym.make('flashgames.DuskDrive-v0')
		env.configure(remotes=1) # TODO: potentially change this for different remotes
	elif args.task == "Torcs":
		raise NotImplementedError()

	sess = get_session(args.gpu_id)
	env = setup(env, args)



def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', type=int, default=0, help='gpu id')
	parser.add_argument('--exp_name', type=str, default="base", help="experiment names to keep track of different runs")
	parser.add_argument('--output_dir', type=str, default="output/", help="where to store all misc. training output")
	parser.add_argument('--monitor_dir', type=str, default='monitor', help='where to store monitor output')
	parser.add_argument('--task', type=str, choices=['DuskDrive', 'Torcs'], default="DuskDrive")
	parser.add_argument('--lr_mult', type=float, default=1.0, help='learning rate multiplier')
	parser.add_argument('--iters', type=int, default=10000, help='number of iterations to train DQN')
	parser.add_argument('--log_file', type=str, default="train.log", help="where to log DQN output")
	return parser.parse_args()


if __name__=="__main__":
	args = parse_args()
	main(args)