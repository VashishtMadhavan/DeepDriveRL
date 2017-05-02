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
Setting up parameters for running Universe + Training Behavioral Cloning Model
"""

def load_data(data_file, frame_history=4):
	data = pickle.load(open(data_file))
	obs = data['obs']; act = data['actions'].flatten()
	obs = obs.reshape(-1, arr.shape[2], arr.shape[3], arr.shape[4])

	observations = []; actions = []
	for i, o in enumerate(obs):
		for j in range(frame_history):
			observations.append(obs[i][:, :, 3*j:3*(j+1)])
			actions.append(act[i])
	return observations, actions

def setup(env, args):
	# setting up dir to store monitor output
	monitor_dir = os.path.join(args.output_dir, "monitor")
	if not os.path.exists(monitor_dir):
		os.mkdir(monitor_dir)

	# setting up dir to store model weights
	args.snapshot_dir = os.path.join(args.output_dir, "weights")
	if not os.path.exists(args.snapshot_dir):
		os.mkdir(args.snapshot_dir)

        args.summary_dir = os.path.join(args.output_dir, "summary")
	if not os.path.exists(args.summary_dir):
		os.mkdir(args.summary_dir)

	env = Logger(env)
	env = Monitor(env, monitor_dir, force=True)
	env = Vision(env)
	env = CropObservations(env)
	env = SafeActionSpace(env)

	if args.model == "BaseDQN":
		args.network = dqn_base
	else:
		raise NotImplementedError("Add support for different Q network models")
	return env

def train(env, session, args, batch_size=32, epsilon=0.03):
	observations, actions = load_data(args.data_file)

	#TODO: implement Torcs action map
	if args.task == "DuskDrive":
		action_map = env.action_space
	elif args.task == "Torcs":
		raise NotImplementedError("Please implement Torcs Functionality...")

	input_shape = observations[0].shape
	num_actions = len(action_map)
	n = observations.shape[0]

	# Setting up Training and Validation sets
	train_idx = np.random.choice(n, int(0.9*n),replace=False)
	test_idx = np.delete(np.arange(n),train_idx)
	
	train_obs = observations[train_idx]; train_act = actions[train_idx]
	val_obs = observations[test_idx]; val_act = actions[test_idx]

    	# Placeholder Formatting
	obs_t_ph = tf.placeholder(tf.uint8, [None] + list(input_shape))
	act_t_ph = tf.placeholder(tf.int32, [None])
	obs_t_float = tf.cast(obs_t_ph, tf.float32) / 255.0

	# Q network imitation
	q_net = args.q_func(obs_t_float, num_actions, scope='q_func', reuse=False)
	y_pred = tf.argmax(q_net, axis=1)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=act_t_ph, logits=y_pred))
	tf.summary.scalar("Loss", loss)

	pred = tf.equal(tf.argmax(q_net,1), tf.argmax(ac_t_ph,1))
	acc = tf.reduce_mean(tf.cast(pred, tf.float32))
	tf.summar.scalar("Accuracy", acc)

	opt = tf.train.AdamOptimizer(args.lr).minimize(loss)

	merge = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(args.summary_dir + "/train")
	test_writer = tf.summary.FileWriter(args.summary_dir + "/test")


	###############
	# Run Env    #
	###############
	model_initialized = False
	eval_iters = 10000
	test_batch_size = 1000
	
	saver = tf.train.Saver()
	last_obs = env.reset()

	if args.weights:
		model_initialized = True
		saver.restore(session, args.weights)

	if args.phase == "test":
		avgAcc = 0.

	for t in range(args.iters):
		batch_idx = np.random.choice(train_obs.shape[0], batch_size, replace=False)
		obs_batch = obs_batch[batch_idx]
		act_batch = []

		# To add randomness to learned representations for imitation learning
		for bi in batch_idx:
			if random.random() < epsilon:
				act_batch.append(random.randint(0,num_actions-1))
			else:
				act_batch.append(train_act[bi])
		act_batch = np.array(act_batch)

		if not model_initialized:
			initialize_interdependent_variables(session, 
				tf.global_variables(), 
				{obs_t_ph: obs_batch, act_t_ph: act_batch})

		# Training step
		train_dict = {obs_t_ph: obs_batch, act_t_ph: act_batch}
		if args.phase == "train":
			summary, _ = session.run([merged, opt], feed_dict=train_dict)
			train_writer.add_summary(summary, t)

		elif args.phase == "test":
			train_acc = session.run(acc, feed_dict=train_dict)
			avgAcc += train_acc

		if t % eval_iters == 0 and args.phase == "train":
			test_batch_idx = np.random.choice(val_obs.shape[0], test_batch_size, replace=False)
			summary, test_acc = session.run([merged, acc], feed_dict={obs_t_ph: val_obs[test_batch_idx], act_t_ph: val_act[test_batch_idx]})
			test_writer.add_summary(summary, t)

	if args.phase == "train":
		saver.save(session, os.path.join(args.snapshot_dir, "model_%s.ckpt" % (args.iters)))
		print "Done Training..."
	elif args.phase == "test":
		print "Test Accuracy: " + str(avgAcc / float(args.iters))


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
		env.configure(remotes=1)

	elif args.task == "Torcs":
		raise NotImplementedError()

	sess = get_session(str(args.gpu))
	env = setup(env, args)
	train(env, sess, args)

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', type=int, default=0, help='gpu id')
	parser.add_argument('--data_file', type=str, help="Data file with observations and actions")
	parser.add_argument('--phase', type=str, choices=['train', 'test'], default='train', help='training or test phase')
	parser.add_argument('--model', type=str, default="BaseDQN", help="type of network model for the Q network")
	parser.add_argument('--output_dir', type=str, default="output/", help="where to store all misc. training output")
	parser.add_argument('--train', action='store_true', help="If true, executes training phase of network")
	parser.add_argument('--task', type=str, choices=['DuskDrive', 'Torcs'], default="DuskDrive")
	parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
	parser.add_argument('--weights', type=str, default=None, help="path to model weights")
	parser.add_argument('--iters', type=int, default=20000, help='number of timesteps to run DQN')
	return parser.parse_args()

if __name__=="__main__":
	args = parse_args()
	main(args)
