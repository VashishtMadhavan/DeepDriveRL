import os,argparse,sys, random
import gym
from gym import wrappers
import pickle
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


OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

"""
Script for training DfQD networks from Learning from Demonstration for Real
World RL paper 

"""
def setup(env, args):
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

    args.exploration_schedule = LinearSchedule(1000, 0.01, initial_p=1.0)
    return env

def task_input_shape(task):
    if task == "DuskDrive":
        return (128, 128, 3)
    elif task == "Torcs":
        return (64, 64, 3)
    elif task == "Torcs_novision":
        return (1, 1, 70)
    

def load_demonstrations(dem_file):
    data = h5py.File(open(dem_file))
    obs = data['obs']
    actions = data['actions']
    rewards = data['rewards']
    done = data['done']
    return obs, actions, rewards, done

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


def loss(q_net, q_net_tp1, target_q_net, act_t_ph, done_mask_ph, num_actions):
    lambda1 = 1.0
    lambda2 = 10e-5

    # Double DQN loss
    actions_mat = tf.one_hot(act_t_ph, num_actions)
    q_val = tf.reduce_max(q_net * actions_mat, axis=1)

    q_actions_mat = tf.one_hot(tf.argmax(q_net_tp1, axis=1), num_actions)
    target_q_val = rew_t_ph + gamma * tf.reduce_max(target_q_net * q_actions_mat, axis=1) * done_mask_ph
    loss_dqn = tf.reduce_mean(tf.square(target_q_val - q_val))

    # Regularization Loss
    loss_l2 = tf.reduce_sum([tf.reduce_mean(reg_loss) for reg_loss in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope="q_func")])

    # Supervised Large Margin Loss
    loss_l = (1. = tf.one_hot(act_t_ph, num_actions)) * 0.8
    loss_margin = tf.reduce_max(q_net + loss_l, axis=1) - q_val

    return loss_dqn + lambda1 * loss_margin + lambda2 * loss_l2

def train(env, session, args,
    replay_buffer_size=100000,
    batch_size=32,
    gamma=0.99,
    learning_starts=10000,
    learning_freq=4,
    frame_skip=8,
    frame_history_len=4,
    target_update_freq=1000,
    grad_norm_clipping=10):
    
    img_h, img_w, img_c = task_input_shape(args.task)
    input_shape = (img_h, img_w, frame_history_len * img_c)
    
    actions, num_actions = task_actions(args.task, env)
    dem_obs, dem_act, dem_rew = load_demonstrations(args.demonstrations)

    demo_buffer = ReplayBuffer(dem_obs.shape[0], frame_history_len)
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    obs_t_ph = tf.placeholder(tf.uint8, [None] + list(input_shape))
    act_t_ph = tf.placeholder(tf.int32, [None])
    rew_t_ph = tf.placeholder(tf.float32, [None])
    obs_tp1_ph = tf.placeholder(tf.uint8, [None] + list(input_shape))
    done_mask_ph = tf.placeholder(tf.float32, [None])

    obs_t_float   = tf.cast(obs_t_ph,   tf.float32) / 255.0
    obs_tp1_float = tf.cast(obs_tp1_ph, tf.float32) / 255.0
    lr = tf.placeholder(tf.float32, (), name="learn_rate")

    q_net = args.q_func(obs_t_float, num_actions, scope='q_func', reuse=False, regularize=True)
    q_net_tp1 = args.q_func(obs_tp1_float, num_actions, scope='q_func', reuse=True, regularize=True)
    q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func') 

    target_q_net= args.q_func(obs_tp1_float, num_actions, scope='tq_func', reuse=True)
    target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='tq_func')

    actions_mat = tf.one_hot(tf.argmax(q_net, axis=1), num_actions)
    q_val = tf.reduce_max(q_net * actions_mat, axis=1)
    
    q_actions_mat = tf.one_hot(tf.argmax(q_net_tp1, axis=1), num_actions)
    target_q_val = rew_t_ph + gamma * tf.reduce_max(target_q_net * q_actions_mat, axis=1) * done_mask_ph

    total_loss = loss(q_net, q_net_tp1, target_q_net, act_t_ph, done_mask_ph, num_actions)
    optim = tf.train.AdamOptimizer(learning_rate).minimize(pretrain_loss)
    train_fn = minimize_and_clip(optim, total_loss, var_list=q_func_vars, clip_val=grad_norm_clipping)

    update_target_fn = []
    for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name), 
                            sorted(target_q_func_vars, key=lambda v: v.name)):
        update_target_fn.append(var_target.assign(var))
    update_target_fn = tf.group(*update_target_fn)

    model_initailized = False
    saver = tf.train.Saver()
    if args.weights:
        saver.restore(session, args.weights)
        model_initailized = True

    # Step 1: Loading demonstration buffer
    for idx in range(dem_obs.shape[0]):
        scale_down = imresize(dem_obs[idx], (img_w, img_h, img_c))
        demo_obs_idx = demo_buffer.store_fame(scale_down)
        demo_buffer.store_effect(obs_idx, dem_act[idx], dem_rew[idx], dem_done[idx])

    # Step 2: Pretraining
    for p_step in range(args.pretrain_steps):
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
        session.run(train_fn, feed_dict=train_dict)

    # Step 3: Modified Q-Learning
    mean_episode_reward      = -float('nan')
    best_mean_episode_reward = -float('inf')
    total_episode_rewards = []
    t = 0
    for _ in range(args.q_learn_steps):
        last_done = False
        episode_reward = 0.
        last_obs = env.reset()
        while not last_done:
            if last_obs[0] is None:
                last_obs, reward, done, info = env.step([actions[0]])
                continue

            if t % args.save_period == 0:
                saver.save(session, os.path.join(args.snapshot_dir, "model_%s.ckpt" %(str(t))))

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

            demo_num = int(batch_size * args.demo_prob)
            demo_obs_batch, demo_act_batch, demo_rew_batch, demo_next_obs_batch, demo_done_mask = demo_buffer.sample(demo_num)
            replay_obs_batch, replay_act_batch, replay_rew_batch, replay_next_obs_batch, replay_done_mask = replay_buffer.sample(batch_size - demo_num)

            train_dict = {obs_t_ph: np.vstack((demo_obs_batch,replay_obs_batch)),
                        act_t_ph: np.concatenate((demo_act_batch, replay_act_batch)),
                        rew_t_ph: np.concatenate((demo_rew_batch, replay_rew_batch)),
                        obs_tp1_ph: np.vstack((demo_next_obs_batch, replay_next_obs_batch)),
                        done_mask_ph: np.concatenate((demo_done_mask, replay_done_mask)),
                        lr: args.optimizer.lr_schedule.value(t)
                        }
            session.run(train_fn, feed_dict=train_dict)

            last_done = done[0] if args.task == "DuskDrive" else done
            t += 1
        
        last_obs = env.reset()
        total_episode_rewards.append(episode_reward)
        episode_reward = 0
        session.run(update_target_fn)

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
        from gym_torcs import TorcsEnv
        env = TorcsEnv(vision=True, throttle=False)
    elif args.task == "Torcs_novision":
        from gym_torcs import TorcsEnv
        env = TorcsEnv(vision=False, throttle=False)
        

    sess = get_session(str(args.gpu))
    env = setup(env, args)
    train(env, sess, args)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demonstrations', type=str, default="demonstrations.h5", help='demonstrations file')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--model', type=str, default="BaseDQN", help="type of network model for the Q network")
    parser.add_argument('--output_dir', type=str, default="output/", help="where to store all misc. training output")
    parser.add_argument('--task', type=str, choices=['DuskDrive', 'Torcs', 'Torcs_novision'], default="DuskDrive")
    parser.add_argument('--weights', type=str, default=None, help="path to model weights")
    parser.add_argument('--max_iters', type=int, default=10e6, help='number of timesteps to run DQN')
    parser.add_argument('--render', action='store_true', help='If true, will call env.render()')
    parser.add_argument('--save_period', type=int, default=1e6, help='period of saving checkpoints')
    
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    main(args)
