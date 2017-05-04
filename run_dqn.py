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
Setting up learning rate scheduler and exploration criterion
"""
def setup(env, args):
    args.lr_schedule = PiecewiseSchedule([
    (0, 1e-4 * args.lr_mult),
    (args.max_iters / 10, 1e-4 * args.lr_mult),
    (args.max_iters / 2,  5e-5 * args.lr_mult),
    ], outside_value=5e-5 * args.lr_mult)

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
        (args.max_iters / 2, 0.01),
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

    # Placeholder Formatting
    obs_t_ph = tf.placeholder(tf.uint8, [None] + list(input_shape))
    act_t_ph = tf.placeholder(tf.int32, [None])
    rew_t_ph = tf.placeholder(tf.float32, [None])
    obs_tp1_ph = tf.placeholder(tf.uint8, [None] + list(input_shape)) # action at next timestep
    done_mask_ph = tf.placeholder(tf.float32, [None]) # 0 if next state is end of episode

    obs_t_float   = tf.cast(obs_t_ph,   tf.float32) / 255.0
    obs_tp1_float = tf.cast(obs_tp1_ph, tf.float32) / 255.0

    # Double Q learning dynamics
    q_net = args.q_func(obs_t_float, num_actions, scope='q_func', reuse=False)
    q_net_tp1 = args.q_func(obs_tp1_float, num_actions, scope='q_func', reuse=True)
    q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func') 

    target_q_net= args.q_func(obs_tp1_float, num_actions, scope='tq_func', reuse=False)
    target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='tq_func')

    actions_mat = tf.one_hot(tf.argmax(q_net, axis=1), num_actions)
    q_val = tf.reduce_max(q_net * actions_mat, axis=1)
    
    q_actions_mat = tf.one_hot(tf.argmax(q_net_tp1, axis=1), num_actions)
    target_q_val = rew_t_ph + gamma * tf.reduce_max(target_q_net * q_actions_mat, axis=1) * done_mask_ph

    error = tf.reduce_mean(tf.square(target_q_val - q_val))

    # Optimization parameters
    lr = tf.placeholder(tf.float32, (), name="learn_rate")
    opt = args.optimizer.constructor(learning_rate=lr, **args.optimizer.kwargs)
    train_fn = minimize_and_clip(opt, error, var_list=q_func_vars, clip_val=grad_norm_clipping)

    update_target_fn = []
    for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name), 
                            sorted(target_q_func_vars, key=lambda v: v.name)):

        update_target_fn.append(var_target.assign(var))

    update_target_fn = tf.group(*update_target_fn)
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    ###############
    # Run Env    #
    ###############
    model_initialized = False
    num_param_updates = 0
    mean_episode_reward      = -float('nan')
    best_mean_episode_reward = -float('inf')
    total_episode_rewards = []
    episode_reward = 0.   

    log_steps = 10000
    with open(args.log_file,'w') as lfp:
        lfp.write("Timestep, Mean Reward, Best Reward\n")
 

    saver = tf.train.Saver()
    if args.task == 'DuskDrive':
        last_obs = env.reset()
    else:
        last_obs = env.reset(relaunch=True)

    if args.weights:
        model_initialized = True
        saver.restore(session, args.weights)


    for t in itertools.count():
        if args.render:
            env.render()

        if t % args.save_period == 0 and model_initialized:
            save_path = saver.save(session, os.path.join(args.snapshot_dir, "model_%s.ckpt" %(str(t))))

        if t >= args.max_iters:
            break

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

        # Loading ReplayBuffer with actions that are:
        #   1. Random with probability exploration_schedule.value(t)
        #   2. Chosen from the target DQN
	
        obs_idx = replay_buffer.store_frame(down_samp)
        eps = args.exploration_schedule.value(t)

        if np.random.rand(1) < eps or (not model_initialized):
            action_idx = np.random.choice(num_actions)
        else:
            encoded_obs = replay_buffer.encode_recent_observation()[np.newaxis,...]
            q_net_eval = session.run(q_net, feed_dict={obs_t_ph: encoded_obs})
            action_idx = np.argmax(q_net_eval)
        
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
            
            if not model_initialized:
                model_initialized = True
                initialize_interdependent_variables(session, tf.global_variables(), 
                        {obs_t_ph: obs_batch, obs_tp1_ph: next_obs_batch})


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
            if len(total_episode_rewards) > 100:
                best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
            if t % log_steps == 0 and model_initialized:
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
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--model', type=str, default="BaseDQN", help="type of network model for the Q network")
    parser.add_argument('--output_dir', type=str, default="output/", help="where to store all misc. training output")
    parser.add_argument('--task', type=str, choices=['DuskDrive', 'Torcs', 'Torcs_novision'], default="DuskDrive")
    parser.add_argument('--lr_mult', type=float, default=1.0, help='learning rate multiplier')
    parser.add_argument('--weights', type=str, default=None, help="path to model weights")
    parser.add_argument('--max_iters', type=int, default=10e6, help='number of timesteps to run DQN')
    parser.add_argument('--render', action='store_true', help='If true, will call env.render()')
    parser.add_argument('--save_period', type=int, default=1e6, help='period of saving checkpoints')
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    main(args)
