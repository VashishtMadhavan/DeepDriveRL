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
    (0,1e-4 * args.lr_mult),
    (args.max_iters / 4, 1e-4 * args.lr_mult),
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

    args.summary_dir = os.path.join(args.output_dir, "summary")
    if not os.path.exists(args.summary_dir):
        os.mkdir(args.summary_dir)

    if args.phase == "test":
        args.demonstrations_file = os.path.join(args.output_dir, "demonstrations.pkl")

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
        (args.max_iters / 10, 0.1),
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

    


#TODO: add support for different Q networks
def train(env, session, args,
    replay_buffer_size=100000,
    batch_size=32,
    gamma=0.99,
    learning_starts=10000,
    learning_freq=4,
    frame_history_len=4,
    target_update_freq=1000,
    grad_norm_clipping=10):
    
    # resizing all network input to (128,128,3) for ease of processing
    if args.task == "DuskDrive":
        img_h, img_w, img_c = (128, 128, 3)
    elif args.task == "Torcs":
        img_h, img_w, img_c = (64, 64, 3)
    elif args.task == "Torcs_novision":
        img_h, img_w, img_c = (1, 1, 70)
    input_shape = (img_h, img_w, frame_history_len * img_c)
    
    if args.task == "DuskDrive":
        actions = env.action_space.actions
        num_actions = len(actions)
    elif args.task == "Torcs":
        actions = [-1, 0, 1]
        num_actions = 3
    elif args.task == "Torcs_novision":
        actions = [-1, 0, 1]
        num_actions = 3


    # Placeholder Formatting
    obs_t_ph = tf.placeholder(tf.uint8, [None] + list(input_shape))
    act_t_ph = tf.placeholder(tf.int32, [None])
    rew_t_ph = tf.placeholder(tf.float32, [None])
    obs_tp1_ph = tf.placeholder(tf.uint8, [None] + list(input_shape)) # action at next timestep
    done_mask_ph = tf.placeholder(tf.float32, [None]) # 0 if next state is end of episode

    obs_t_float   = tf.cast(obs_t_ph,   tf.float32) / 255.0
    obs_tp1_float = tf.cast(obs_tp1_ph, tf.float32) / 255.0

    tf.summary.scalar("Reward Mean", tf.reduce_mean(rew_t_ph))
    tf.summary.scalar("Reward Max", tf.reduce_max(rew_t_ph))
    tf.summary.scalar("Reward Min", tf.reduce_min(rew_t_ph))
    tf.summary.histogram("Reward Hist", rew_t_ph)

    # Q learning dynamics
    actions_mat = tf.one_hot(act_t_ph, num_actions, off_value=0.0, on_value=1.0, axis=-1)
    q_net = args.q_func(obs_t_float, num_actions, scope='q_func', reuse=False)
    target_q_net= args.q_func(obs_tp1_float, num_actions, scope='tq_func', reuse=False)

    q_val = tf.reduce_sum(q_net * actions_mat, reduction_indices=1)
    target_q_val = rew_t_ph + gamma * tf.reduce_max(target_q_net, reduction_indices=1) * done_mask_ph
    error = tf.reduce_mean(tf.square(target_q_val - q_val))
    tf.summary.scalar("Train Error", error)

    q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
    target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='tq_func')
    
    # Optimization parameters
    lr = tf.placeholder(tf.float32, (), name="learn_rate")
    tf.summary.scalar("Learning Rate", lr)
    opt = args.optimizer.constructor(learning_rate=lr, **args.optimizer.kwargs)
    train_fn = minimize_and_clip(opt, error, var_list=q_func_vars, clip_val=grad_norm_clipping)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(args.summary_dir)

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
    episode_rewards = []
    log_steps = 1000
    last_reward = None
 
    saver = tf.train.Saver()
    if args.task == 'DuskDrive':
        last_obs = env.reset()
    else:
        last_obs = env.reset(relaunch=True)
    log_file = "%s/%s" % (args.output_dir, args.log_file)

    lfp = open(log_file, 'w')
    lfp.write("Timestep\tMean Reward\tBest Mean Reward\tEpisodes\tExploration\tLearning Rate\n")
    lfp.close()

    if args.weights:
        model_initialized = True
        print('loaded weights ' + args.weights)
        saver.restore(session, args.weights)

    if args.phase == "test":
        dem_obs = []; dem_actions = []

    for t in itertools.count():
        #print ("Iteration: " + str(t))
        if args.render:
            env.render()

        if t % args.save_period == 0 and model_initialized:
            if args.phase == "train":
                save_path = saver.save(session, os.path.join(args.snapshot_dir, "model_%s.ckpt" %(str(t))))

        if t >= args.max_iters:
            if args.phase == "test":
                ret_dict = {'obs': np.squeeze(np.array(dem_obs)), 'actions': np.array(dem_actions)}
                pickle.dump(ret_dict, open(args.demonstrations_file,'w'))
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
        obs_idx = replay_buffer.store_frame(down_samp)

        # Loading ReplayBuffer with actions that are:
        #   1. Random with probability exploration_schedule.value(t)
        #   2. Chosen from the target DQN

        if (random.random() < args.exploration_schedule.value(t) or not model_initialized) and args.phase == "train":
            action_idx = random.randint(0,num_actions-1)
            #print('random', action_idx)
        else:
            encoded_obs = replay_buffer.encode_recent_observation()
            encoded_obs = encoded_obs[np.newaxis, ...]
            q_net_eval = session.run(q_net, feed_dict={obs_t_ph: encoded_obs})
            action_idx = np.argmax(q_net_eval)
            #print(action_idx, q_net_eval)

        last_obs, reward, done, info = env.step([actions[action_idx]])
        #reward = reward_from_obs(last_obs, reward, done)
        #reward = -reward
        #print('reward: ' + str(reward))
        
        if not args.velocity:
            last_reward = reward[0] if args.task == "DuskDrive" else reward
        else:
            if not last_reward or reward[0] >= last_reward:
	        last_reward = reward[0]
            else:
                # adding penalty for negative changes in reward
		last_reward = reward[0] + (reward[0] - last_reward)


        episode_rewards.append(last_reward)
        replay_buffer.store_effect(obs_idx, action_idx, last_reward, done)

        #observation collection for imitation learning
        if args.phase == "test":
            dem_obs.append(encoded_obs)
            dem_actions.append(action_idx)

        last_done = done[0] if args.task == "DuskDrive" else done
        if last_done:
            if args.task == "DuskDrive":
                last_obs = env.reset()
            else:
                last_obs = env.reset(relaunch=True)
            episode_rewards = []


        # Performing Experience Replay by sampling from the ReplayBuffer
        # and training the network with some random exploration criterion

        if t > learning_starts and t % learning_freq == 0 and replay_buffer.can_sample(batch_size) and args.phase == "train":
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
            summary, _ = session.run([merged, train_fn], feed_dict=train_dict)
            train_writer.add_summary(summary, t)

            if t % target_update_freq == 0:
                print('updating target networks...')
                session.run(update_target_fn)
                num_param_updates += 1            


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
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    #tf_config.gpu_options.visible_device_list=gpu_id
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
    parser.add_argument('--gpu', type=int, default=4, help='gpu id')
    parser.add_argument('--model', type=str, default="BaseDQN", help="type of network model for the Q network")
    parser.add_argument('--output_dir', type=str, default="output_full_test2/", help="where to store all misc. training output")
    parser.add_argument('--task', type=str, choices=['DuskDrive', 'Torcs', 'Torcs_novision'], default="DuskDrive")
    parser.add_argument('--lr_mult', type=float, default=1.0, help='learning rate multiplier')
    parser.add_argument('--phase', nargs='?', choices=['train', 'test'], default='test')
    parser.add_argument('--weights', type=str, default="output_full/weights/model_7000000.ckpt", help="path to model weights")
    parser.add_argument('--max_iters', type=int, default=20000, help='number of timesteps to run DQN')
    parser.add_argument('--log_file', type=str, default="train.log", help="where to log DQN output")
    parser.add_argument('--render', action='store_true', help='If true, will call env.render()')
    parser.add_argument('--velocity', action='store_true', help='velocity constraint for dusk drive')
    parser.add_argument('--save_period', type=int, default=1e6, help='period of saving checkpoints')
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    main(args)
