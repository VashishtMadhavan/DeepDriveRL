import matplotlib.pyplot as plt
import argparse

def plot_basic(args):
    lines = [x.rstrip().split(',') for x in open(args.log_file).readlines()][1:]
    iters = [int(x[0]) for x in lines]
    mean_rew = [float(x[1]) for x in lines]
    best_rew = [float(x[2]) for x in lines]
    plt.title("Rewards over Time")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.plot(iters, mean_rew, 'r', label="Mean Reward")
    plt.plot(iters, best_rew, 'g', label="Best Reward")
    plt.legend()
    plt.show()
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', type=str, help="Log File to extract rewards from")
    args = parser.parse_args()
    plot_basic(args)
