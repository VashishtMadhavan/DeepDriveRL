# DeepDriveRL
RL project for driving

### Dependencies

- Universe
- Gym
- Tensorflow
- Numpy, Scipy + various conda packages


### Testing your Universe Setup

The test_universe.py script provides a simple script for testing your Universe installation. To run it use the following:

```
python test_universe.py
```

### Training A DQN model

```
python run_dqn.py --gpu <gpu_id> --task DuskDrive --model BaseDQN
```

The command above runs a "BaseDQN" model defined inside network.py on the Dusk Drive task. The other arguments are defined inside run_dqn.py. 

For testing out new network architectures:

- define them in network.py
- add support for them in setup() of run_dqn.py


## Training A Imitation Model

```
python imitation.py --gpu <gpu_id> --task DuskDrive --model BaseDQN --obs <observation_file>
```

The command above trains a "BaseDQN" model on the observations (i.e. images + actions) on the observations
defined in  observation_file. This should be a pickle file with keys 'obs' and 'actions'. Other arguments are
defined inside the imitation.py file


### Misc. Notes

- original DuskDrive input is [512, 800, 3]. In order to effectively train a DQN we need to downsample and resize to (128,128,3) and
  reduce the ReplayBuffer size from 1M to 100,000. This will probably carry over to other games with high dimensional output as well.

