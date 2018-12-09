# DoodleGAN
COMP 4651 Cloud Computing Course Project

## Getting Started
1. install Tensorflow using Conda
2. install remaining python packages
```
$ pip install -r requirements.txt
```
3. download dataset
```
$ ./download.sh
```

### Local Training
```
$ python main.py -c configs/baseline.json
```

### Tensorboard Visualization
```
$ tensorboard --logdir=experiments
```

