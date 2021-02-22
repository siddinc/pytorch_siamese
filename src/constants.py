import os


DATASET_PATH = os.path.abspath("../dataset")
MPL_PATH = os.path.abspath("../fig.mplstyle")

INPUT_SHAPE = (3000, 1)
N_TIMESTEPS = 3000
N_FEATURES = 1
CHANNEL_INDEX = 22    #22, 55, 63
VAL_SPLIT = 0.3
TEST_SPLIT = 0.3

L1_PENALTY = 0.001
L2_PENALTY = 0.001
ALPHA = 0.2
MARGIN = 1.0

BATCH_SIZE = 16
N_WORKERS = 4
SHUFFLE = True
N_EPOCHS = 100
MAX_LR = 0.0003
WEIGHT_DECAY = 0.0001
BETA_1 = 0.5
BETA_2 = 0.999

N_TRIALS = 500
SUPPORT_SET_SIZE = 20

NORM_DEG = {
  "manhattan": 1,
  "euclidean": 2,
}