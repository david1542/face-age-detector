import os

# Paths
ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(ROOT_PATH, 'data')

RAW_DATA_PATH = os.path.join(DATA_PATH, 'raw')

ALIGNED_FACES_PATH = os.path.join(RAW_DATA_PATH, 'aligned')
REGULAR_FACES_PATH = os.path.join(RAW_DATA_PATH, 'faces')
