# Settings for the softmax optimisiation.
MAX_EM = 50
MIN_M = 0
MIN_V = 0

MULTIOBJECTIVE_BETA = 50
TARGET_EM = 20
TARGET_M = [200, 0.1, 1, 100]
TARGET_V = 94


NORM_M = TARGET_M
NORM_V = TARGET_V - MIN_V
NORM_EM = MAX_EM - TARGET_EM

TARGET_BIN = 10**(-2.3)