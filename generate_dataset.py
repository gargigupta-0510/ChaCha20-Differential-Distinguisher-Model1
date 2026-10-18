import numpy as np
from chacha import chacha_permutation       #from chacha.py file


# -----------------------------
# Configuration
# -----------------------------

ROUNDS = 4

TRAIN_SAMPLES = 50000
TEST_SAMPLES = 10000

SEED = 42             #make random generation reproducible

rng = np.random.default_rng(SEED)


# One-bit input difference
DIFF_WORD = 14
DIFF_BIT = 6

# CREATING DELTA I.E. THE DIFFERENCE BETWEEN THE PLAINTEXTS.
def create_difference():
    delta = np.zeros(16, dtype=np.uint32)   #initailalize a 16-word array of zeros

    delta[DIFF_WORD] = np.uint32(1 << DIFF_BIT)    #delta[14] = 1 << 6 = 64, i.e. the difference is in the 14th word at the 6th bit position

    return delta


DELTA = create_difference()

# CREATING STAT MATRIX
def random_state():
    return rng.integers(
        0,               #minimum value (inclusive)
        2**32,           #maximum value (exclusive) 
        size=16,
        dtype=np.uint32
    )


def make_pair(differential=True):

    state0 = random_state()

    if differential:

        state1 = state0.copy()       

        state1 ^= DELTA

    else:

        state1 = random_state()

    cipher0 = chacha_permutation(
        state0,
        ROUNDS
    )

    cipher1 = chacha_permutation(
        state1,
        ROUNDS
    )

    return cipher0, cipher1


def state_to_bits(state):

    bits = np.unpackbits(
        state.astype(">u4").view(np.uint8)
    )

    return bits


def create_dataset(number):

    X = []
    y = []

    half = number // 2

    # -----------------------------
    # Differential samples
    # Label = 1
    # -----------------------------

    for _ in range(half):

        c0, c1 = make_pair(
            differential=True
        )

        sample = np.concatenate([
            state_to_bits(c0),
            state_to_bits(c1)
        ])

        X.append(sample)
        y.append(1)

    # -----------------------------
    # Random samples
    # Label = 0
    # -----------------------------

    for _ in range(half):

        c0, c1 = make_pair(
            differential=False
        )

        sample = np.concatenate([
            state_to_bits(c0),
            state_to_bits(c1)
        ])

        X.append(sample)
        y.append(0)

    X = np.array(X, dtype=np.float32)  #convert list into array as models works efficiently with arrays
    y = np.array(y, dtype=np.float32)

    # Shuffle
    indices = rng.permutation(len(X))  #creates random permutation of indexes of the array X and y, so that the order of samples is randomized

    X = X[indices]
    y = y[indices]

    return X, y


if __name__ == "__main__":

    print(f"Generating {TRAIN_SAMPLES} training samples...")

    X_train, y_train = create_dataset(
        TRAIN_SAMPLES
    )

    print(f"Generating {TEST_SAMPLES} test samples...")

    X_test, y_test = create_dataset(
        TEST_SAMPLES
    )

    np.savez_compressed(
        "data/train.npz",
        X=X_train,
        y=y_train
    )

    np.savez_compressed(
        "data/test.npz",
        X=X_test,
        y=y_test
    )

    print("\nDataset created.")

    print("Training shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    print("Differential samples:",
          np.sum(y_train == 1))

    print("Random samples:",
          np.sum(y_train == 0))