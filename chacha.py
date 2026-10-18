import numpy as np

MASK32 = 0xFFFFFFFF

def rotl32(x, n):
    #32-bit left rotation.
    return ((x << n) | (x >> (32 - n))) & MASK32


def quarter_round(a, b, c, d):
    #ChaCha quarter round.

    a = (a + b) & MASK32
    d ^= a
    d = rotl32(d, 16)

    c = (c + d) & MASK32
    b ^= c
    b = rotl32(b, 12)

    a = (a + b) & MASK32
    d ^= a
    d = rotl32(d, 8)

    c = (c + d) & MASK32
    b ^= c
    b = rotl32(b, 7)

    return a, b, c, d


def column_round(state):
    #Apply quarter rounds to ChaCha columns.
    

    state[0], state[4], state[8], state[12] = quarter_round(
        state[0], state[4], state[8], state[12])

    state[1], state[5], state[9], state[13] = quarter_round(
        state[1], state[5], state[9], state[13])

    state[2], state[6], state[10], state[14] = quarter_round(
        state[2], state[6], state[10], state[14])

    state[3], state[7], state[11], state[15] = quarter_round(
        state[3], state[7], state[11], state[15])

    return state


def diagonal_round(state):
    #Apply quarter rounds to ChaCha diagonals.
    

    state[0], state[5], state[10], state[15] = quarter_round(
        state[0], state[5], state[10], state[15])

    state[1], state[6], state[11], state[12] = quarter_round(
        state[1], state[6], state[11], state[12])

    state[2], state[7], state[8], state[13] = quarter_round(
        state[2], state[7], state[8], state[13])

    state[3], state[4], state[9], state[14] = quarter_round(
        state[3], state[4], state[9], state[14])

    return state


def chacha_permutation(state, rounds):
    """
    Reduced-round ChaCha permutation.

    rounds = number of individual rounds.

    Example:
        rounds=3  -> column, diagonal, column
        rounds=4  -> column, diagonal, column, diagonal
    """

    state = state.copy()

    for r in range(rounds):

        if r % 2 == 0:
            state = column_round(state)
        else:
            state = diagonal_round(state)

    return state