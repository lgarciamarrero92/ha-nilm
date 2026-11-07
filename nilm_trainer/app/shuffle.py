"""Reproduce the TensorFlow 2.15 training batch order without TensorFlow."""

import numpy as np


def _philox_words(seed, counter_seed, count):
    """Generate the 32-bit words of the Philox4x32-10 stream used by tf.data."""
    mask = 0xFFFFFFFF
    first = np.arange((count + 3) // 4, dtype=np.uint64)
    second = np.zeros_like(first)
    third = np.full_like(first, counter_seed & mask)
    fourth = np.full_like(first, counter_seed >> 32)
    key_low, key_high = seed & mask, seed >> 32

    for _ in range(10):
        left = first * 0xD2511F53
        right = third * 0xCD9E8D57
        first, second = (right >> 32) ^ second ^ key_low, right & mask
        third, fourth = (left >> 32) ^ fourth ^ key_high, left & mask
        key_low = (key_low + 0x9E3779B9) & mask
        key_high = (key_high + 0xBB67AE85) & mask

    return np.column_stack((first, second, third, fourth)).reshape(-1)[:count]


def epoch_orders(size, seed, epochs):
    """Shuffle the same rolling buffer as the original seeded training stages."""
    max_seed = 2**31 - 1
    seed %= max_seed
    # The original trainer set both the global and dataset seeds to the job seed.
    # tf.data maps the all-zero pair to (0, max_seed).
    stage_seeds = _philox_words(seed, seed or max_seed, 2 * epochs).reshape(-1, 2)
    capacity = min(size, 10_000)
    positions = np.arange(size)
    available = np.minimum(size - positions, capacity).astype(np.uint64)

    for first_seed, second_seed in stage_seeds.tolist():
        offsets = _philox_words(first_seed, second_seed, size) % available
        slots = ((positions + offsets.astype(np.int64)) % capacity).tolist()
        buffer = list(range(capacity))
        order = np.empty(size, dtype=np.intp)
        for position, slot in enumerate(slots):
            front = position % capacity
            order[position] = buffer[slot]
            buffer[slot] = buffer[front]
            # The front becomes the next insertion slot after the dequeue.
            buffer[front] = capacity + position
        yield order
