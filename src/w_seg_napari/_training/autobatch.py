"""Pick the largest batch size that fits in GPU memory (pure logic, no torch)."""


def find_max_batch(fits, start=2, limit=128):
    """
    Largest batch size b (1 <= b <= limit) for which ``fits(b)`` is True,
    assuming that if b fits then every smaller size fits too.

    Doubles the size from `start` until it no longer fits (or reaches the
    limit), then bisects between the last size that fit and the first that did
    not. Returns 0 when not even a batch of 1 fits.
    """
    if not fits(1):
        return 0
    good, bad = 1, None
    b = max(2, int(start))
    while b <= limit:
        if fits(b):
            good = b
            b *= 2
        else:
            bad = b
            break
    if bad is None:
        if good >= limit:
            return limit
        # the doubling jumped over the limit: try the limit itself
        if fits(limit):
            return limit
        bad = limit
    while bad - good > 1:
        mid = (good + bad) // 2
        if fits(mid):
            good = mid
        else:
            bad = mid
    return good


def safe_batch(max_batch, margin=0.85):
    """
    Batch size to train with: a margin below the largest that fits, because
    validation, the progress figure and memory fragmentation need headroom.
    Even from 4 up (it is a little faster), never below 1.
    """
    if max_batch <= 0:
        return 0
    b = max(1, int(max_batch * margin))
    if b >= 4:
        b -= b % 2
    return b
