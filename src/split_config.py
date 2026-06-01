CURRENT_SPLIT = 0

SPLITS = [
    0.6,
    0.7,
    0.8,
]


def get_current_split():
    return CURRENT_SPLIT


def next_split():
    global CURRENT_SPLIT

    if CURRENT_SPLIT < len(SPLITS) - 1:
        CURRENT_SPLIT += 1

    return CURRENT_SPLIT


def get_train_ratio():
    return SPLITS[CURRENT_SPLIT]