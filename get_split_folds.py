def get_folds(fold_num, seed):
    """
    given fold number for test fold and random seed number, return [train, val, test] indices as list of lists
    total number of folds = 5
    """
    FOLDS = split_data_folds(seed)

    test_inds = FOLDS[fold_num]
    val_inds = FOLDS[(fold_num + 1) % 5]
    train_inds = FOLDS[(fold_num + 2) % 5] + FOLDS[(fold_num + 3) % 5] + FOLDS[(fold_num + 4) % 5]

    return [train_inds, val_inds, test_inds]


def split_data_folds(seed):
    """
    split subjects into 12 buckets by MAP value, then produce 5 folds of subject indices with 1 subject per bucket in
    each fold
    returns dictionary of fold num (0-4): subject indices
    """
    import numpy as np
    import random
    random.seed(seed)
    np.random.seed(seed)

    part = 1
    map_all = np.load(f'./data/measured_mit_v1/npy/measured_mit_v1_part{part}_map_all.npy')
    age_all = np.load(f'./data/measured_mit_v1/npy/measured_mit_v1_part{part}_age_all.npy')

    print("map_all", sorted(list(map_all)))
    print("age_all", sorted(list(age_all)))

    map_with_inds = list(enumerate(map_all))
    print(len(map_all))

    map_with_inds.sort(key=lambda x: x[1])
    print("map_all sorted", map_with_inds)
    num_folds = 5  # 3/1/1, bucket sizes: 12x5, 1x4
    bucket_sizes = [5, 5, 5, 5, 5, 5, 4, 5, 5, 5, 5, 5, 5]
    folds = {i: [] for i in range(num_folds)}
    i = 0
    bucket = 0
    while i < len(map_with_inds):
        bucket_size = bucket_sizes[bucket]
        inds = np.random.permutation(bucket_size)
        for fold, j in enumerate(inds):
            folds[fold].append(map_with_inds[i + j])
        print(inds)
        i += bucket_size
        bucket += 1
        print(i, bucket)

    fold_indices = {}
    for i in range(len(folds)):
        fold_indices[i] = sorted([folds[i][j][0] for j in range(len(folds[i]))])

    print("indices", fold_indices)

    print("maps")
    for i in range(len(folds)):
        maps = sorted([round(folds[i][j][1], 1) for j in range(len(folds[i]))])
        print(f"fold {i}", np.mean(maps), maps)

    print("ages")
    for i in range(len(folds)):
        ages = sorted([age_all[folds[i][j][0]] for j in range(len(folds[i]))])
        print(f"fold {i}", np.mean(ages), ages)
    return fold_indices
