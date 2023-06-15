from sklearn.model_selection import train_test_split


def split_array_in_train_test_val(array, train_ratio: float, val_ratio: float, test_ratio: float,
                                  random_state: int = 42):
    assert train_ratio + val_ratio + test_ratio == 1.0

    # split in train and (val, test) data
    train, val_and_test = train_test_split(array, test_size=val_ratio+test_ratio, random_state=random_state)

    # computation of test_ratio on the remaining data
    remaining_test_ratio = test_ratio / (val_ratio + test_ratio)

    # split (val, test) in val and test
    val, test = train_test_split(val_and_test, test_size=remaining_test_ratio, random_state=random_state)

    return train, val, test
