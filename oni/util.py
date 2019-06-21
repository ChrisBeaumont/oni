def as_list(x):
    try:
        return list(x)
    except TypeError:
        return [x]