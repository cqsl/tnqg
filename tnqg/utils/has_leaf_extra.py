def has_leaf_extra(d):
    for key, value in d.items():
        if key == "extra":
            return True
    return False
