import string


def cify_name(name: str) -> str:
    forbidden = string.punctuation + string.whitespace
    return name.translate(str.maketrans(forbidden, '_' * len(forbidden)))
