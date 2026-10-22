"""Iteration helpers."""
from collections import OrderedDict


def group_by(items, key):
    groups = OrderedDict()
    for item in items:
        groups.setdefault(key(item), []).append(item)
    return groups


def chunked(items, size):
    chunk = []
    for item in items:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def pairwise(items):
    items = list(items)
    return list(zip(items, items[1:]))


def running_total(values):
    total, out = 0, []
    for value in values:
        total += value
        out.append(total)
    return out
