"""Append complete, explicitly labelled feature rows to a local CSV."""
import csv
import math
from pathlib import Path


def append_sample(path, label, features, *, feature_count, class_count):
    if not isinstance(label, int) or not 0 <= label < class_count:
        return False
    if len(features) != feature_count or not all(math.isfinite(v) for v in features):
        return False
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', newline='', encoding='utf-8') as stream:
        csv.writer(stream).writerow([label, *features])
    return True
