import os
import sys
import json
import numpy as np
from collections import defaultdict


def read_npz(paths, keys):
    res = defaultdict(list)
    for path in paths:
        data_dict = np.load(path)
        for key in keys:
            res[key].append(data_dict[key].tolist())
    return dict(res)


if __name__ == '__main__':
    dest_path = sys.argv[1]
    paths = []
    for cur_file in os.scandir(dest_path):
        if os.path.isfile(cur_file):
            if cur_file.name.split('.')[-1] == 'npz':
                paths.append(cur_file)
    with open(sys.argv[2], 'w') as f:
        f.write(json.dumps(read_npz(paths, ['q_value', 'v_value']), indent=4))
