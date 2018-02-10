import os.path as osp

import os
import importlib


def load_data_module(path):
    '''
    Loads a data module at the given path.
    Can actually be used to load any module, but was designed for loading data modules from
    the variables in the config, easier.
    '''
    if not osp.exists(path):
        raise ValueError(f'path "{path}" does not exist.')

    path = path.replace('/', '.').replace('\\', '.') + '.data'
    module = importlib.import_module(path)

    return module


def get_latest_trained_data(path):
    '''
    Returns a list of the latest trained data in a directory, descending by time.
    '''
    if not osp.exists(path):
        raise ValueError(f'path "{path}" does not exist.')

    files = [x for x in os.scandir(path) if x.is_file() and x.name.endswith('.npz')]
    times = [x.stat().st_mtime for x in files]
    pairs = sorted(list(zip([f'{path}/{x.name}' for x in files], times)), key=lambda x: x[1], reverse=True)

    return pairs


class AttrObj:
    """
    Object that takes in a dict, and converts it to a dot format thing.
    Makes accessing the config a tad easier.
    """
    def __init__(self, d: dict):
        for k, v in d.items():
            # Recursively convert child dicts to AttrObjs.
            if type(v) == dict:
                d[k] = AttrObj(v)

        self.__dict__ = d

    def to_dict(self):
        """
        Transforms the AttrObj back into a dictionary, including all nested AttrObjs.
        """
        res = {}

        for k, v in self.__dict__.items():
            if isinstance(v, AttrObj):
                res[k] = v.to_dict()
            else:
                res[k] = v

        return res
