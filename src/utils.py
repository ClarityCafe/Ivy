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