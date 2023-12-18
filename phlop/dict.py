#
#
#
#
#


# pylintrc ignored no-member
class ValDict(dict):
    def __init__(self, **d):
        for k, v in d.items():
            object.__setattr__(self, k, v)
