from monoprune.env import *

@dataclass(frozen=True)
class Unique:
    name: str
    num: int


_last_unique : Dict[str, int] = {}


def unique(name: str = "") -> Unique:
    global _last_unique
    if name not in _last_unique:
        _last_unique[name] = 0
    ret = Unique(name, _last_unique[name])
    _last_unique[name] += 1
    return ret
