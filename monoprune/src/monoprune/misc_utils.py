from monoprune.env import *

def list_to_bag(l: Tuple[T, ...]) -> Dict[T, int]:
    ret: Dict[T, int] = dict()
    for x in l:
        if x not in ret:
            ret[x] = 0
        ret[x] += 1
    return ret
