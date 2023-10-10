from monoprune.env import *
from monoprune.unique import Unique
from monoprune.aexpr.syntax import BoolTerm, IntTerm, RatTerm  # type: ignore
from monoprune.interval import *

from monoprune.aexpr.semantics_concrete import semantics_concrete  # type: ignore
from monoprune.aexpr.semantics_interval import semantics_interval  # type: ignore
import random


def _empty_dict_fun(k: Any) -> Any:
    assert False


def semantics_interval_sampling(
    ctx_bool: Callable[[Unique], Interval[bool]] = _empty_dict_fun,
    ctx_int: Callable[[Unique], Interval[int]] = _empty_dict_fun,
    ctx_rat: Callable[[Unique], Interval[rat]] = _empty_dict_fun,
    rand: Optional[random.Random] = None,
) -> Tuple[
    Callable[[BoolTerm], bool],
    Callable[[IntTerm], int],
    Callable[[RatTerm], rat],
]:
    if rand is None:
        rand = random.Random()

    def sample_bool(k: Unique) -> bool:
        interval = ctx_bool(k)
        return rand.randint(int(interval.lower), int(interval.upper)) == 1

    def sample_int(k: Unique) -> int:
        interval = ctx_int(k)
        return rand.randint(interval.lower, interval.upper)

    def sample_rat(k: Unique) -> rat:
        interval = ctx_rat(k)
        ret = rat.from_float(rand.uniform(float(interval.lower), float(interval.upper)))
        assert ret >= interval.lower and ret <= interval.upper
        return ret

    (
        eval_sample_bool,
        eval_sample_int,
        eval_sample_rat,
    ) = semantics_concrete(sample_bool, sample_int, sample_rat)

    eval_interval_bool, eval_interval_int, eval_interval_rat = semantics_interval(
        ctx_bool, ctx_int, ctx_rat
    )

    def eval_bool(e: BoolTerm) -> bool:
        interval = eval_interval_bool(e)
        sample = eval_sample_bool(e)
        assert sample >= interval.lower and sample <= interval.upper, (sample, interval)
        return sample

    def eval_int(e: IntTerm) -> int:
        interval = eval_interval_int(e)
        sample = eval_sample_int(e)
        assert sample >= interval.lower and sample <= interval.upper, (sample, interval)
        return sample

    def eval_rat(e: RatTerm) -> rat:
        interval = eval_interval_rat(e)
        sample = eval_sample_rat(e)
        assert sample >= interval.lower and sample <= interval.upper, (sample, interval)
        return sample

    return eval_bool, eval_int, eval_rat
