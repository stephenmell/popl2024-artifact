from monoprune.env import *
from monoprune.unique import Unique
from monoprune.aexpr.syntax import BoolTerm, IntTerm, RatTerm  # type: ignore


def _empty_dict_fun(k: Any) -> Any:
    assert False


def semantics_concrete(
    ctx_bool: Callable[[Unique], bool] = _empty_dict_fun,
    ctx_int: Callable[[Unique], int] = _empty_dict_fun,
    ctx_rat: Callable[[Unique], rat] = _empty_dict_fun,
) -> Tuple[
    Callable[[BoolTerm], bool], Callable[[IntTerm], int], Callable[[RatTerm], rat]
]:
    def eval_bool(e: BoolTerm) -> bool:
        match e:
            case "var", (var,):
                ret = ctx_bool(var)
            case "lit", (lit,):
                ret = lit
            case "inv", (a,):
                v_a = eval_bool(a)
                ret = not v_a
            case "and", (a, b):
                v_a = eval_bool(a)
                v_b = eval_bool(b)
                ret = v_a and v_b
            case "or", (a, b):
                v_a = eval_bool(a)
                v_b = eval_bool(b)
                ret = v_a or v_b
            case "ite", (c, a, b):
                v_c = eval_bool(c)
                v_a = eval_bool(a)
                v_b = eval_bool(b)
                ret = v_a if v_c else v_b
            case "lt", (a, b):
                v_a = eval_rat(a)
                v_b = eval_rat(b)
                ret = v_a < v_b
            case "le", (a, b):
                v_a = eval_rat(a)
                v_b = eval_rat(b)
                ret = v_a <= v_b
            case "gt", (a, b):
                v_a = eval_rat(a)
                v_b = eval_rat(b)
                ret = v_a > v_b
            case "ge", (a, b):
                v_a = eval_rat(a)
                v_b = eval_rat(b)
                ret = v_a >= v_b
            case "eq", (a, b):
                v_a = eval_rat(a)
                v_b = eval_rat(b)
                ret = v_a == v_b
        assert type(ret) == bool
        return ret

    def eval_int(e: IntTerm) -> int:
        match e:
            case "var", (var,):
                ret = ctx_int(var)
            case "lit", (lit,):
                ret = lit
            case "neg", (a,):
                v_a = eval_int(a)
                ret = -v_a
            case "add", (a, b):
                v_a = eval_int(a)
                v_b = eval_int(b)
                ret = v_a + v_b
            case "mul", (a, b):
                v_a = eval_int(a)
                v_b = eval_int(b)
                ret = v_a * v_b
            case "ite", (c, a, b):
                v_c = eval_bool(c)
                v_a = eval_int(a)
                v_b = eval_int(b)
                ret = v_a if v_c else v_b
        assert type(ret) == int
        return ret

    def eval_rat(e: RatTerm) -> rat:
        match e:
            case "var", (var,):
                ret = ctx_rat(var)
            case "lit", (lit,):
                ret = lit
            case "of_int", (a,):
                v_a = eval_int(a)
                ret = rat(v_a, 1)
            case "neg", (a,):
                v_a = eval_rat(a)
                ret = -v_a
            case "add", (a, b):
                v_a = eval_rat(a)
                v_b = eval_rat(b)
                ret = v_a + v_b
            case "mul", (a, b):
                v_a = eval_rat(a)
                v_b = eval_rat(b)
                ret = v_a * v_b
            case "truediv", (a, b):
                v_a = eval_rat(a)
                v_b = eval_rat(b)
                ret = v_a / v_b
            case "ite", (c, a, b):
                v_c = eval_bool(c)
                v_a = eval_rat(a)
                v_b = eval_rat(b)
                ret = v_a if v_c else v_b
            case "sb_right", (a,):
                v_a = eval_rat(a)
                ret = v_a / (1 + v_a)
        assert type(ret) == rat
        return ret

    return eval_bool, eval_int, eval_rat


if False:
    e: BoolTerm = (
        "and",
        (
            ("lit", (True,)),
            ("var", ("foob",)),
        ),
    )
    print(semantics_concrete(lambda _: True, lambda _: None, lambda _: None)[0](e))
