from monoprune.env import *
from monoprune.aexpr.symbolic_syntax import (
    BoolSymbolic,
    IntSymbolic,
    RatSymbolic,
    symbolic_bool_lit,
    symbolic_int_lit,
    symbolic_int_var,
    symbolic_rat_lit,
    symbolic_rat_var,
)
from monoprune.aexpr.syntax import BoolTerm, IntTerm, RatTerm
from monoprune.unique import unique, Unique
from monoprune.interval import Interval
import monoprune.exp_synth_param.quivr_data as quivr_data
import torch

T = TypeVar("T")
Param: TypeAlias = Tuple[RatSymbolic, RatSymbolic]
Vec: TypeAlias = Tuple[T, ...]
Mat: TypeAlias = Vec[Vec[T]]


def conj_all(vec: Vec[BoolSymbolic]) -> BoolSymbolic:
    if len(vec) == 0:
        return symbolic_bool_lit(True)
    else:
        return vec[0] & conj_all(vec[1:])


def disj_all(vec: Vec[BoolSymbolic]) -> BoolSymbolic:
    if len(vec) == 0:
        return symbolic_bool_lit(False)
    else:
        return vec[0] | disj_all(vec[1:])


def holds_continuously(vec: Vec[BoolSymbolic]) -> Mat[BoolSymbolic]:
    return tuple(
        tuple(symbolic_bool_lit(False) for _j in range(0, i))
        + tuple(conj_all(vec[i : j + 1]) for j in range(i, len(vec)))
        for i in range(len(vec))
    )


# computes matmul of mat0 and mat1 and then takes the top right entry
def sequence_two(mat0: Mat[BoolSymbolic], mat1: Mat[BoolSymbolic]) -> BoolSymbolic:
    n = len(mat0)
    return disj_all(tuple(mat0[0][k] & mat1[k][n - 1] for k in range(n)))


def matvec_mul(mat0: Mat[BoolSymbolic], vec1: Vec[BoolSymbolic]) -> Vec[BoolSymbolic]:
    n = len(mat0)
    return tuple(
        disj_all(tuple(mat0[i][j] & vec1[j] for j in range(n))) for i in range(n)
    )


from monoprune.stat_metrics import f1_sb


def symbolify_vec(v: Vec[BoolTerm]) -> Vec[BoolSymbolic]:
    return tuple(BoolSymbolic(c) for c in v)


def desymbolify_vec(v: Vec[BoolSymbolic]) -> Vec[BoolTerm]:
    return tuple(c.t for c in v)


def symbolify_mat(m: Mat[BoolTerm]) -> Mat[BoolSymbolic]:
    return tuple(symbolify_vec(v) for v in m)


def desymbolify_mat(m: Mat[BoolSymbolic]) -> Mat[BoolTerm]:
    return tuple(desymbolify_vec(v) for v in m)


if False:
    m1 = tuple(
        [
            tuple([sf, st, sf]),
            tuple([sf, sf, sf]),
            tuple([sf, sf, st]),
        ]
    )
    m2 = tuple(
        [
            tuple([sf, sf, sf]),
            tuple([sf, st, sf]),
            tuple([sf, st, sf]),
        ]
    )
    v3 = tuple([sf, st, st])
    e: QuivrExpr = ("seq", (("pred", ("a",)), ("pred", ("b",))))
    r = eval_quivr_expr({"a": m1, "b": m2}, e)

    from monoprune.aexpr.semantics_concrete import semantics_concrete

    semantics_concrete(None, None, None)[0](r)


# ('SEQ', (('PRED1', '+y', 0), ('PRED1', '-x', 1)))
# y >= param[0]; -x >= param[1]
#
# def eval_sketch_a(param: Param, trace: Mat[rat]):
#     pred_0_pointwise = tuple(x[1] >= param[0] for x in trace)
#     pred_1_pointwise = tuple(x[0] <= param[1] for x in trace)
#     pred_0_intervalwise = holds_continuously(pred_0_pointwise)
#     pred_1_intervalwise = holds_continuously(pred_1_pointwise)
#     return sequence_two(pred_0_intervalwise, pred_1_intervalwise)
#
#
# def eval_sketch_b(param: Param, trace: Mat[rat]):
#     pred_0_pointwise = tuple(x[1] <= param[0] for x in trace)
#     pred_1_pointwise = tuple(x[0] <= param[1] for x in trace)
#     pred_0_intervalwise = holds_continuously(pred_0_pointwise)
#     pred_1_intervalwise = holds_continuously(pred_1_pointwise)
#     return sequence_two(pred_0_intervalwise, pred_1_intervalwise)


Trace = Mat[rat]
QuivrExpr: TypeAlias = Union[
    Tuple[Literal["pred"], Tuple[str, T]],
    Tuple[Literal["seq"], Tuple["QuivrExpr", "QuivrExpr"]],
]

PredDens: TypeAlias = Tuple[int, Dict[str, Callable[[RatSymbolic], Mat[BoolSymbolic]]]]


def add_names(e: QuivrExpr[None]) -> Tuple[QuivrExpr[Unique], Dict[Unique, str]]:
    match e:
        case "pred", (pred_name, _data):
            ident = unique(pred_name)
            return (("pred", (pred_name, ident)), {ident: pred_name})
        case "seq", (l_e, r_e):
            l_e, l_names = add_names(l_e)  # type: ignore
            r_e, r_names = add_names(r_e)  # type: ignore
            return (
                ("seq", (l_e, r_e)),  # type: ignore
                l_names | r_names,
            )


def eval_quivr_expr_multi(
    pred_dens: PredDens,
    parameters: Dict[Unique, RatSymbolic],
) -> Callable[[QuivrExpr[Unique], Vec[BoolSymbolic]], Vec[BoolSymbolic]]:
    def rec(e: QuivrExpr[Unique], v: Vec[BoolSymbolic]) -> Vec[BoolSymbolic]:
        match e:
            case "pred", (pred_name, param):
                return matvec_mul(pred_dens[1][pred_name](parameters[param]), v)
            case "seq", (l_e, r_e):
                r_v = rec(r_e, v)  # type: ignore
                l_v = rec(l_e, r_v)  # type: ignore
                return l_v

    return rec


st = symbolic_bool_lit(True)
sf = symbolic_bool_lit(False)


def eval_quivr_expr(
    pred_dens: PredDens, parameters: Dict[Unique, RatSymbolic], e: QuivrExpr[Unique]
) -> BoolSymbolic:
    n = pred_dens[0]
    initial_vec: Tuple[BoolSymbolic] = (sf,) * (n - 1) + (st,)
    final_vec = eval_quivr_expr_multi(pred_dens, parameters)(e, initial_vec)
    return final_vec[0]


pred_dens: Callable[[Trace], PredDens] = lambda trace: (
    len(trace),
    {
        "x_gt": lambda param: holds_continuously(tuple(x[0] >= param for x in trace)),
        "x_lt": lambda param: holds_continuously(tuple(x[0] <= param for x in trace)),
        "y_gt": lambda param: holds_continuously(tuple(x[1] >= param for x in trace)),
        "y_lt": lambda param: holds_continuously(tuple(x[1] <= param for x in trace)),
        # "any": lambda param: None,
    },
)

import itertools

pred_names: Tuple[str] = tuple(pred_dens(())[1].keys())
# quivr_sketches_one: Dict[str, QuivrExpr[None]] = {
#     f"{n}": ("pred", (n, None)) for n in pred_names
# }
quivr_sketches_two: Dict[str, QuivrExpr[None]] = {
    f"{l};{r}": ("seq", (("pred", (l, None)), ("pred", (r, None))))
    for l, r in tuple(itertools.permutations(pred_names, 2))
}

quivr_sketches_special: Dict[str, QuivrExpr[None]] = {
    # we used this previously, since it's the best in the size 2 set of sketches, but it gets f1 score 1, which may bias in favor of intervals over smt
    # "x_gt;x_lt": ("seq", (("pred", ("x_gt", None)), ("pred", ("x_lt", None)))),
    # the first size 2 query we tried that didn't get f1 score 1 for seed 0 size 6
    # "x_gt;x_gt": ("seq", (("pred", ("x_gt", None)), ("pred", ("x_gt", None)))),
    # "foobar": (
    #     "seq",
    #     (
    #         (
    #             "seq",
    #             (
    #                 ("pred", ("anything", None)),
    #                 ("pred", ("distance_atmost", None)),
    #             ),
    #         ),
    #         (
    #             "seq",
    #             (
    #                 ("pred", ("cont_obj0_speed_always_atmost", None)),
    #                 ("pred", ("anything", None)),
    #             ),
    #         ),
    #     ),
    # ),
    "foobar": (
        "seq",
        (
            ("pred", ("+x", None)),
            ("pred", ("+x", None)),
        ),
    ),
}
quivr_sketches: Dict[str, QuivrExpr[None]] = (
    # quivr_sketches_one |
    # quivr_sketches_two
    quivr_sketches_special
)


def get_train(dataset_seed: int):
    assert dataset_seed == 0
    # dataset_name = "monoprune_mabe22"
    # dataset_task = "approach"
    dataset_name = "maritime_surveillance"
    dataset_task = "a"
    dataset_data = quivr_data.load_data(dataset_name)
    dataset_labels = quivr_data.load_labels(dataset_name, dataset_task)
    return dataset_data, dataset_labels


def get_util_exprs(dataset_size: int, dataset_seed: int):
    dataset_data, dataset_labels = get_train(dataset_seed)

    n = dataset_data["traces"]["anything"].shape[-1]
    pred_bounds = {
        k: Interval(rat(v[0]), rat(v[1]))
        for k, v in dataset_data["pred1_bounds"].items()
    }
    pred0_names = {"anything", "nothing"}
    train_y = tuple(dataset_labels["labels"][i] for i in range(dataset_size))

    def quivr_sketch_to_sketch(
        quivr_sketch: QuivrExpr[None],
    ) -> Tuple[RatTerm, Dict[Unique, Interval[rat]]]:
        quivr_expr, param_idents = add_names(quivr_sketch)
        param_symbols = {
            param_ident: symbolic_rat_var(param_ident)
            if param_name not in pred0_names
            else None
            for param_ident, param_name in param_idents.items()
        }
        param_bounds = {
            param_ident: pred_bounds[param_name]
            for param_ident, param_name in param_idents.items()
            if param_name not in pred0_names
        }

        traces_symbolic = {
            k: tuple(
                tuple(
                    tuple(
                        symbolic_rat_lit(rat.from_float(max(x, -100000)))
                        if type(x) == float
                        else symbolic_bool_lit(x)
                        for j in range(n)
                        for x in [dataset_data["traces"][k][ex, i, j].item()]
                    )
                    for i in range(n)
                )
                for ex in range(dataset_size)
            )
            for k, v in dataset_data["traces"].items()
        }

        # print("=== traces symbolic ===")
        # for k, v in traces_symbolic.items():
        #     print(k)
        #     print(v[0][0])

        def eval_sketch(ex: int) -> BoolSymbolic:
            def pred0_den(k):
                def r(thresh):
                    assert thresh is None
                    return traces_symbolic[k][ex]

                return r

            def pred1_den(k):
                def r(thresh):
                    return tuple(
                        tuple(traces_symbolic[k][ex][i][j] >= thresh for j in range(n))
                        for i in range(n)
                    )

                return r

            dens = (
                n,
                {
                    k: pred0_den(k) if v.dtype == torch.bool else pred1_den(k)
                    for k, v in dataset_data["traces"].items()
                },
            )
            return eval_quivr_expr(dens, param_symbols, quivr_expr)

        preds = tuple(eval_sketch(i) for i in range(dataset_size))
        util_expr = f1_sb(train_y, preds).t
        return util_expr, param_bounds

    from tqdm import tqdm

    util_exprs: Dict[str, Tuple[RatTerm, Dict[Unique, Interval[rat]]]] = {
        k: quivr_sketch_to_sketch(s) for k, s in tqdm(quivr_sketches.items())
    }
    return util_exprs

    # util_exps = {
    #     "b": (util(eval_sketch_b, naval_data, naval_labels, param).t, param_bounds),
    #     "a": (util(eval_sketch_a, naval_data, naval_labels, param).t, param_bounds),
    # }


state_util_bounds = Interval(rat(0, 1), rat(1, 1))
