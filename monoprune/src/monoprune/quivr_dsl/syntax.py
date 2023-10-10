from monoprune.env import *

QuivrExpr: TypeAlias = Union[
    Tuple[Literal["pred"], Tuple[str, T]],
    Tuple[Literal["seq"], Tuple["QuivrExpr[T]", ...]],
    Tuple[Literal["conj"], Tuple["QuivrExpr[T]", ...]],
]
# tentatively we're not splitting out pred0 and pred1 because we can
# intantiate T with a tuple of indices

QuivrExprIndexed: TypeAlias = QuivrExpr[Tuple[int, ...]]
IndexPredMap: TypeAlias = Tuple[Tuple[str, int], ...]


def add_param_indices(
    pred_arities: Dict[str, int], e: QuivrExpr[None], prev_map: IndexPredMap
) -> Tuple[QuivrExprIndexed, IndexPredMap]:
    match e:
        case "pred", (pred_name, _data):
            new_indices = tuple(range(len(prev_map), len(prev_map) + pred_arities[pred_name]))
            new_expr = ("pred", (pred_name, new_indices))
            addl_map = tuple((pred_name, i) for i in range(pred_arities[pred_name]))
            return new_expr, prev_map + addl_map
        case "seq", subes:
            new_subes : List[QuivrExprIndexed] = []
            cur_map = prev_map
            for sube in subes:
                new_sube, cur_map = add_param_indices(pred_arities, sube, cur_map)
                new_subes.append(new_sube)

            new_e = ("seq", tuple(new_subes))
            return new_e, cur_map
        case "conj", subes:
            new_subes : List[QuivrExprIndexed] = []
            cur_map = prev_map
            for sube in subes:
                new_sube, cur_map = add_param_indices(pred_arities, sube, cur_map)
                new_subes.append(new_sube)

            new_e = ("conj", tuple(new_subes))
            return new_e, cur_map

if False:
    add_param_indices({
        "a": 0,
        "b" : 1,
        "c": 2    
    },
    ("seq", (("pred", ("b", None)), ("conj", (
        ("pred", ("a", None)),
        ("pred", ("c", None))
    )))),
    (),
    )