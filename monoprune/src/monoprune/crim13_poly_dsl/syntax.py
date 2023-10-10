from monoprune.env import *
from monoprune.crim13_paper_dsl.syntax import (
    CrimFeatureName as CrimFeatureName,
    crim_feature_name_len as crim_feature_name_len,
)

F = TypeVar("F")
Monomial: TypeAlias = Tuple[F, ...]
PolynomialSketch: TypeAlias = Tuple[Monomial[F], ...]
Polynomial: TypeAlias = Tuple[PolynomialSketch[F], Tuple[T, ...]]

IfRefOrF: TypeAlias = Union[
    Tuple[Literal["if_ref"], int],
    Tuple[Literal["f"], F],
]
FoldRefOrIfRefOrF: TypeAlias = Union[
    Tuple[Literal["if_ref"], int],
    Tuple[Literal["fold_ref"]],
    Tuple[Literal["f"], F],
]
ExprAtomToAtom: TypeAlias = Tuple[
    Tuple["ExprAtomToAtom[F, T]", ...],
    Polynomial[IfRefOrF[F], T],
]

FoldExprAtomToAtom: TypeAlias = Tuple[
    Tuple["FoldExprAtomToAtom[F, T]", ...],
    Polynomial[FoldRefOrIfRefOrF[F], T],
]

ExprListToAtom: TypeAlias = Union[
    Tuple[Literal["fold"], Tuple[FoldExprAtomToAtom[F, T]]],
    Tuple[
        Literal["ite"],
        Tuple["ExprListToAtom[F, T]", "ExprListToAtom[F, T]", "ExprListToAtom[F, T]"],
    ],
    # Tuple[Literal["average_last_5"], Tuple["ExprListToAtom[F, T]"]],
    # TODO: other averagers
]

ExprListToList: TypeAlias = Union[
    Tuple[Literal["map"], Tuple[ExprAtomToAtom[F, T]]],
    Tuple[Literal["map_prefixes"], Tuple[ExprListToAtom[F, T]]],
    Tuple[
        Literal["ite"],
        Tuple[ExprListToAtom[F, T], "ExprListToList[F, T]", "ExprListToList[F, T]"],
    ],
]

""" DSL_DICT = {('list', 'list') : [dsl.MapFunction, dsl.MapPrefixesFunction, dsl.SimpleITE],
                        ('list', 'atom') : [dsl.FoldFunction, dsl.running_averages.RunningAverageLast5Function, dsl.SimpleITE, 
                                            dsl.running_averages.RunningAverageLast10Function, dsl.running_averages.RunningAverageWindow11Function,
                                            dsl.running_averages.RunningAverageWindow5Function],
                        ('atom', 'atom') : [dsl.SimpleITE, dsl.AddFunction, dsl.MultiplyFunction, dsl.crim13.Crim13PositionSelection, 
                                            dsl.crim13.Crim13DistanceSelection, dsl.crim13.Crim13DistanceChangeSelection,
                                            dsl.crim13.Crim13VelocitySelection, dsl.crim13.Crim13AccelerationSelection,
                                            dsl.crim13.Crim13AngleSelection, dsl.crim13.Crim13AngleChangeSelection]} """

# def add_param_indices(
#     ps: PolynomialSketch[CrimFeatureName], prev_map: Tuple[None, ...]
# ) -> Tuple[PolynomialIndexed[CrimFeatureName], Tuple[None, ...]]:
#     new_e = (ps, tuple(i for i in range(len(prev_map), len(prev_map) + len(ps))))
#     return new_e, prev_map + (None,) * len(ps)


def add_param_indices_atom_to_atom(
    outline: ExprAtomToAtom[F, T], prev_map: Tuple[T, ...]
) -> Tuple[ExprAtomToAtom[F, int], Tuple[T, ...]]:
    cur_map = prev_map
    new_subexprs: List[ExprAtomToAtom[F, int]] = []
    for subexpr in outline[0]:
        new_subexpr, cur_map = add_param_indices_atom_to_atom(subexpr, cur_map)
        new_subexprs.append(new_subexpr)

    sketch, params = outline[1]
    new_params = tuple(i for i in range(len(cur_map), len(cur_map) + len(params)))
    cur_map = cur_map + params
    new_expr = tuple(new_subexprs), (sketch, new_params)
    return new_expr, cur_map


def add_param_indices_list_to_atom(
    outline: ExprListToAtom[F, T], prev_map: Tuple[T, ...]
) -> Tuple[ExprListToAtom[F, int], Tuple[T, ...]]:
    cur_map = prev_map

    match outline:
        case "fold", (outline_fold_fun,):
            new_fold_fun, cur_map = add_param_indices_atom_to_atom(
                outline_fold_fun, cur_map
            )
            ret = ("fold", (new_fold_fun,))
        case "ite", (outline_c, outline_a, outline_b):
            new_c, cur_map = add_param_indices_list_to_atom(outline_c, cur_map)
            new_a, cur_map = add_param_indices_list_to_atom(outline_a, cur_map)
            new_b, cur_map = add_param_indices_list_to_atom(outline_b, cur_map)
            ret = ("ite", (new_c, new_a, new_b))

    return ret, cur_map


def add_param_indices_list_to_list(
    outline: ExprListToList[F, T], prev_map: Tuple[T, ...]
) -> Tuple[ExprListToList[F, int], Tuple[T, ...]]:
    cur_map = prev_map

    match outline:
        case "map", (outline_map_fun,):
            new_fold_fun, cur_map = add_param_indices_atom_to_atom(
                outline_map_fun, cur_map
            )
            ret = ("map", (new_fold_fun,))
        case "map_prefixes", (outline_map_prefixes_fun,):
            new_fold_fun, cur_map = add_param_indices_list_to_atom(
                outline_map_prefixes_fun, cur_map
            )
            ret = ("map_prefixes", (new_fold_fun,))
        case "ite", (outline_c, outline_a, outline_b):
            new_c, cur_map = add_param_indices_list_to_atom(outline_c, cur_map)
            new_a, cur_map = add_param_indices_list_to_list(outline_a, cur_map)
            new_b, cur_map = add_param_indices_list_to_list(outline_b, cur_map)
            ret = ("ite", (new_c, new_a, new_b))

    return ret, cur_map


def mangle_poly_feature_names(
    has_fold_var: bool,
    n_conditionals: int,
    e: Polynomial[FoldRefOrIfRefOrF[CrimFeatureName], int],
) -> Polynomial[int, int]:
    n_fold_vars = 1 if has_fold_var else 0

    def replace(f: FoldRefOrIfRefOrF[CrimFeatureName]) -> int:
        match f:
            case "fold_ref",:
                assert has_fold_var
                return 0
            case "if_ref", i:
                return i + n_fold_vars
            case "f", i:
                assert i < crim_feature_name_len
                return i + n_conditionals + n_fold_vars

    new_sketch = tuple(tuple(replace(f) for f in monomial) for monomial in e[0])
    return (new_sketch, e[1])
