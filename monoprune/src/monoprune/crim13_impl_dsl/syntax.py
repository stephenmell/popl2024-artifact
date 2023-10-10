from monoprune.env import *

from monoprune.type_grammar_tools import cfg_prods_from_type, cfg_prods_from_type_simple

"""
DSL_DICT = {('list', 'list') : [dsl.MapFunction, dsl.MapPrefixesFunction, dsl.SimpleITE],
                        ('list', 'atom') : [dsl.FoldFunction, dsl.running_averages.RunningAverageLast5Function, dsl.SimpleITE, 
                                            dsl.running_averages.RunningAverageLast10Function, dsl.running_averages.RunningAverageWindow11Function,
                                            dsl.running_averages.RunningAverageWindow5Function],
                        ('atom', 'atom') : [dsl.SimpleITE, dsl.AddFunction, dsl.MultiplyFunction, dsl.crim13.Crim13PositionSelection, 
                                            dsl.crim13.Crim13DistanceSelection, dsl.crim13.Crim13DistanceChangeSelection,
                                            dsl.crim13.Crim13VelocitySelection, dsl.crim13.Crim13AccelerationSelection,
                                            dsl.crim13.Crim13AngleSelection, dsl.crim13.Crim13AngleChangeSelection]}

"""


CrimFeatureName: TypeAlias = Union[
    Literal["position"],
    Literal["distance"],
    Literal["distance_change"],
    Literal["velocity"],
    Literal["acceleration"],
    Literal["angle"],
    Literal["angle_change"],
]

CrimExprAtomAtom: TypeAlias = Union[
    Tuple[
        Literal["ite"],
        Tuple["CrimExprAtomAtom[T]", "CrimExprAtomAtom[T]", "CrimExprAtomAtom[T]"],
    ],
    Tuple[Literal["add"], Tuple["CrimExprAtomAtom[T]", "CrimExprAtomAtom[T]"]],
    Tuple[Literal["mul"], Tuple["CrimExprAtomAtom[T]", "CrimExprAtomAtom[T]"]],
    Tuple[Literal["sel"], Tuple[CrimFeatureName, T]],
]

# TODO: extract manually
# grammar = {
#     "CrimExprAtomAtom": cfg_prods_from_type(CrimExprAtomAtom),
#     "CrimFeatureName": cfg_prods_from_type_simple(CrimFeatureName),
#     "T": {"param": ()},
# }

grammar = {
    "CrimExprAtomAtom": {
        "ite": ("CrimExprAtomAtom", "CrimExprAtomAtom", "CrimExprAtomAtom"),
        "add": ("CrimExprAtomAtom", "CrimExprAtomAtom"),
        "mul": ("CrimExprAtomAtom", "CrimExprAtomAtom"),
        "sel": ("CrimFeatureName",),
    },
    "CrimFeatureName": {
        "position": (),
        "distance": (),
        "distance_change": (),
        "velocity": (),
        "acceleration": (),
        "angle": (),
        "angle_change": (),
    },
    "T": {"param": ()},
}


CrimExprAtomAtomIndexed: TypeAlias = CrimExprAtomAtom[Tuple[int, ...]]
IndexPredMap: TypeAlias = Tuple[Tuple[str, int], ...]


def atom_atom_add_param_indices(
    pred_arities: Dict[str, int],
    output_size: int,
    e: CrimExprAtomAtom[None],
    prev_map: IndexPredMap,
) -> Tuple[CrimExprAtomAtomIndexed, IndexPredMap]:
    assert output_size == 1
    match e:
        case "sel", (feat_name, _data):
            new_indices = tuple(
                range(len(prev_map), len(prev_map) + pred_arities[feat_name])
            )
            new_expr = ("sel", (feat_name, new_indices))
            addl_map = tuple((feat_name, i) for i in range(pred_arities[feat_name]))
            return new_expr, prev_map + addl_map
        case "add", subes:
            new_subes: List[CrimExprAtomAtomIndexed] = []
            cur_map = prev_map
            for sube in subes:
                new_sube, cur_map = add_param_indices(pred_arities, sube, cur_map)
                new_subes.append(new_sube)

            new_e = ("add", tuple(new_subes))
            return new_e, cur_map
        case "mul", subes:
            new_subes: List[CrimExprAtomAtomIndexed] = []
            cur_map = prev_map
            for sube in subes:
                new_sube, cur_map = add_param_indices(pred_arities, sube, cur_map)
                new_subes.append(new_sube)

            new_e = ("mul", tuple(new_subes))
            return new_e, cur_map
        case "ite", subes:
            new_subes: List[CrimExprAtomAtomIndexed] = []
            cur_map = prev_map
            for sube in subes:
                new_sube, cur_map = add_param_indices(pred_arities, sube, cur_map)
                new_subes.append(new_sube)

            new_e = ("ite", tuple(new_subes))
            return new_e, cur_map
