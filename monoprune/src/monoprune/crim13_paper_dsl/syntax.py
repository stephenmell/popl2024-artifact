from monoprune.env import *
from monoprune.type_grammar_tools import cfg_prods_from_type, cfg_prods_from_type_simple

CrimFeatureName: TypeAlias = Union[
    Literal[0],
    Literal[1],
    Literal[2],
    Literal[3],
    Literal[4],
    Literal[5],
    Literal[6],
    Literal[7],
    Literal[8],
    Literal[9],
    Literal[10],
    Literal[11],
    Literal[12],
    Literal[13],
    Literal[14],
    Literal[15],
    Literal[16],
    Literal[17],
    Literal[18],
]
crim_feature_name_len = 19

CrimExprAtomAtom: TypeAlias = Union[
    Tuple[
        Literal["ite"],
        Tuple["CrimExprAtomAtom[T]", "CrimExprAtomAtom[T]", "CrimExprAtomAtom[T]"],
    ],
    Tuple[Literal["add"], Tuple["CrimExprAtomAtom[T]", "CrimExprAtomAtom[T]"]],
    Tuple[Literal["mul"], Tuple["CrimExprAtomAtom[T]", "CrimExprAtomAtom[T]"]],
    Tuple[Literal["sel"], Tuple["CrimFeatureName"]],
    Tuple[Literal["lit"], Tuple[T]],
]

grammar = {
    "CrimExprAtomAtom": cfg_prods_from_type(CrimExprAtomAtom),
    "CrimFeatureName": cfg_prods_from_type_simple(CrimFeatureName),
    "T": {"param": ()},
}

CrimExprAtomAtomIndexed: TypeAlias = CrimExprAtomAtom[int]
IndexPredMap: TypeAlias = Tuple[T, ...]


def atom_atom_add_param_indices(
    e: CrimExprAtomAtom[T],
    prev_map: IndexPredMap[T],
) -> Tuple[CrimExprAtomAtomIndexed, IndexPredMap[T]]:
    match e:
        case "lit", (data,):
            new_index = len(prev_map)
            new_expr = ("lit", (new_index,))
            return new_expr, prev_map + (data,)
        case "sel", (feature_name,):
            return ("sel", (feature_name,)), prev_map
        case "add", subes:
            new_subes: List[CrimExprAtomAtomIndexed] = []
            cur_map = prev_map
            for sube in subes:
                new_sube, cur_map = atom_atom_add_param_indices(sube, cur_map)
                new_subes.append(new_sube)

            new_e = ("add", tuple(new_subes))
            return new_e, cur_map
        case "mul", subes:
            new_subes: List[CrimExprAtomAtomIndexed] = []
            cur_map = prev_map
            for sube in subes:
                new_sube, cur_map = atom_atom_add_param_indices(sube, cur_map)
                new_subes.append(new_sube)

            new_e = ("mul", tuple(new_subes))
            return new_e, cur_map
        case "ite", subes:
            new_subes: List[CrimExprAtomAtomIndexed] = []
            cur_map = prev_map
            for sube in subes:
                new_sube, cur_map = atom_atom_add_param_indices(sube, cur_map)
                new_subes.append(new_sube)

            new_e = ("ite", tuple(new_subes))
            return new_e, cur_map
