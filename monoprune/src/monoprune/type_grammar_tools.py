from monoprune.env import *

import types
import typing

def name_from_forward_ref(ty):
    assert type(ty) == typing.ForwardRef, ty
    s = ty.__forward_arg__
    if "[" in s:
        ss = s.split("[")
        assert len(ss) == 2, s
        assert ss[1][-1] == "]", s
        return ss[0]
    else:
        return s


def name_from_prod_arg(ty):
    if type(ty) == typing.ForwardRef:
        return name_from_forward_ref(ty)
    elif type(ty) == typing.TypeVar:
        return ty.__name__
    else:
        assert False, ty

CfgTypeName : TypeAlias = str
CfgProdName : TypeAlias = str
CfgType : TypeAlias = Dict[CfgProdName, Tuple[CfgTypeName, ...]]
Cfg : TypeAlias = Dict[CfgTypeName, CfgType]
CfgExpr : TypeAlias = Tuple[CfgProdName, Tuple["CfgExpr", ...]]
CfgExprPartial : TypeAlias = Union[
    Tuple[None, CfgTypeName],
    Tuple[CfgProdName, Tuple["CfgExprPartial", ...]]
]

def cfg_prods_from_type(ty: Any) -> CfgType:
    assert ty.__origin__ == typing.Union, ty
    ret = {}
    for ty_prod in ty.__args__:
        assert ty_prod.__origin__ == tuple, ty_prod
        ty_prod_name, ty_prod_args = ty_prod.__args__
        assert ty_prod_name.__origin__ == typing.Literal, ty_prod_name
        prod_name = ty_prod_name.__args__[0]
        assert ty_prod_args.__origin__ == tuple, ty_prod_args
        arg_types = tuple(
            name_from_prod_arg(ty_prod_arg) for ty_prod_arg in ty_prod_args.__args__
        )
        ret[prod_name] = arg_types
    return ret

def cfg_prods_from_type_simple(ty: Any) -> CfgType:
    assert ty.__origin__ == typing.Union, ty
    ret = {}
    for ty_prod_name in ty.__args__:
        assert ty_prod_name.__origin__ == typing.Literal, ty_prod_name
        prod_name = ty_prod_name.__args__[0]
        ret[prod_name] = ()
    return ret

def expand_first_hole(
    grammar: Cfg, partial_expr: CfgExprPartial
) -> Iterable[CfgExprPartial]:
    match partial_expr:
        case (None, ty):
            for op_id, op_arg_types in grammar[ty].items():
                new_args = tuple((None, ty2) for ty2 in op_arg_types)
                yield (op_id, new_args)

        case op_id, op_args:
            for i, arg in enumerate(op_args):
                for rec_arg in expand_first_hole(grammar, arg):
                    new_args = op_args[:i] + (rec_arg,) + op_args[i + 1 :]
                    yield (op_id, new_args)