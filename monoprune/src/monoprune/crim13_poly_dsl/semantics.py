from monoprune.env import *

from monoprune.torch_intervals import mul_interval, pow_interval
from jaxtyping import Float, Bool, Int, Shaped
from torch import Tensor
import torch
from monoprune.crim13_poly_dsl.syntax import (
    Monomial,
    PolynomialSketch,
    Polynomial,
    FoldRefOrIfRefOrF,
    CrimFeatureName,
    FoldExprAtomToAtom,
    # ExprAtomToAtom,
    ExprListToAtom,
    ExprListToList,
    crim_feature_name_len,
    mangle_poly_feature_names,
)
from monoprune.misc_utils import list_to_bag

DO_DBG = True


def compute_monomial(
    monomial: Monomial[int],
    x: Float[Tensor, "#*b n"],
) -> Float[Tensor, "#*b"]:
    ret = torch.ones(x.shape[:-1], dtype=x.dtype, device=x.device)
    for feature_index, degree in list_to_bag(monomial).items():
        ret *= torch.pow(x[..., feature_index], degree)
    return ret


def augment_vector_for_polynomial(
    ps: PolynomialSketch[int],
    x: Float[Tensor, "#*b n"],
) -> Float[Tensor, "#*b p"]:
    assert len(ps) > 0
    return torch.stack(tuple(compute_monomial(monomial, x) for monomial in ps), -1)


def compute_monomial_interval(
    monomial: Monomial[int],
    x_lower: Float[Tensor, "#*b n"],
    x_upper: Float[Tensor, "#*b n"],
) -> Tuple[Float[Tensor, "#*b"], Float[Tensor, "#*b"]]:
    ret_lower = torch.ones(
        x_lower.shape[:-1], dtype=x_lower.dtype, device=x_lower.device
    )
    ret_upper = torch.ones(
        x_lower.shape[:-1], dtype=x_lower.dtype, device=x_lower.device
    )
    for feature_index, degree in list_to_bag(monomial).items():
        y_lower, y_upper = pow_interval(
            x_lower[..., feature_index], x_upper[..., feature_index], degree
        )
        ret_lower, ret_upper = mul_interval(ret_lower, ret_upper, y_lower, y_upper)
    return ret_lower, ret_upper


def augment_vector_for_polynomial_interval(
    ps: PolynomialSketch[int],
    x_lower: Float[Tensor, "#*b n"],
    x_upper: Float[Tensor, "#*b n"],
) -> Tuple[Float[Tensor, "#*b p"], Float[Tensor, "#*b p"]]:
    rets_lower: List[Float[Tensor, "#*b"]] = []
    rets_upper: List[Float[Tensor, "#*b"]] = []
    for monomial in ps:
        l, u = compute_monomial_interval(monomial, x_lower, x_upper)
        rets_lower.append(l)
        rets_upper.append(u)
    ret_lower = torch.stack(rets_lower, -1)
    ret_upper = torch.stack(rets_upper, -1)

    if DO_DBG:
        x_mid = (x_lower + x_upper) / 2
        if (x_mid == x_mid).all():
            ret_mid = augment_vector_for_polynomial(ps, x_mid)
            assert (ret_mid >= ret_lower).all()
            assert (ret_mid <= ret_upper).all()
    return ret_lower, ret_upper


if False:
    ps = ((), (0,), (0, 0), (0, 0, 0), (0, 0, 1))
    augment_vector_for_polynomial_interval(
        ps,
        torch.tensor([-2.0, 3]),
        torch.tensor([-1.0, 4]),
    )


def eval_poly(
    x: Float[Tensor, "#*b n"],
    all_params: Float[Tensor, "p"],
    p: Polynomial[int, int],
) -> Float[Tensor, "#*b"]:
    x_aug = augment_vector_for_polynomial(p[0], x)
    params = all_params[(p[1],)]
    return (x_aug * params).sum(-1)


def eval_poly_interval(
    x_lower: Float[Tensor, "#*b n"],
    x_upper: Float[Tensor, "#*b n"],
    all_params_lower: Float[Tensor, "p"],
    all_params_upper: Float[Tensor, "p"],
    p: Polynomial[int, int],
) -> Tuple[Float[Tensor, "#*b"], Float[Tensor, "#*b"]]:
    x_aug_lower, x_aug_upper = augment_vector_for_polynomial_interval(
        p[0], x_lower, x_upper
    )
    params_lower = all_params_lower[(p[1],)]
    params_upper = all_params_upper[(p[1],)]
    int_lower, int_upper = mul_interval(
        x_aug_lower, x_aug_upper, params_lower, params_upper
    )
    res_lower = int_lower.sum(-1)
    res_upper = int_upper.sum(-1)
    if DO_DBG:
        x_mid = (x_lower + x_upper) / 2
        all_params_mid = (all_params_lower + all_params_upper) / 2
        if (x_mid == x_mid).all() and (all_params_mid == all_params_mid).all():
            res_mid = eval_poly(x_mid, all_params_mid, p)
            assert (res_mid >= res_lower).all(), (res_mid >= res_lower).nonzero()
            assert (res_mid <= res_upper).all(), (res_mid <= res_upper).nonzero()
    return res_lower, res_upper


def eval_atom_to_atom(
    x: Float[Tensor, "#*b n"],
    fold_var: Optional[Float[Tensor, "#*b"]],
    params: Float[Tensor, "p"],
    e: FoldExprAtomToAtom[CrimFeatureName, int],
) -> Float[Tensor, "#*b"]:
    assert len(x.shape) in {2, 3}, x.shape
    assert fold_var is None or len(fold_var.shape) == len(x.shape) - 1, (
        fold_var.shape,
        x.shape,
    )
    assert len(params.shape) == 1

    all_inputs: List[Float[Tensor, "#*b p"]] = []
    if fold_var is not None:
        all_inputs.append(fold_var.unsqueeze(dim=-1))

    if len(e[0]) > 0:
        conditional_reals = tuple(
            eval_atom_to_atom(x, fold_var, params, p2) for p2 in e[0]
        )
        conditionals = (torch.stack(conditional_reals, -1) >= 0).float()
        all_inputs.append(conditionals)

    all_inputs.append(x)

    p_mangled = mangle_poly_feature_names(fold_var is not None, len(e[0]), e[1])
    return eval_poly(
        pad_shape_concat(all_inputs, -1),
        params,
        p_mangled,
    )


# this doesn't actually broadcast, it just pads shapes with 1s
def pad_shape_concat(ts: Sequence[Tensor], axis: int) -> Tensor:
    assert axis == -1
    shared_shape = max((len(t.shape), t.shape[:-1]) for t in ts)[1]
    ts_reshaped: List[Tensor] = []
    for t in ts:
        new_shape = shared_shape + (t.shape[-1],)
        shape_prefix_len = len(new_shape) - len(t.shape)
        assert new_shape[shape_prefix_len:] == t.shape, (new_shape, t.shape)
        broad_shape = (1,) * shape_prefix_len + t.shape
        ts_reshaped.append(t.reshape(broad_shape).expand(new_shape))
    return torch.concat(ts_reshaped, axis=axis)  # type: ignore


if False:
    pad_shape_concat(
        (
            torch.tensor(
                [
                    [
                        [
                            [0, 1, 3, 4],
                            [5, 6, 7, 8],
                        ]
                    ]
                ]
            ),
            torch.tensor([9, 10]),
        ),
        axis=-1,
    )


def eval_atom_to_atom_interval(
    x: Float[Tensor, "#*b n"],
    fold_var_lower: Optional[Float[Tensor, "#*b"]],
    fold_var_upper: Optional[Float[Tensor, "#*b"]],
    params_lower: Float[Tensor, "p"],
    params_upper: Float[Tensor, "p"],
    e: FoldExprAtomToAtom[CrimFeatureName, int],
) -> Tuple[Float[Tensor, "#*b"], Float[Tensor, "#*b"]]:
    assert len(x.shape) in {2, 3}, x.shape
    assert fold_var_lower is None or len(fold_var_lower.shape) == len(x.shape) - 1, (
        fold_var_lower.shape,
        x.shape,
    )
    assert fold_var_upper is None or len(fold_var_upper.shape) == len(x.shape) - 1, (
        fold_var_upper.shape,
        x.shape,
    )
    assert len(params_lower.shape) == 1
    assert len(params_upper.shape) == 1
    all_inputs_lower: List[Float[Tensor, "#*b p"]] = []
    all_inputs_upper: List[Float[Tensor, "#*b p"]] = []
    if fold_var_lower is not None:
        assert fold_var_upper is not None
        all_inputs_lower.append(fold_var_lower.unsqueeze(dim=-1))
        all_inputs_upper.append(fold_var_upper.unsqueeze(dim=-1))
    else:
        assert fold_var_upper is None

    if len(e[0]) > 0:
        conditional_reals_bounds = tuple(
            eval_atom_to_atom_interval(
                x, fold_var_lower, fold_var_upper, params_lower, params_upper, p2
            )
            for p2 in e[0]
        )
        conditionals_lower = (
            torch.stack(tuple(bounds[0] for bounds in conditional_reals_bounds), -1)
            >= 0
        ).float()
        conditionals_upper = (
            torch.stack(tuple(bounds[1] for bounds in conditional_reals_bounds), -1)
            >= 0
        ).float()
        all_inputs_lower.append(conditionals_lower)
        all_inputs_upper.append(conditionals_upper)

    all_inputs_lower.append(x)
    all_inputs_upper.append(x)
    p_mangled = mangle_poly_feature_names(fold_var_lower is not None, len(e[0]), e[1])
    return eval_poly_interval(
        pad_shape_concat(all_inputs_lower, -1),
        pad_shape_concat(all_inputs_upper, -1),
        params_lower,
        params_upper,
        p_mangled,
    )


if False:
    e0a: ExprAtomToAtomIndexed[CrimFeatureName] = (
        (),  # conditional exprs
        (  # polynomial
            (  # monomials
                (("f", 1),),  # monomial
                (),  # monomial
            ),
            (3, 4),  # polynomial coefficient indices
        ),
    )

    e0: ExprAtomToAtomIndexed[CrimFeatureName] = (
        (e0a,),  # conditional exprs
        (  # polynomial
            (  # monomials
                (  # monomial 1 (dist^2)
                    ("f", 0),
                    ("f", 0),
                ),
                (  # monomial 2 (dist)
                    ("f", 0),
                    ("if_ref", 0),
                ),
                (),  # monomial 3 (1)
            ),
            (0, 1, 2),  # polynomial coefficient indices
        ),
    )
    p0_lower = torch.tensor([0.0, 1.0, -130.0, 1.0, -0.5])
    p0_upper = torch.tensor([0.0, 1.0, -130.0, 1.0, -0.0])
    x = torch.tensor(
        [
            [120.0, 1.0],
            [140.0, 0.0],
        ]
    )

    eval_atom_to_atom_interval(x, p0_lower, p0_upper, e0)


def eval_list_to_atom(
    x: Float[Tensor, "#*b t n"],
    params: Float[Tensor, "p"],
    e: ExprListToAtom[CrimFeatureName, int],
) -> Float[Tensor, "#*b"]:
    match e:
        case "fold", (fold_e,):
            batch_shape = x.shape[:-2]
            fold_var = torch.zeros(batch_shape)
            for i in range(x.shape[-2]):
                fold_var = eval_atom_to_atom(
                    x[..., i, :],
                    fold_var,
                    params,
                    fold_e,
                )
            return fold_var
        case "ite", (c_e, a_e, b_e):
            assert False


def eval_list_to_atom_interval(
    x: Float[Tensor, "#*b t n"],
    params_lower: Float[Tensor, "p"],
    params_upper: Float[Tensor, "p"],
    e: ExprListToAtom[CrimFeatureName, int],
) -> Tuple[Float[Tensor, "#*b"], Float[Tensor, "#*b"]]:
    assert len(x.shape) == 3
    assert len(params_lower.shape) == 1
    assert len(params_upper.shape) == 1
    match e:
        case "fold", (fold_e,):
            batch_shape = x.shape[:-2]
            fold_var_lower = torch.zeros(batch_shape)
            fold_var_upper = torch.zeros(batch_shape)
            for i in range(x.shape[-2]):
                next_lower, next_upper = eval_atom_to_atom_interval(
                    x[..., i, :],
                    fold_var_lower,
                    fold_var_upper,
                    params_lower,
                    params_upper,
                    fold_e,
                )
                fold_var_lower = next_lower
                fold_var_upper = next_upper
            return fold_var_lower, fold_var_upper
        case "ite", (c_e, a_e, b_e):
            assert False


def eval_list_to_list(
    x: Float[Tensor, "#*b t n"],
    params: Float[Tensor, "p"],
    e: ExprListToList[CrimFeatureName, int],
) -> Float[Tensor, "#*b t"]:
    match e:
        case "map", (map_e,):
            return eval_atom_to_atom(x, None, params, map_e)
        case "map_prefixes", (map_prefixes_e,):
            ret: List[Float[Tensor, "#*b"]] = []
            for i in range(1, x.shape[-2] + 1):
                r = eval_list_to_atom(x[..., 0:i, :], params, map_prefixes_e)
                ret.append(r)
            return torch.stack(ret, -1)
        case "ite", (c_e, a_e, b_e):
            assert False


def eval_list_to_list_interval(
    x: Float[Tensor, "#*b t n"],
    params_lower: Float[Tensor, "p"],
    params_upper: Float[Tensor, "p"],
    e: ExprListToList[CrimFeatureName, int],
) -> Tuple[Float[Tensor, "#*b t"], Float[Tensor, "#*b t"]]:
    assert len(x.shape) == 3
    assert len(params_lower.shape) == 1
    assert len(params_upper.shape) == 1
    match e:
        case "map", (map_e,):
            return eval_atom_to_atom_interval(
                x, None, None, params_lower, params_upper, map_e
            )
        case "map_prefixes", (map_prefixes_e,):
            ret_lower: List[Float[Tensor, "#*b"]] = []
            ret_upper: List[Float[Tensor, "#*b"]] = []
            for i in range(1, x.shape[-2] + 1):
                l, u = eval_list_to_atom_interval(
                    x[..., 0:i, :], params_lower, params_upper, map_prefixes_e
                )
                ret_lower.append(l)
                ret_upper.append(u)
            return torch.stack(ret_lower, -1), torch.stack(ret_upper, -1)
        case "ite", (c_e, a_e, b_e):
            assert False


if False:
    eval_list_to_list_interval(
        torch.zeros([10, 100, 19]),
        torch.tensor([-1.0, -1.0]),
        torch.tensor([1.0, 1.0]),
        (
            "map_prefixes",
            (("fold", (((), (((), (("fold_ref",), ("fold_ref",))), (0, 1))),)),),
        ),
    )

if False:
    ex_x_0 = torch.tensor(
        [
            [
                [0.0],
                [1.0],
                [2.0],
            ]
        ]
    )
    ex_param_0 = torch.tensor(
        [
            2.0,
            1.0,
        ],
    )

    ex_e_f0: FoldExprAtomToAtom[CrimFeatureName, int] = (
        (),
        (
            ((("f", 0),),),
            (0,),
        ),
    )

    ex_e_f0_plus_prev: FoldExprAtomToAtom[CrimFeatureName, int] = (
        (),
        (
            ((("f", 0),), (("fold_ref",),)),
            (
                0,
                1,
            ),
        ),
    )

    ex_fold_e_0: ExprListToAtom[CrimFeatureName, int] = ("fold", (ex_e_f0,))
    ex_fold_e_1: ExprListToAtom[CrimFeatureName, int] = ("fold", (ex_e_f0_plus_prev,))
    ex_map_e_0: ExprListToList[CrimFeatureName, int] = ("map", (ex_e_f0,))
    ex_mapprefix_e_0: ExprListToList[CrimFeatureName, int] = (
        "map_prefixes",
        (ex_fold_e_0,),
    )
    ex_mapprefix_e_1: ExprListToList[CrimFeatureName, int] = (
        "map_prefixes",
        (ex_fold_e_1,),
    )

    eval_atom_to_atom_interval(ex_x_0, None, None, ex_param_0, ex_param_0, ex_e_f0)
    eval_list_to_atom_interval(ex_x_0, ex_param_0, ex_param_0, ex_fold_e_0)
    eval_list_to_atom_interval(ex_x_0, ex_param_0, ex_param_0, ex_fold_e_1)
    eval_list_to_list_interval(ex_x_0, ex_param_0, ex_param_0, ex_map_e_0)
    eval_list_to_list_interval(ex_x_0, ex_param_0, ex_param_0, ex_mapprefix_e_0)
    eval_list_to_list_interval(ex_x_0, ex_param_0, ex_param_0, ex_mapprefix_e_1)
