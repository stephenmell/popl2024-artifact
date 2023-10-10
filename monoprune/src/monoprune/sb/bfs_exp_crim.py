from monoprune.env import *

from jaxtyping import Float, Bool, Int, Shaped
from monoprune.exp_torch.common import *
from monoprune.crim13_poly_dsl.syntax import *
from monoprune.crim13_poly_dsl.semantics import *
import monoprune.crim13_poly_dsl.enumeration as enumeration
import numpy as np

if False:
    dataset_name = "crim"
    dataset_task = "b"


def get_experiment(dataset_name: str, dataset_task: str, downsamp_n = 100) -> Tuple[Any, Any, Any]:
    assert dataset_name == "crim"

    p: Callable[[str], str] = lambda x: f"datasets/crim13_processed_data/{x}.npy"

    def load_data(x: str) -> Float[Tensor, "..."]:
        return torch.from_numpy(np.load(p(x)))  # type: ignore

    data_nonnormal = load_data("train_crim13_data")
    data_nonnormal_flat = data_nonnormal.reshape((12404 * 100, 19))
    data_absmax = (
        torch.max(torch.abs(data_nonnormal_flat), 0).values.unsqueeze(0).unsqueeze(0)  # type: ignore
    )
    data_absmax.shape
    data_all: Float[Tensor, "#*b n"] = data_nonnormal / data_absmax

    if dataset_task == "a":
        labels_all = load_data("train_crim13_labels") == 1
        label_weights = torch.tensor(1.5)
    elif dataset_task == "b":
        labels_all = load_data("train_crim13_labels_other") == 1
        label_weights = torch.tensor(1.0)
    else:
        assert False, dataset_task

    n_tot = data_all.shape[0]
    if downsamp_n is None:
        downsamp_n = n_tot
    subset_indices = make_subset_indices(0, n_tot, downsamp_n)
    data: Float[Tensor, "#*b n"] = data_all[subset_indices, :, :]
    labels = labels_all[subset_indices, :]
    # features_to_use = tuple([4])
    features_to_use = tuple(range(19))

    def f1_pre(
        trues: Bool[Tensor, "#*b n"], preds: Bool[Tensor, "#*b n"]
    ) -> Tuple[Int[Tensor, "#*b"], Int[Tensor, "#*b"], Int[Tensor, "#*b"],]:
        tp = preds[trues].sum()
        fp = preds[~trues].sum()
        np = trues.sum()
        return tp, fp, np  # type: ignore

    def f1(
        trues: Bool[Tensor, "#*b n"], preds: Bool[Tensor, "#*b n"]
    ) -> Float[Tensor, "#*b"]:
        tp, fp, np = f1_pre(trues, preds)
        return 2 * tp / (tp + fp + np)

    def flatten(t: Shaped[Tensor, "a b *c"]) -> Shaped[Tensor, "a*b *c"]:
        return t.reshape((t.shape[0] * t.shape[1],) + t.shape[2:])

    def util_f1(
        expr: ExprListToList[CrimFeatureName, int],
        parameters: Float[Tensor, "param"],
    ) -> float:
        logit = eval_list_to_list(data, parameters, expr)
        preds = logit >= 0
        assert preds.shape == labels.shape
        return f1(flatten(labels), flatten(preds)).item()

    def util_bce(
        expr: ExprListToList[CrimFeatureName, int],
        parameters: Float[Tensor, "param"],
    ) -> float:
        logit = eval_list_to_list(data, parameters, expr)
        assert logit.shape == labels.shape
        crossent = bce_logit(flatten(logit), flatten(labels), label_weights)
        return -torch.mean(crossent).item()

    DO_DBG = True

    def parameter_box_is_terminal(
        parameters_lower: Float[Tensor, "param"],
        parameters_upper: Float[Tensor, "param"],
    ):
        if parameters_lower.shape[0] == 0:
            return True
        return torch.min(parameters_upper - parameters_lower) <= 0.00001

    def util_f1_interval(
        expr: ExprListToList[CrimFeatureName, int],
        parameters_lower: Float[Tensor, "param"],
        parameters_upper: Float[Tensor, "param"],
    ) -> Tuple[float, float]:
        if parameter_box_is_terminal(parameters_lower, parameters_upper):
            u = util_f1(expr, (parameters_upper + parameters_lower) / 2)
            return (u, u)
        logit_lower_incl, logit_upper_excl = eval_list_to_list_interval(
            data, parameters_lower, parameters_upper, expr
        )
        preds_lower = logit_lower_incl >= 0
        preds_upper = logit_upper_excl >= 0  # exclusive
        assert (preds_upper >= preds_lower).all(), (
            ~(preds_upper >= preds_lower)
        ).nonzero()

        assert preds_lower.shape == labels.shape
        f1_lower, f1_upper = f1_interval(
            flatten(labels), flatten(preds_lower), flatten(preds_upper)
        )

        if DO_DBG:
            parameters_mid = (parameters_lower + parameters_upper) / 2
            logit = eval_list_to_list(data, parameters_mid, expr)
            preds = logit >= 0
            f1_ = f1(flatten(labels), flatten(preds))
            test_logit_lower = (logit >= logit_lower_incl).all()
            test_logit_upper = (logit_upper_excl >= logit).all()
            test_pred_lower = (preds >= preds_lower).all()
            test_pred_upper = (preds_upper >= preds).all()
            test_f1_lower = (f1_ >= f1_lower).all()
            test_f1_upper = (f1_upper >= f1_).all()
            if (
                not test_logit_lower
                or not test_logit_upper
                or not test_pred_lower
                or not test_pred_upper
                or not test_f1_lower
                or not test_f1_upper
            ):
                print(
                    test_logit_lower,
                    test_logit_upper,
                    test_pred_lower,
                    test_pred_upper,
                    test_f1_lower,
                    test_f1_upper,
                )
                print("params")
                print("\tlower")
                print("\t", parameters_lower)
                print("\tmid")
                print("\t", parameters_mid)
                print("\tupper")
                print("\t", parameters_upper)
                print("logits")
                print("\tlower")
                print("\t", logit_lower_incl)
                print("\tmid")
                print("\t", logit)
                print("\tupper")
                print("\t", logit_upper_excl)
                print("preds")
                print("\tlower")
                print("\t", preds_lower)
                print("\tmid")
                print("\t", preds)
                print("\tupper")
                print("\t", preds_upper)
                print("f1")
                print("\tlower")
                print("\t", f1_lower)
                print("\tmid")
                print("\t", f1_)
                print("\tupper")
                print("\t", f1_upper)
                assert False
        return f1_lower.item(), f1_upper.item()

    def util_bce_interval(
        expr: ExprListToList[CrimFeatureName, int],
        parameters_lower: Float[Tensor, "param"],
        parameters_upper: Float[Tensor, "param"],
    ) -> Tuple[float, float]:
        if parameter_box_is_terminal(parameters_lower, parameters_upper):
            u = util_bce(expr, (parameters_upper + parameters_lower) / 2)
            return (u, u)
        logit_lower_incl, logit_upper_excl = eval_list_to_list_interval(
            data, parameters_lower, parameters_upper, expr
        )
        crossent_lower, crossent_upper = bce_logit_interval(
            flatten(logit_lower_incl),
            flatten(logit_upper_excl),
            flatten(labels),
            label_weights,
        )
        ret = -torch.mean(crossent_upper).item(), -torch.mean(crossent_lower).item()

        if DO_DBG:
            pass  # TODO
        return ret

    lit_bounds = {
        "c0": torch.tensor([-1.0, -0.00001]),
        "c1": torch.tensor([0.00001, 1.0]),
    }

    # outlines = tuple(("sel", (feature, None)) for feature in feature_bounds.keys())
    features: Tuple[CrimFeatureName] = tuple(range(19))
    # outlines: Tuple[PolynomialSketch[CrimFeatureName], ...] = tuple(
    #     enumerate_sketches_nonempty(features, 1, 2)
    # )
    # outlines: Tuple[PolynomialSketch, ...] = (
    #     ((), (4,),),
    # )
    # outlines: Tuple[ExprAtomToAtom[CrimFeatureName, None], ...] = (
    #     (
    #         (),  # cond exprs
    #         (
    #             (
    #                 (),
    #                 (("f", 4),),
    #             ),  # monomials  # dist^1
    #             (  # param data
    #                 None,
    #                 None,
    #             ),
    #         ),
    #     ),
    # )
    if True:
        outlines: Tuple[ExprListToList[CrimFeatureName, None], ...] = tuple(
            enumeration.enumerate_expr_ll(features_to_use, 4)
        )

    if False:
        outlines: Tuple[ExprListToList[CrimFeatureName, None], ...] = tuple(
            ("map", (x,))
            for x in enumeration.enumerate_expr_aa(features_to_use, False, 3)
        )
    if False:
        outlines: Tuple[ExprListToList[CrimFeatureName, None], ...] = (
            (
                "map",
                (
                    (
                        (),
                        (
                            (
                                (),
                                (("f", 4),),
                            ),
                            (None, None),
                        ),
                    ),
                ),
            ),
            ("map", (((), (((),), (None,))),)),
            ("map", ((((((), (((),), (None,))),), (((("if_ref", 0),),), (None,))),))),
        )

    param_bounds = torch.tensor([-1.0, 1.0])

    sketches_with_bounds: Tuple[
        IntervalState[ExprListToList[CrimFeatureName, int]], ...
    ] = tuple(
        (
            sketch,
            torch.tensor([param_bounds[0] for n in sketch_param_preds]),
            torch.tensor([param_bounds[1] for n in sketch_param_preds]),
        )
        for outline in outlines
        for sketch, sketch_param_preds in [add_param_indices_list_to_list(outline, ())]
    )

    return sketches_with_bounds, util_f1, util_f1_interval, util_bce, util_bce_interval
