from monoprune.env import *

import monoprune.exp_synth_param.quivr_data as quivr_data
from monoprune.exp_torch.common import *
from monoprune.quivr_dsl.binary_semantics import eval_bool, BatchedTrace
from monoprune.quivr_dsl.syntax import QuivrExpr, add_param_indices

QuivrExprIndexed: TypeAlias = QuivrExpr[Tuple[int, ...]]

QuivrIntervalState: TypeAlias = Tuple[
    QuivrExprIndexed, Float[Tensor, "p"], Float[Tensor, "p"]
]


if False:
    dataset_name = "monoprune_warsaw_one"
    dataset_task = "eastward_4_high_accel"


def get_experiment(
    dataset_name: str, dataset_task: str, dataset_size: int, dataset_seed: int
) -> Tuple[Any, Any, Any]:
    assert dataset_seed == 0
    dataset_data = quivr_data.load_data(dataset_name)
    dataset_labels = quivr_data.load_labels(dataset_name, dataset_task)

    sub_traces: Dict[str, Any] = {}
    sub_labels = dataset_labels["labels"][0:dataset_size]
    for k, v in dataset_data["traces"].items():
        sub_traces[k] = v[0:dataset_size, :, :]
    traces: BatchedTrace = (
        {k: v for k, v in sub_traces.items() if v.dtype == torch.bool},
        {k: v for k, v in sub_traces.items() if v.dtype == torch.float32},
    )
    labels = sub_labels

    @jaxtyped
    def util_f1(
        initial_expr: QuivrExpr[Tuple[int, ...]],
        threshold: Float[Tensor, "p"],
    ) -> float:
        preds = eval_bool(traces, threshold, initial_expr)
        return f1(labels, preds).item()

    @jaxtyped
    def util_f1_interval(
        initial_expr: QuivrExpr[Tuple[int, ...]],
        threshold_lower: Float[Tensor, "p"],
        threshold_upper: Float[Tensor, "p"],
    ) -> Tuple[float, float]:
        # swapping upper and lower because the semantics are antitone
        preds_lower = eval_bool(traces, threshold_upper, initial_expr)
        preds_upper = eval_bool(traces, threshold_lower, initial_expr)

        f1_lower, f1_upper = f1_interval(labels, preds_lower, preds_upper)
        return f1_lower.item(), f1_upper.item()

    # skeletons = (
    #     (
    #         "seq",
    #         (
    #             ("pred", ("+y", None)),
    #             ("pred", ("-x", None)),
    #         ),
    #     ),
    #     (
    #         "seq",
    #         (
    #             ("pred", ("+y", None)),
    #             ("pred", ("+x", None)),
    #         ),
    #     ),
    #     (
    #         "seq",
    #         (
    #             ("pred", ("-y", None)),
    #             ("pred", ("+x", None)),
    #         ),
    #     ),
    #     (
    #         "seq",
    #         (
    #             ("pred", ("-y", None)),
    #             ("pred", ("-x", None)),
    #         ),
    #     ),
    # )

    pred_bounds = {
        k: (torch.tensor(v),) for k, v in dataset_data["pred1_bounds"].items()
    } | {k: () for k in traces[0].keys()}
    pred_arities = {k: len(v) for k, v in pred_bounds.items()}

    import monoprune.quivr_dsl.enumeration as quivr_enum

    if dataset_name == "maritime_surveillance":
        cfg_wrap_with_anything = False
    else:
        cfg_wrap_with_anything = True

    cfg_max_preds = 3
    cfg_max_pred1s = 2

    def enumerator(
        pred0_exprs: FrozenSet[QuivrExpr[None]], pred1_exprs: FrozenSet[QuivrExpr[None]]
    ):
        enumerated: quivr_enum.Enumerated = {
            (0, 0): frozenset(),
            (1, 0): pred0_exprs,
            (0, 1): pred1_exprs,
        }
        for i in range(cfg_max_pred1s + 1):
            quivr_enum.fill_up_to(enumerated, cfg_max_preds - i, i)

        if cfg_wrap_with_anything:
            return frozenset(
                (
                    "seq",
                    (
                        ("pred", ("anything", None)),
                        e,
                        ("pred", ("anything", None)),
                    ),
                )
                for k, v in enumerated.items()
                for e in v
            )
        else:
            return frozenset(e for k, v in enumerated.items() for e in v)

    # skeletons = tuple(
    #     enumerator(
    #         frozenset({("pred", (k, None)) for k in traces[0].keys()}),
    #         frozenset({("pred", (k, None)) for k in traces[1].keys()}),
    #     )
    # )
    skeletons = (("seq", (("pred", ("+x", None)), ("pred", ("+x", None)))),)
    print(skeletons[0])

    sketches_with_bounds: Tuple[IntervalState[QuivrExprIndexed], ...] = tuple(
        (
            sketch,
            torch.tensor(
                [
                    pred_bounds[param_name][param_index][0]
                    for param_name, param_index in sketch_param_preds
                ]
            ),
            torch.tensor(
                [
                    pred_bounds[param_name][param_index][1]
                    for param_name, param_index in sketch_param_preds
                ]
            ),
        )
        for skel in skeletons
        for sketch, sketch_param_preds in [add_param_indices(pred_arities, skel, ())]
    )

    return (
        sketches_with_bounds,
        util_f1,
        util_f1_interval,
        None,
        None,
    )  # util_bce, util_bce_interval
