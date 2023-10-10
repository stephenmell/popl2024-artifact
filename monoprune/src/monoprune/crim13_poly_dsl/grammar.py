from monoprune.env import *

GrammarKind: TypeAlias = str
GrammarExpr: TypeAlias = Any
GrammarKindData: TypeAlias = Any
GrammarPartialExpr: TypeAlias = Any
GrammarRealizer: TypeAlias = Callable[
    [GrammarPartialExpr, Tuple[GrammarExpr, ...]], GrammarExpr
]
GrammarExpanderResult: TypeAlias = Tuple[
    GrammarPartialExpr, Tuple[Tuple[GrammarKind, GrammarKindData], ...]
]
GrammarExpander: TypeAlias = Callable[
    [
        GrammarKindData,
    ],
    Iterable[GrammarExpanderResult],
]
Grammar: TypeAlias = Tuple[
    Dict[GrammarKind, Tuple[GrammarExpander, GrammarRealizer]],
    Tuple[GrammarKind, GrammarKindData],
]


def aa_expander(data: GrammarKindData) -> Iterable[GrammarExpanderResult]:
    assert data is None

    def prod_fixme(subexprs: Tuple[GrammarExpr, ...]) -> GrammarExpr:
        assert len(subexprs) == 0
        return ("FIXME",)

    yield (prod_fixme, ())


def la_expander(data: GrammarKindData) -> Iterable[GrammarExpanderResult]:
    assert data is None

    def prod_fold(subexprs: Tuple[GrammarExpr, ...]) -> GrammarExpr:
        (fold_expr,) = subexprs
        return ("fold", (fold_expr,))

    yield (prod_fold, (("aa", None),))

    def prod_ite(subexprs: Tuple[GrammarExpr, ...]) -> GrammarExpr:
        (
            c_expr,
            a_expr,
            b_expr,
        ) = subexprs
        return (
            "ite",
            (
                c_expr,
                a_expr,
                b_expr,
            ),
        )

    yield (
        prod_ite,
        (
            ("la", None),
            ("la", None),
            ("la", None),
        ),
    )


def ll_expander(data: GrammarKindData) -> Iterable[GrammarExpanderResult]:
    assert data is None
    assert False


def aa_realizer(
    part: GrammarPartialExpr, subexprs: Tuple[GrammarExpr, ...]
) -> GrammarExpr:
    return part(*subexprs)


def la_realizer(
    part: GrammarPartialExpr, subexprs: Tuple[GrammarExpr, ...]
) -> GrammarExpr:
    return part(*subexprs)


def ll_realizer(
    part: GrammarPartialExpr, subexprs: Tuple[GrammarExpr, ...]
) -> GrammarExpr:
    return part(*subexprs)


grammar: Grammar = (
    {
        "aa": (aa_expander, aa_realizer),
        "la": (la_expander, la_realizer),
        "ll": (ll_expander, ll_realizer),
    },
    ("la", None),
)

V = TypeVar("V")


class SearchGraph(Protocol[T, V]):
    def initial(self) -> Iterable[T]:
        ...

    def expand(self, node: T) -> Tuple[V, Iterable[T]]:
        ...


def grammar_leftmost_search_graph(
    grammar: Grammar,
) -> SearchGraph[GrammarExpanderResult, GrammarPartialExpr]:
    # foob is needed because python made poor choices for closures and loop variables
    def foob(
        base_pexpr: GrammarPartialExpr,
        base_hole_data_tl : Tuple[Tuple[GrammarKind, GrammarKindData], ...],
        first_hole_pexpr: GrammarPartialExpr,
        first_hole_data : Tuple[Tuple[GrammarKind, GrammarKindData], ...],
    ) -> GrammarExpanderResult:
        def new_pexpr(subexprs : Tuple[GrammarExpr]) -> GrammarExpr:
            assert len(subexprs) == len(first_hole_data) + len(base_hole_data_tl)
            subexprs_first = subexprs[0:len(first_hole_data)]
            subexprs_tl = subexprs[len(first_hole_data):]
            first_subexpr = first_hole_pexpr(subexprs_first)
            return base_pexpr((first_subexpr,) + subexprs_tl)
        return new_pexpr, first_hole_data + base_hole_data_tl

    class ret:
        @staticmethod
        def initial() -> Iterable[GrammarExpanderResult]:
            def f(x: GrammarPartialExpr) -> GrammarPartialExpr:
                return x

            yield f, (grammar[1],)

        @staticmethod
        def expand(
            node: GrammarExpanderResult,
        ) -> Tuple[GrammarPartialExpr, Iterable[GrammarExpanderResult],]:
            partial_expr, hole_data = node

            def iterable_() -> Iterable[GrammarExpanderResult]:
                if len(hole_data) > 0:
                    first_hole_kind, first_hole_data = hole_data[0]
                    for hole_expr, hole_args in grammar[0][first_hole_kind][0](
                        first_hole_data
                    ):
                        yield foob(partial_expr, hole_data[1:], hole_expr, hole_args)

            return partial_expr, iterable_()

    return ret
