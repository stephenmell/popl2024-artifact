from monoprune.env import *

from monoprune.crim13_poly_dsl.syntax import (
    CrimFeatureName,
    Monomial,
    PolynomialSketch,
    FoldExprAtomToAtom,
    ExprListToAtom,
    ExprListToList,
    Polynomial,
    FoldRefOrIfRefOrF,
)
import itertools
from monoprune.misc_utils import list_to_bag

F = TypeVar("F")

if False:

    def enumerate_monomials(
        vars: Iterable[CrimFeatureName], max_degree: int
    ) -> Iterable[Monomial[CrimFeatureName]]:
        vars_immutable = tuple(vars)
        return itertools.chain(
            *(
                itertools.combinations_with_replacement(vars_immutable, i)
                for i in range(max_degree + 1)
            )
        )

    def enumerate_sketches_nonempty(
        vars: Tuple[CrimFeatureName], max_degree: int, max_size: int
    ) -> Iterable[PolynomialSketch[CrimFeatureName]]:
        monomials = tuple(enumerate_monomials(vars, max_degree))
        return itertools.chain(
            *(itertools.combinations(monomials, i) for i in range(1, max_size + 1))
        )

    def simple_structural_cost_polynomial_sketch(e: PolynomialSketch[F]) -> float:
        return len(e) + sum(len(monomial) for monomial in e)

    def simple_structural_cost_atom_to_atom(e: FoldExprAtomToAtom[F, T]) -> float:
        this_cost = len(e[0]) + simple_structural_cost_polynomial_sketch(e[1][0])
        rec_cost = sum(simple_structural_cost_atom_to_atom(e2) for e2 in e[0])
        return this_cost + rec_cost

    def simple_structural_cost_list_to_atom(e: ExprListToAtom[F, T]) -> float:
        match e:
            case "fold", (fold_e,):
                return 1 + simple_structural_cost_atom_to_atom(fold_e)
            case "ite", (c_e, a_e, b_e):
                assert False

    def simple_structural_cost_list_to_list(e: ExprListToList[F, T]) -> float:
        match e:
            case "map", (map_e,):
                return 1 + simple_structural_cost_atom_to_atom(map_e)
            case "map_prefixes", (map_prefixes_e,):
                return 1 + simple_structural_cost_list_to_atom(map_prefixes_e)
            case "ite", (c_e, a_e, b_e):
                assert False


if False:
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
    simple_structural_cost_list_to_list(ex_mapprefix_e_0)
    simple_structural_cost_atom_to_atom(ex_e_f0)

import functools


@functools.lru_cache
def enumerate_fixed_split_ordered(n: int, k: int) -> FrozenSet[Tuple[int, ...]]:
    ret: MutableSet[Tuple[int, ...]] = set()
    if k == 0:
        if n == 0:
            ret.add(())
    else:
        for hd in range(0, n + 1):
            for tl in enumerate_fixed_split_ordered(n - hd, k - 1):
                ret.add((hd,) + tl)
    return frozenset(ret)


@functools.lru_cache
def enumerate_fixed_split(n: int, k: int) -> FrozenSet[Tuple[int, ...]]:
    ret: MutableSet[Tuple[int, ...]] = set()
    if k == 0:
        if n == 0:
            ret.add(())
    else:
        for hd in range(0, n + 1):
            for tl in enumerate_fixed_split(n - hd, k - 1):
                cons = tuple(sorted((hd,) + tl))
                ret.add(cons)
    return frozenset(ret)


@functools.lru_cache
def enumerate_fixed_split_nonempty(n: int, k: int) -> FrozenSet[Tuple[int, ...]]:
    ret: MutableSet[Tuple[int, ...]] = set()
    if k == 0 and n == 0:
        ret.add(())
    elif k <= 0 or n <= 0:
        pass
    else:
        for hd in range(1, n + 1):
            for tl in enumerate_fixed_split_nonempty(n - hd, k - 1):
                cons = tuple(sorted((hd,) + tl))
                ret.add(cons)
    return frozenset(ret)


@functools.lru_cache
def enumerate_splits_nonempty(n: int) -> FrozenSet[Tuple[int, ...]]:
    ret: MutableSet[Tuple[int, ...]] = set()

    if n == 0:
        return frozenset({()})
    else:
        for k in range(1, n + 1):
            ret.update(enumerate_fixed_split_nonempty(n, k))

    return frozenset(ret)


@functools.lru_cache
def enumerate_splits_with_split_cost(
    n: int, split_cost: int
) -> FrozenSet[Tuple[int, ...]]:
    ret: MutableSet[Tuple[int, ...]] = set()

    if n == 0:
        return frozenset({()})
    else:
        for k in range(1, n + 1):
            fuel_remaining = n - k * split_cost
            if fuel_remaining >= 0:
                ret.update(enumerate_fixed_split(fuel_remaining, k))

    return frozenset(ret)


if False:
    enumerate_fixed_split(0, 0)
    enumerate_splits_with_split_cost(3, 1)


@functools.lru_cache
def enumerate_monomials(
    indicators: Tuple[F, ...],
    variables: Tuple[F, ...],
    size: int,
) -> FrozenSet[Monomial[F]]:
    ret: MutableSet[Monomial[F]] = set()
    for size_indicators, size_vars in enumerate_fixed_split_ordered(size, 2):
        for sel_indic in itertools.combinations(indicators, size_indicators):
            for sel_var in itertools.combinations_with_replacement(
                variables, size_vars
            ):
                ret.add(sel_indic + sel_var)
    return frozenset(ret)


# here we have a class of things X, parameterized by size
# so we have a function f : int -> Set[X] which returns all X of that size
# here we'd like to generate sets of things whose size adds up to n
# so we're going to do an interger partition of n
# then group those subsizes by multiplicity
# then compute all of the things (by calling f) for each subsize
# then take combinations of those
def enumerate_sets_of_total_size(
    enumerate_objects: Callable[[int], FrozenSet[T]],
    subsizes_iter: Iterable[Tuple[int, ...]],
) -> FrozenSet[FrozenSet[T]]:
    ret: MutableSet[FrozenSet[T]] = set()

    for subsizes in subsizes_iter:
        prev_sets: MutableSet[FrozenSet[T]] = {frozenset()}
        for subsize, count in list_to_bag(subsizes).items():
            objects_at_subsize = enumerate_objects(subsize)
            next_sets: MutableSet[FrozenSet[T]] = set()
            for prev_set in prev_sets:
                for next_set_ in itertools.combinations(objects_at_subsize, count):
                    next_set: FrozenSet[T] = frozenset(next_set_)  # type: ignore
                    new_set: FrozenSet[T] = prev_set | next_set
                    next_sets.add(new_set)
            prev_sets = next_sets
        ret.update(prev_sets)

    return frozenset(ret)


# enumerate_sets_of_total_size(lambda subsize: {"Foo", "Var"}, 0)
# enumerate_sets_of_total_size(
#     lambda subsize: frozenset({(subsize, i) for i in range(10)}), 2
# )


@functools.lru_cache
def enumerate_polynomial_sketches(
    indicators: Tuple[F, ...],
    variables: Tuple[F, ...],
    size: int,
) -> FrozenSet[PolynomialSketch[F]]:
    return frozenset(
        tuple(sorted(monomials))  # type: ignore
        for monomials in enumerate_sets_of_total_size(
            lambda monomial_size: enumerate_monomials(indicators, variables, monomial_size),  # type: ignore
            enumerate_splits_with_split_cost(size, 1),
        )
    )


if False:
    enumerate_polynomial_sketches((), ("x",), 2)
    enumerate_splits_with_split_cost(1, 1)

""" @functools.lru_cache
def enumerate_polynomial_sketches_old(
    indicators: Tuple[F, ...],
    variables: Tuple[F, ...],
    size: int,
) -> FrozenSet[PolynomialSketch[F]]:
    ret: MutableSet[PolynomialSketch[F]] = set()

    for monomial_sizes in enumerate_splits_nonempty(size):
        prev_polys: MutableSet[FrozenSet[Monomial[F]]] = set({frozenset({})})
        for monomial_size, count in list_to_bag(monomial_sizes).items():
            next_polys: MutableSet[FrozenSet[Monomial[F]]] = set()
            for polynomial in prev_polys:
                for monomials_ in itertools.combinations(
                    enumerate_monomials(indicators, variables, monomial_size), count
                ):
                    monomials: FrozenSet[Monomial[F]] = frozenset(
                        monomials_
                    )  # type: ignore
                    new_polynomial: FrozenSet[Monomial[F]] = polynomial | monomials
                    next_polys.add(new_polynomial)
            prev_polys = next_polys

        for poly in prev_polys:
            ret.add(tuple(sorted(poly)))

    return frozenset(ret) """

# enumerate_polynomial_sketches_old(("c",), ("x", "y", "z"), 3) == enumerate_polynomial_sketches(("c",), ("x", "y", "z"), 3)


def polynomial_sketch_to_polynomial(ps: PolynomialSketch[F]) -> Polynomial[F, None]:
    return (ps, tuple(None for _m in ps))


@functools.lru_cache
def enumerate_cond_expr_sets(
    features: Tuple[F, ...],
    is_inside_fold: bool,
    size: int,
) -> FrozenSet[FrozenSet[FoldExprAtomToAtom[F, None]]]:
    return enumerate_sets_of_total_size(
        lambda expr_size: enumerate_expr_aa(features, is_inside_fold, expr_size),  # type: ignore
        enumerate_splits_nonempty(size),
    )


"""     ret: MutableSet[FrozenSet[FoldExprAtomToAtom[F, None]]] = set()
    for expr_sizes in enumerate_splits_nonempty(size):
        prev_exprs: MutableSet[FrozenSet[FoldExprAtomToAtom[F, None]]] = set({frozenset({})})
        for expr_size, count in list_to_bag(expr_sizes).items():
            next_exprs: MutableSet[FrozenSet[FoldExprAtomToAtom[F, None]]] = set({})

            for exprs in prev_exprs:
                for 
            prev_exprs = next_exprs
        ret.update(prev_exprs)

    return frozenset(ret) """


def uses_fold_var(e: FoldExprAtomToAtom[F, None]) -> bool:
    for cond_e in e[0]:
        if uses_fold_var(cond_e):
            return True

    for monomial in e[1][0]:
        for var in monomial:
            if var == ("fold_ref",):
                return True

    return False

def uses_some_feature(e: FoldExprAtomToAtom[F, None]) -> bool:
    for cond_e in e[0]:
        if uses_some_feature(cond_e):
            return True

    for monomial in e[1][0]:
        for var in monomial:
            if var[0] == "f":
                return True

    return False


@functools.lru_cache
def enumerate_expr_aa(
    features: Tuple[F, ...],
    is_inside_fold: bool,
    size: int,
) -> FrozenSet[FoldExprAtomToAtom[F, None]]:
    ret: MutableSet[FoldExprAtomToAtom[F, None]] = set()
    assert size >= 0

    for cond_size, poly_size in enumerate_fixed_split_ordered(size, 2):
        if poly_size == 0:
            continue
        for cond_expr_sets in enumerate_cond_expr_sets(
            features, is_inside_fold, cond_size
        ):
            cond_exprs_list: Tuple[FoldExprAtomToAtom[F, None]] = tuple(
                sorted(cond_expr_sets)
            )  # type: ignore
            indicators: Tuple[FoldRefOrIfRefOrF[F]] = tuple(
                ("if_ref", i) for i, _ in enumerate(cond_exprs_list)
            )
            variables: Tuple[FoldRefOrIfRefOrF[F]] = tuple(
                ("f", f) for f in features
            ) + ((("fold_ref",),) if is_inside_fold else ())

            assert len(cond_exprs_list) == len(frozenset(cond_exprs_list))

            for poly_sketch_ in enumerate_polynomial_sketches(
                indicators, variables, poly_size
            ):
                poly_sketch: PolynomialSketch[FoldRefOrIfRefOrF[F]] = poly_sketch_  # type: ignore

                used_conds: MutableSet[int] = set()
                for monomial in poly_sketch:
                    for x in monomial:
                        match x:
                            case "if_ref", i:
                                used_conds.add(i)
                            case _:
                                pass
                new_expr: FoldExprAtomToAtom[F, None] = (
                    cond_exprs_list,
                    polynomial_sketch_to_polynomial(poly_sketch),
                )
                if len(used_conds) == len(cond_exprs_list):
                    # we only consider expressions that use all conditionals
                    if not is_inside_fold or uses_fold_var(new_expr):
                        # we only consider expressions inside of folds that use the fold var at least once
                        if uses_some_feature(new_expr):
                            # only consider expressions that actually use some feature
                            ret.add(new_expr)

    return frozenset(ret)


if False:
    enumerate_polynomial_sketches((), ("a",), 3)
    enumerate_expr_aa(("a",), False, 3)

# enumerate_polynomial_sketches(("c"), ("x", "y", "z"), 3)
# len(enumerate_expr_aa(tuple(range(19)), True, 4))
# enumerate_cond_expr_sets(("x", "y", "z"), False, 2)
#
#     if size > 0:
#         for x in enumerate_aa(size - 1):
#             with_new_cond =
#             pass
#
#     return frozenset(ret)

# TODO: distinguish raw, non-raw, filter out duplicate monomials, ordering?
# sort each monomial, setize list of monomials
# ideally we do this at each step, except settizing needs to be baggizing


@functools.lru_cache
def enumerate_expr_la(
    features: Tuple[F, ...],
    size: int,
) -> FrozenSet[ExprListToAtom[F, None]]:
    ret: MutableSet[ExprListToAtom[F, None]] = set()
    # if size > 0:
    if (
        size >= 0
    ):  # we don't decrement size, because otherwise to get mapprefix; fold costs two, which really biases the search space toward just map
        # fold
        for expr_aa in enumerate_expr_aa(features, True, size):
            ret.add(("fold", (expr_aa,)))

    if size > 0:  # we decrement here to avoid an infinite loop
        # ite
        for c_size, a_size, b_size in enumerate_fixed_split_ordered(size - 1, 3):
            for c_expr in enumerate_expr_la(features, c_size):
                for a_expr in enumerate_expr_la(features, a_size):
                    for b_expr in enumerate_expr_la(features, b_size):
                        ret.add(("ite", (c_expr, a_expr, b_expr)))

    return frozenset(ret)


@functools.lru_cache
def enumerate_expr_ll(
    features: Tuple[F, ...],
    size: int,
) -> FrozenSet[ExprListToList[F, None]]:
    ret: MutableSet[ExprListToList[F, None]] = set()
    if size > 0:
        # fold
        for expr_aa in enumerate_expr_aa(features, False, size - 1):
            ret.add(("map", (expr_aa,)))

        # fold
        for expr_la in enumerate_expr_la(features, size - 1):
            ret.add(("map_prefixes", (expr_la,)))

        # ite
        for c_size, a_size, b_size in enumerate_fixed_split_ordered(size - 1, 3):
            for c_expr in enumerate_expr_la(features, c_size):
                for a_expr in enumerate_expr_ll(features, a_size):
                    for b_expr in enumerate_expr_ll(features, b_size):
                        ret.add(("ite", (c_expr, a_expr, b_expr)))

    return frozenset(ret)


if False:
    # len(tuple(x for x in enumerate_expr_ll(("a", "b", "c"), 8) if x[0] == "map_prefixes" and x[1][0][0] == "ite"))
    l = tuple(enumerate_expr_la(("a", "b", "c"), 7))
    l2 = [x for x in l if x[0] == "ite"]

    import monoprune.crim13_poly_dsl.syntax as syntax

    syntax.add_param_indices_list_to_atom(l2[10], ())

if False:
    l = tuple(enumerate_expr_ll(("a", "b", "c"), 8))
    l2 = [x for x in l if x[0] == "ite"]
    import monoprune.crim13_poly_dsl.syntax as syntax

    syntax.add_param_indices_list_to_list(l2[200], ())
