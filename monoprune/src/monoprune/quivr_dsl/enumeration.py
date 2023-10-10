from monoprune.env import *

from monoprune.quivr_dsl.syntax import QuivrExpr

Enumerated: TypeAlias = Dict[Tuple[int, ...], FrozenSet[QuivrExpr[None]]]


# ways of splitting n into k numbers that sum to n
def integer_partition(k: int, n: int) -> Iterable[Tuple[int, ...]]:
    assert k == 2
    for i in range(n + 1):
        yield (i, n - i)


# ways of splitting ns into k tuples that sum (component-wise) to ns
import itertools


def integer_tuple_partition(
    k: int, ns: Tuple[int, ...]
) -> Iterable[Tuple[Tuple[int, ...], ...]]:
    return itertools.product(*(integer_partition(k, n) for n in ns))


def maybe_inline_expr(ctor: str, expr: QuivrExpr[T]) -> Tuple[QuivrExpr[T]]:
    if expr[0] == ctor:
        return expr[1]  # type: ignore
    else:
        return (expr,)


def fill_i_j(enumerated: Enumerated, i: int, j: int):
    assert (i, j) not in enumerated

    new_set: MutableSet[QuivrExpr[None]] = set()

    for (i_l, i_r), (j_l, j_r) in integer_tuple_partition(2, (i, j)):
        if (i_l == i and j_l == j) or (i_r == i and j_r == j):
            # if we're trying to a build thing of size (i, j), in principle we could use
            # a thing of size (i, j) and a thing of size (0, 0), but there are no things
            # of size (0, 0), and yet if we didn't bail here we'd look at enumerated[i, j]
            # and find it doesn't exist (since we're trying to populate it!)
            continue

        for l in enumerated[i_l, j_l]:
            for r in enumerated[i_r, j_r]:
                new_set.add(
                    (
                        "conj",
                        (
                            *maybe_inline_expr("CONJ", l),
                            *maybe_inline_expr("CONJ", r),
                        ),
                    )
                )
                # FIXME: this is probably leading to redundancy in later steps
                # by only doing it in one order we get rid of commutativity
                # duplication, but not associativity duplication
                # taking a 1 and a 1 is okay, but if we take a 1 and a 2 or a
                # 2 and a 1, there's overlap
                # FIXME: wait I think the proceeding is optimistic: we enumerate the
                # big sets once on the left and once on the right, so we are also losing
                # out on commutativity

                new_set.add(
                    (
                        "seq",
                        (
                            *maybe_inline_expr("SEQ", l),
                            *maybe_inline_expr("SEQ", r),
                        ),
                    )
                )
                new_set.add(
                    (
                        "seq",
                        (
                            *maybe_inline_expr("SEQ", r),
                            *maybe_inline_expr("SEQ", l),
                        ),
                    )
                )

    enumerated[i, j] = frozenset(new_set)


def fill_up_to(enumerated: Enumerated, n: int, m: int):
    for i in range(n + 1):
        for j in range(m + 1):
            if (i, j) not in enumerated:
                fill_i_j(enumerated, i, j)


def enumerate_up_to_separate_01(
    pred0: FrozenSet[QuivrExpr[None]],
    pred1: FrozenSet[QuivrExpr[None]],
    num_pred0: int,
    num_pred1: int,
):
    enumerated: Enumerated = {
        (0, 0): frozenset(),
        (1, 0): pred0,
        (0, 1): pred1,
    }
    fill_up_to(enumerated, num_pred0, num_pred1)
    return frozenset(x for k, v in enumerated.items() for x in v)


def enumerate_up_to_combined_01(
    pred0: FrozenSet[QuivrExpr[None]],
    pred1: FrozenSet[QuivrExpr[None]],
    num_pred01: int,
):
    enumerated: Enumerated = {
        (0, 0): frozenset(),
        (1, 0): pred0,
        (0, 1): pred1,
    }
    for i in range(num_pred01 + 1):
        fill_up_to(enumerated, i, num_pred01 - i)
    return frozenset(x for k, v in enumerated.items() for x in v)


if False:
    len(
        enumerate_up_to_separate_01(
            frozenset(
                {
                    ("PRED0", "top"),
                    ("PRED0", "bot"),
                }
            ),
            frozenset(
                {
                    ("PRED1", "t+", None),
                    ("PRED1", "t-", None),
                }
            ),
            2,
            1,
        )
    )
