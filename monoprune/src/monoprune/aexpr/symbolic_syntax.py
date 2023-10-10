from typing import overload
from monoprune.env import *
from monoprune.aexpr.syntax import BoolTerm, IntTerm, RatTerm  # type: ignore
from monoprune.unique import Unique

T = TypeVar("_T", "BoolSymbolic", "IntSymbolic", "RatSymbolic")

Concrete: TypeAlias = Union[bool, int, rat]
Symbolic: TypeAlias = Union["BoolSymbolic", "IntSymbolic", "RatSymbolic"]
ConcreteOrSymbolic: TypeAlias = Union[Concrete, Symbolic]


@dataclass
class BoolSymbolic:
    t: BoolTerm

    def __invert__(self) -> "BoolSymbolic":
        return BoolSymbolic(("inv", (self.t,)))

    def __and__(self, other: Any) -> "BoolSymbolic":
        other = _symbolic_bool(other)

        return BoolSymbolic(("and", (self.t, other.t)))

    def __rand__(self, other: Any) -> "BoolSymbolic":
        other = _symbolic_bool(other)

        return BoolSymbolic(("and", (other.t, self.t)))

    def __or__(self, other: Any) -> "BoolSymbolic":
        other = _symbolic_bool(other)

        return BoolSymbolic(("or", (self.t, other.t)))

    def __ror__(self, other: Any) -> "BoolSymbolic":
        other = _symbolic_bool(other)

        return BoolSymbolic(("or", (other.t, self.t)))

    @overload
    def __ite__(self, t: bool, f: bool) -> "BoolSymbolic":  # type: ignore
        ...

    @overload
    def __ite__(self, t: "BoolSymbolic", f: bool) -> "BoolSymbolic":
        ...

    @overload
    def __ite__(self, t: bool, f: "BoolSymbolic") -> "BoolSymbolic":
        ...

    @overload
    def __ite__(self, t: "BoolSymbolic", f: "BoolSymbolic") -> "BoolSymbolic":
        ...

    @overload
    def __ite__(self, t: int, f: int) -> "IntSymbolic":
        ...

    @overload
    def __ite__(self, t: "IntSymbolic", f: int) -> "IntSymbolic":
        ...

    @overload
    def __ite__(self, t: int, f: "IntSymbolic") -> "IntSymbolic":
        ...

    @overload
    def __ite__(self, t: "IntSymbolic", f: "IntSymbolic") -> "IntSymbolic":
        ...

    @overload
    def __ite__(self, t: rat, f: rat) -> "RatSymbolic":
        ...

    @overload
    def __ite__(self, t: "RatSymbolic", f: rat) -> "RatSymbolic":
        ...

    @overload
    def __ite__(self, t: rat, f: "RatSymbolic") -> "RatSymbolic":
        ...

    @overload
    def __ite__(self, t: "RatSymbolic", f: "RatSymbolic") -> "RatSymbolic":
        ...

    def __ite__(self, t: ConcreteOrSymbolic, f: ConcreteOrSymbolic) -> Symbolic:
        t = _symbolic(t)
        f = _symbolic(f)

        if isinstance(t, BoolSymbolic):
            assert isinstance(f, BoolSymbolic)
            return BoolSymbolic(("ite", (self.t, t.t, f.t)))
        elif isinstance(t, IntSymbolic):
            assert isinstance(f, IntSymbolic)
            return IntSymbolic(("ite", (self.t, t.t, f.t)))
        else:  # isinstance(t, RatSymbolic):
            assert isinstance(f, RatSymbolic)
            return RatSymbolic(("ite", (self.t, t.t, f.t)))


@dataclass
class IntSymbolic:
    t: IntTerm

    def __neg__(self) -> "IntSymbolic":
        return IntSymbolic(("neg", (self.t,)))

    def __add__(self, other: Any) -> "IntSymbolic":
        other_int = _symbolic_int(other)

        return IntSymbolic(("add", (self.t, other_int.t)))

    def __radd__(self, other: Any) -> "IntSymbolic":
        other_int = _symbolic_int(other)

        return IntSymbolic(("add", (other_int.t, self.t)))

    def __mul__(self, other: Any) -> "IntSymbolic":
        other_int = _symbolic_int(other)

        return IntSymbolic(("mul", (self.t, other_int.t)))

    def __rmul__(self, other: Any) -> "IntSymbolic":
        other_int = _symbolic_int(other)

        return IntSymbolic(("mul", (other_int.t, self.t)))

    def __truediv__(self, other: Any) -> "RatSymbolic":
        other_rat = _symbolic_rat_cast(other)
        self_rat = _symbolic_rat_cast(self)

        return RatSymbolic(("truediv", (self_rat.t, other_rat.t)))

    def __lt__(self, other: Any) -> "BoolSymbolic":
        self_rat = _symbolic_rat_cast(self)
        other_rat = _symbolic_rat_cast(other)

        return BoolSymbolic(("lt", (self_rat.t, other_rat.t)))

    def __le__(self, other: Any) -> "BoolSymbolic":
        self_rat = _symbolic_rat_cast(self)
        other_rat = _symbolic_rat_cast(other)

        return BoolSymbolic(("le", (self_rat.t, other_rat.t)))

    def __gt__(self, other: Any) -> "BoolSymbolic":
        self_rat = _symbolic_rat_cast(self)
        other_rat = _symbolic_rat_cast(other)

        return BoolSymbolic(("gt", (self_rat.t, other_rat.t)))

    def __ge__(self, other: Any) -> "BoolSymbolic":
        self_rat = _symbolic_rat_cast(self)
        other_rat = _symbolic_rat_cast(other)

        return BoolSymbolic(("ge", (self_rat.t, other_rat.t)))

    def __eq__(self, other: Any) -> "BoolSymbolic":  # type: ignore
        self_rat = _symbolic_rat_cast(self)
        other_rat = _symbolic_rat_cast(other)

        return BoolSymbolic(("eq", (self_rat.t, other_rat.t)))


@dataclass
class RatSymbolic:
    t: RatTerm

    def __neg__(self) -> "RatSymbolic":
        return RatSymbolic(("neg", (self.t,)))

    def __add__(self, other: Any) -> "RatSymbolic":
        other_rat = _symbolic_rat_cast(other)

        return RatSymbolic(("add", (self.t, other_rat.t)))

    def __radd__(self, other: Any) -> "RatSymbolic":
        other_rat = _symbolic_rat_cast(other)

        return RatSymbolic(("add", (other_rat.t, self.t)))

    def __mul__(self, other: Any) -> "RatSymbolic":
        other_rat = _symbolic_rat_cast(other)

        return RatSymbolic(("mul", (self.t, other_rat.t)))

    def __rmul__(self, other: Any) -> "RatSymbolic":
        other_rat = _symbolic_rat_cast(other)

        return RatSymbolic(("mul", (other_rat.t, self.t)))

    def __truediv__(self, other: Any) -> "RatSymbolic":
        other_rat = _symbolic_rat_cast(other)
        self_rat = _symbolic_rat_cast(self)

        return RatSymbolic(("truediv", (self_rat.t, other_rat.t)))

    def __lt__(self, other: Any) -> "BoolSymbolic":
        other_rat = _symbolic_rat_cast(other)

        return BoolSymbolic(("lt", (self.t, other_rat.t)))

    def __le__(self, other: Any) -> "BoolSymbolic":
        other_rat = _symbolic_rat_cast(other)

        return BoolSymbolic(("le", (self.t, other_rat.t)))

    def __gt__(self, other: Any) -> "BoolSymbolic":
        other_rat = _symbolic_rat_cast(other)

        return BoolSymbolic(("gt", (self.t, other_rat.t)))

    def __ge__(self, other: Any) -> "BoolSymbolic":
        other_rat = _symbolic_rat_cast(other)

        return BoolSymbolic(("ge", (self.t, other_rat.t)))

    def __eq__(self, other: Any) -> "BoolSymbolic":  # type: ignore
        other_rat = _symbolic_rat_cast(other)

        return BoolSymbolic(("eq", (self.t, other_rat.t)))


def _symbolic_bool(x: Any) -> "BoolSymbolic":
    if isinstance(x, BoolSymbolic):
        return x
    elif isinstance(x, bool):
        return BoolSymbolic(("lit", (x,)))
    else:
        assert False, (type(x), x)


def _symbolic_int(x: Any) -> "IntSymbolic":
    if isinstance(x, IntSymbolic):
        return x
    elif isinstance(x, int):
        return IntSymbolic(("lit", (x,)))
    else:
        assert False, (type(x), x)


def _symbolic_rat_cast(x: Any) -> "RatSymbolic":
    if isinstance(x, RatSymbolic):
        return x
    elif isinstance(x, rat):
        return RatSymbolic(("lit", (x,)))
    elif isinstance(x, IntSymbolic):
        return RatSymbolic(("of_int", (x.t,)))
    elif isinstance(x, int):
        return RatSymbolic(("of_int", (("lit", (x,)),)))
    else:
        assert False, (type(x), x)


def _symbolic(x: Any) -> Union[BoolSymbolic, IntSymbolic, RatSymbolic]:
    if isinstance(x, RatSymbolic):
        return x
    elif isinstance(x, rat):
        return RatSymbolic(("lit", (x,)))
    if isinstance(x, IntSymbolic):
        return x
    elif isinstance(x, int):
        return IntSymbolic(("lit", (x,)))
    if isinstance(x, BoolSymbolic):
        return x
    elif isinstance(x, bool):
        return BoolSymbolic(("lit", (x,)))
    else:
        assert False, (type(x), x)


def symbolic_bool_lit(x: bool) -> BoolSymbolic:
    return _symbolic_bool(x)


def symbolic_int_lit(x: int) -> IntSymbolic:
    return _symbolic_int(x)


def symbolic_rat_lit(x: rat) -> RatSymbolic:
    return _symbolic_rat_cast(x)


def symbolic_bool_var(ident: Unique) -> BoolSymbolic:
    return BoolSymbolic(("var", (ident,)))


def symbolic_int_var(ident: Unique) -> IntSymbolic:
    return IntSymbolic(("var", (ident,)))


def symbolic_rat_var(ident: Unique) -> RatSymbolic:
    return RatSymbolic(("var", (ident,)))


if False:
    a = symbolic_int_lit(5)
    b = symbolic_int_lit(2)
    from monoprune.unique import unique

    c_name = unique("c")
    c = symbolic_int_var(c_name)

    s: RatSymbolic = a / b + b / c
    {0: "0", 1: "!"}.__getitem__(0)
    from monoprune.aexpr.semantics_concrete import semantics as semantics_concrete

    def ffail(*_args, **_kwargs):
        assert False

    v = semantics_concrete({}.__getitem__, {c_name: 0}.__getitem__, {}.__getitem__)[2](
        s.t
    )
    print(t)
    print(v)
