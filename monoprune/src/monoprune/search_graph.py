from monoprune.env import *
from monoprune.type_grammar_tools import *

class SearchGraph(Protocol[T]):
    def initial(self) -> Iterable[T]:
        ...

    def expand(self, node: T) -> Iterable[T]:
        ...


def cfg_search_graph(
    grammar: Cfg, initial_type: CfgTypeName
) -> SearchGraph[CfgExprPartial]:
    class ret:
        @staticmethod
        def initial() -> Iterable[CfgExprPartial]:
            yield (None, initial_type)

        @staticmethod
        def expand(node: CfgExprPartial) -> Iterable[CfgExprPartial]:
            return expand_first_hole(grammar, node)

    return ret


def breadth_first_search(graph: SearchGraph[T]) -> Iterable[T]:
    queue = list(graph.initial())
    while queue:
        x = queue.pop(0)
        yield x
        queue.extend(graph.expand(x))


def depth_first_while(graph: SearchGraph[T], cond: Callable[[T], bool]) -> Iterable[T]:
    queue = list(graph.initial())
    while queue:
        x = queue.pop()
        if cond(x):
            yield x
            queue.extend(graph.expand(x))

def breadth_first_while(graph: SearchGraph[T], cond: Callable[[T], bool]) -> Iterable[T]:
    queue = list(graph.initial())
    while queue:
        x = queue.pop(0)
        if cond(x):
            yield x
            queue.extend(graph.expand(x))
