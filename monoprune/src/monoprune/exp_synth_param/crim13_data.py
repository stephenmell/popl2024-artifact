from monoprune.env import *
from jaxtyping import Shaped, Float, Bool
from torch import Tensor
import torch
import numpy as np
import pickle

p: Callable[[str], str] = lambda x: f"datasets/crim13_processed_data/{x}.npy"


def get_train(seed: int):
    cache_path: str = f"tmp/crim13_tuples_{seed}.pickle"

    try:
        with open(cache_path, "rb") as f:
            train_X, train_y, train_X_np, train_y_np = pickle.load(f)
    except Exception:

        def load_data(x: str) -> Float[Tensor, "..."]:
            return torch.from_numpy(np.load(p(x)))  # type: ignore

        def flatten(t: Shaped[Tensor, "a b *c"]) -> Shaped[Tensor, "a*b *c"]:
            return t.reshape((t.shape[0] * t.shape[1],) + t.shape[2:])

        def float_tensor_to_rat_lists(
            t: Float[Tensor, "n"]
        ) -> Tuple[Tuple[rat, ...], ...]:
            assert len(t.shape) == 2
            return tuple(tuple(rat.from_float(y.item()) for y in x) for x in t)

        def bool_tensor_to_bool_lists(t: Bool[Tensor, "n"]) -> Tuple[bool, ...]:
            assert len(t.shape) == 1
            return tuple(bool(x.item()) for x in t)

        train_X_nonflat = load_data("train_crim13_data")
        train_y_nonflat = load_data("train_crim13_labels") == 1

        flatten_before_permute = False
        n_truncate = 1000
        subset: Callable[[Any], Any] = lambda t: t[0:n_truncate]
        torch.manual_seed(seed)  # type: ignore
        #     if flatten_before_permute:
        #         # 1240400 examples when flattened
        #         train_X_torch = flatten(train_X_nonflat)
        #         train_y_torch = flatten(train_y_nonflat)
        #         perm = torch.randperm(train_X_torch.shape[0])
        #
        #         train_X: Tuple[Tuple[rat, ...], ...] = float_tensor_to_rat_lists(train_X_torch[perm])
        #         train_y: Tuple[bool, ...] = bool_tensor_to_bool_lists(train_y_torch[perm])
        #     else:
        perm = torch.randperm(train_X_nonflat.shape[0])
        train_X_np = flatten(subset(train_X_nonflat[perm]))
        train_X: Tuple[Tuple[rat, ...], ...] = float_tensor_to_rat_lists(train_X_np)
        train_y_np = flatten(subset(train_y_nonflat[perm]))
        train_y: Tuple[bool, ...] = bool_tensor_to_bool_lists(train_y_np)
        with open(cache_path, "wb") as f:
            pickle.dump((train_X, train_y, train_X_np, train_y_np), f)
    return train_X, train_y, train_X_np, train_y_np
