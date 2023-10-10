import torch

from torch import Tensor
from jaxtyping import Shaped, Bool


# out(i, j) = min_k max(A(i, k), B(k, j))
def maxmin_matmul(
    input: Shaped[Tensor, "#*b x z"],
    other: Shaped[Tensor, "#*b z y"],
) -> Shaped[Tensor, "#*b x y"]:
    assert input.shape[-1] == other.shape[-2]
    input_big = input.unsqueeze(-1)
    other_big = other.unsqueeze(-3)
    big = torch.min(input_big, other_big) # type: ignore
    return torch.max(big, -2).values # type: ignore


# out(j) = min_k max(V(k), B(k, j))
def maxmin_vecmatmul(
    vec: Shaped[Tensor, "#*b z"],
    mat: Shaped[Tensor, "#*b z y"],
) -> Shaped[Tensor, "#*b y"]:
    assert vec.shape[-1] == mat.shape[-2]
    vec_big = vec.unsqueeze(-1)
    big = torch.min(vec_big, mat) # type: ignore
    return torch.max(big, -2).values # type: ignore


# out(i) = min_k max(A(i, k), V(k))
def maxmin_matvecmul(
        mat : Shaped[Tensor, "#*b x z"],
        vec : Shaped[Tensor, "#*b z"]
) -> Shaped[Tensor, "#*b x"]:
    assert vec.shape[-1] == mat.shape[-1]
    vec_big = vec.unsqueeze(-2)
    big = torch.min(vec_big, mat) # type: ignore
    return torch.max(big, -1).values # type: ignore


# out(i) = min_k max(A(i, k), V(k))
def disjconj_matvecmul(
        mat : Bool[Tensor, "#*b x z"],
        vec : Bool[Tensor, "#*b z"]
) -> Bool[Tensor, "#*b x"]:
    assert vec.shape[-1] == mat.shape[-1]
    vec_big = vec.unsqueeze(-2)
    big = vec_big & mat
    return torch.any(big, -1)


def min_over_intervals(t : Shaped[Tensor, "#*b n"]) -> Shaped[Tensor, "#*b n n"]:
    n = t.shape[-1]
    ret = torch.zeros(t.shape + (n,), dtype = t.dtype, device = t.device)

    for a in range(n):
        for b in range(n - a):
            i = b
            j = b + a
            if j == i:
                ret[..., i, i] = t[..., i]
            else:
                ret[..., i, j] = torch.min(ret[..., i + 1, j], ret[..., i, j - 1]) # type: ignore
    return ret