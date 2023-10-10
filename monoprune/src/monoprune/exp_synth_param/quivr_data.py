from monoprune.env import *
from jaxtyping import Shaped, Float, Bool
from torch import Tensor
import torch
import numpy as np
import numpy.random
import pandas as pd

p_data: Callable[
    [str], str
] = lambda x: f"datasets/quivr_hard/{x}/data.torch"
p_labels: Callable[
    [str, str], str
] = lambda x, y: f"datasets/quivr_hard/{x}/task_{y}/task.torch"


def load_data(x: str) -> Any:
    return torch.load(p_data(x), map_location="cpu")  # type: ignore


def load_labels(x: str, y: str) -> Any:
    return torch.load(p_labels(x, y), map_location="cpu")  # type: ignore
