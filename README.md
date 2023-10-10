# Artifact for Mell et al. at POPL 2024

## Installation
All commands should be run from the repository root directory.

Installing via Docker is recommended for reproducing the results. Pip may be more convenient for development.

### Via Docker
First, build the Docker image with `docker build --platform linux/amd64 -t monoprune .`. This should take around 5 minutes.

If you are on a non amd64 machine (e.g. an M1 Mac), the `--platform linux/amd64` should ensure that the code runs. However, it may be significantly slower than running natively, which may affect timing results.

Run `docker run -v "$(realpath .)":/root/project -w /root/project --rm -ti monoprune bash` to drop into a shell in the Docker environment, where subsequent commands can be run.

### Via Pip
We suggest using a fresh virtual environment with Python 3.11.4. Unless you need GPU support in PyTorch, run `pip3 install torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu` to install the CPU-only version, which is a smaller download. Finally, install the local `monoprune` package with ``pip3 install -e ./monoprune/``. The dependency versions in `pyproject.toml` are conservative, and you may wish to relax them.

## Reproducing Claims
First, download the dataset tarball (DOI 10.5281/zenodo.8423616) and extract it into the `datasets` directory:
```
wget https://zenodo.org/record/8423617/files/mell2024_dataset.tar.xz
tar -xJvf mell2024_dataset.tar.xz -C datasets/
```

The paper introduces an algorithm (Heuristic) for synthesizing optimal programs more quickly than two baselines (SMT and BFS). We conduct two experiments:

### Heuristic vs SMT (Paper Section 6.2)
This shows that Heuristic converges to the optimal program more quickly than SMT, with the gap increasing as the size of the dataset increases. We support the claim with two charts (Figure 2), showing synthesis runtime vs dataset size for two sketches.

To run synthesis for this experiment, run `python3 -m monoprune.sb.run_smt_all`. Running the full experient may be very slow, particularly for the CRIM13 task (~10 hours on a 36 core server CPU), so you may wish to run a subset of dataset sizes (e.g. by uncommenting `run_smt_all.py` line 19, which should take less than half an hour on a laptop). This will output synthesized programs and their classification scores in `output_exp_smt/`. Note that this script does not support pausing or resuming, so you may wish to use `screen` or `tmux`.

Once synthesis has been run, `python3 -m monoprune.sb.smt_plot` will generate the plots as PDFs in `results/`. If you evaluated on a subset of dataset sizes, there may be gaps in the chart. Though the chart may not match the paper exactly, SMT should be clearly much, much slower than Heuristic.

### Heuristic vs BFS (Paper Section 6.3)
This shows that Heuristic achieves better performance, faster, than BFS. We support this with a table (Table 1), showing the classification performance over time of both approaches.

To run synthesis for this experiment, run `python3 -m monoprune.sb.run_bfs_all`. This will output synthesized programs and their classification scores in `output_exp_bfs`. This will take about 6 hours to run (38 tasks with a 10 minute timeout for each). If this is too long, comment out an appropriate number of elements of `tasks` in `sb/run_bfs_all.py`. Note that this script does not support pausing or resuming, so you may wish to use `screen` or `tmux`.

Once synthesis has been run, `python3 -m monoprune.sb.bfs_table` will generate the table's LaTeX source in `results/`. The numbers are the classification performance at specific points in time, and so are likely to differ due to differences in hardware. Emulating amd64 on an M1 Mac may cause a particularly steep decline in performance. However, in almost all cases Heuristic should achieve higher scores and narrow intervals than BFS at each time.

## Artifact Structure
Top-level directories:
- `datasets/`: where to unpack the dataset archive
- `monoprune/`: the code as an installable Python package
- `output_exp_smt/`: where synthesis output is saved for the Heuristic vs SMT experiment
- `output_exp_bfs/`: where synthesis output is saved for the Heuristic vs BFS experiment
- `results/`: where the final plots and table are placed
- `tmp/`: temporary storage for the scripts, may be deleted

The code, located in `monoprune/src/monoprune/`:
- `aexpr/`: defines an arithmetic language, which can be converted to SMT expressions, entered in `smt_check.py` at `smt_check()`
- `crim13_impl_dsl/`: vestigial, implmentation of the NEAR codebase's DSL
- `crim13_paper_dsl/`: vestigial, implementation of the NEAR paper's DSL
- `crim13_poly_dsl/`: the "polynomial" normalization of the NEAR DSL, described in our paper
- `exp_synth_param/`: synthesis via the SMT solver, entered via `run_exp.py`
- `exp_torch/`: utility functions used for Heuristic and BFS
- `quivr_dsl/`: the Quivr DSL, described in our paper
- `sb/`:
  - `bfs_exp_crim.py`, `bfs_exp_quivr.py`: builds the datasets for Heuristic vs BFS
  - `smt_exp_crim.py`, `smt_exp_emergency_quivr.py`: builds the datasets for Heuristic vs SMT
  - `run_smt_torch.py`: entry-point for the Heuristic part of Heuristic vs SMT
  - `run_bfs_exp.py`: entry-point for the Heuristic and BFS parts of Heuristic vs BFS
  - `run_bfs_all.py`, `run_smt_all.py`: scripts that run all experiments
  - `bfs_table.py`, `smt_plot.py`: scripts that generate plots and tables
- `./`: miscellaneous helper functions

## Extending to New Languages
In order to apply our algorithm to a new language, you must implement the language, a search graph over partial programs, a concrete semantics, and an abstract semantics. Some utilities from this codebase may be useful when doing this for languages with real-valued constants.

`exp_torch/common.py` defines some useful conceptual types:
```
IntervalState: TypeAlias = Tuple[T, Float[Tensor, "p"], Float[Tensor, "p"]]
ConcreteUtil: TypeAlias = Callable[[T, Float[Tensor, "p"]], float]
IntervalUtil: TypeAlias = Callable[
    [T, Float[Tensor, "p"], Float[Tensor, "p"]], Tuple[float, float]
]
```
- An `IntervalState` is a program sketch, a PyTorch tensor of lower bounds for the parameter holes in the sketch, and a PyTorch tensor of upper bounds.
- A `ConcreteUtil` evaluates a program sketch and a tensor of parameters to a real-valued utility.
- An `IntervalUtil` evaluates a program sketch and lower- and upper-bounds to lower- and upper-bounds on a utility.

In `sb/run_bfs_exp.py`, the following is the conceptual entry-point:
```
heuristic_opt = online_search(
    heuristic_search_key(),
    dbg_state_evaluator(util_f1, util_f1_interval),
    dbg_expand_state,
    sketches_with_bounds,
)
```
Taking `sketches_with_bounds` (a set of `IntervalState`s), `util_f1` (a `ConcreteUtil`), `util_f1_interval` (an `IntervalUtil`), and `dbg_expand_state` (a function from `IntervalState`s to sets of `IntervalState`s), it produces a Python iterator that performs heuristic search.

## Synthesizing Quivr Programs on New Datasets
Our implementation of the Quivr language ingests data in the form of Python objects serialized with PyTorch's `torch.load`/`torch.save`. Format it as follows,
where "dataset_name" and "task_name" are an identifiers for your dataset trajectories and task labels:

`datasets/quivr_hard/{dataset_name}/data.torch`
```
{
    'traces': {
        '{predicate_name}': torch tensor with
                            dimensions [n, m, m] where n is the number of traces in the dataset and m is the number of time steps per trace
                            dtype bool (for predicates with no parameters) or float32 (for predicates with parameters)
    },
    'pred1_bounds' :{
        '{predicate_name}': tuple of (lower bound, upper bound) where both are floats
                            only predicates with parameters occur here
    },
    'train_indices': torch int64 tensor with the indices (in 'traces') of the training examples
    'test_indices': torch int64 tensor with the indices (in 'traces') of the test examples
}
```

`datasets/quivr_hard/{dataset_name}/task_{task_name}/task.torch` (`torch.load`)
```
{
    'labels': torch bool tensor with dimensions [n] where n is the number of traces in the dataset
    'labeled_initial_indices': unused
}
```

To do synthesis, run `python3 -m monoprune.sb.run_bfs_exp quivr {dataset_name} {task_name} heuristic f1 {timeout_secs}`.

This will produce a CSV file at `output_exp_bfs/{dataset_name}_{task_name}_heuristic_f1_{timeout_secs}_bound` with format:
```
{line_number},{time},{f1_lower_bound},{f1_upper_bound},{_},{lower_bound_witness_partial_program}"
```
