tasks_with_labels = {
    ("crim", "crim", "a"): "CA",
    ("crim", "crim", "b"): "CB",
    ("quivr", "monoprune_mabe22", "approach"): "QA",
    ("quivr", "monoprune_mabe22", "chase"): "QB",
    ("quivr", "monoprune_mabe22", "nose_ear_contact"): "QC",
    ("quivr", "monoprune_mabe22", "nose_genital_contact"): "QD",
    ("quivr", "monoprune_mabe22", "nose_nose_contact"): "QE",
    ("quivr", "monoprune_mabe22", "watching"): "QF",
    ("quivr", "maritime_surveillance", "a"): "QG",
    ("quivr", "monoprune_shibuya_one", "east_to_north"): "QH",
    ("quivr", "monoprune_shibuya_one", "north_to_west"): "QI",
    ("quivr", "monoprune_shibuya_one", "south_to_east"): "QJ",
    ("quivr", "monoprune_shibuya_one", "west_to_south"): "QK",
    ("quivr", "monoprune_warsaw_one", "eastward_3_high_accel"): "QL",
    ("quivr", "monoprune_warsaw_one", "eastward_4_high_accel"): "QM",
    ("quivr", "monoprune_warsaw_two", "eastward_3_passing_eastward_4"): "QN",
    ("quivr", "monoprune_warsaw_two", "eastward_4_passing_eastward_3"): "QO",
    ("quivr", "monoprune_warsaw_two", "parallel_eastward_4_eastward_3"): "QP",
    ("quivr", "monoprune_warsaw_two", "southward_1_upper_westward_2_rightonred"): "QQ",
}

res = {
    "heuristic": {k: [] for k in tasks_with_labels.keys()},
    "bfs": {k: [] for k in tasks_with_labels.keys()},
}

arg_timeout_secs_str = "600"
for m, d, t in tasks_with_labels.keys():
    ls = {}
    for approach in ["heuristic", "bfs"]:
        p = f"output_exp_bfs/{d}_{t}_{approach}_f1_{arg_timeout_secs_str}_bound"  # type: ignore
        try:
            with open(p, "r") as f:
                res[approach][m, d, t].extend(
                    tuple(
                        float(c) if float(c) == float(c) else 1.0 for c in l.split(",")[1:4]
                    )
                    for l in f.readlines()
                )
            print(f"Read {p}")
        except FileNotFoundError:
            print(f"Failed to read {p}")


def get_lb_ub_at_time(l, targ_t):
    for t, lb, ub in l:
        if t >= targ_t:
            return lb, ub
    if len(l) > 0:
        return l[-1][1], l[-1][2]
    else:
        return float("nan"), float("nan")


def cond_bold(b, s):
    if b:
        return "\\textbf{" + s + "}"
    else:
        return s


targ_ts = [
    (10, "10 s"),
    (30, "30 s"),
    (60, "1 m"),
    (60 * 2, "2 m"),
    (60 * 5, "5 m"),
    (60 * 10, "10 m"),
]

width = 2 * len(targ_ts) + 2
height = 2 * len(tasks_with_labels) + 1
cells = [[None for j in range(width)] for i in range(height)]

cells[0][0] = ""
cells[0][1] = ""
cells[1][0] = ""
for i, ((m, d, t), l) in enumerate(tasks_with_labels.items()):
    cells[2 * i + 1][0] = l
    cells[2 * i + 2][0] = ""

    cells[2 * i + 1][1] = f"H"
    cells[2 * i + 2][1] = f"B"

for j, (targ_t, targ_t_disp) in enumerate(targ_ts):
    cells[0][2 * j + 2] = targ_t_disp
    cells[0][2 * j + 3] = ""
    for i, ((m, d, t), l) in enumerate(tasks_with_labels.items()):
        bfs_lb, bfs_ub = get_lb_ub_at_time(res["bfs"][m, d, t], targ_t)
        bfs_r = bfs_ub - bfs_lb
        h_lb, h_ub = get_lb_ub_at_time(res["heuristic"][m, d, t], targ_t)
        h_r = h_ub - h_lb
        h_bigger = h_lb > bfs_lb
        h_r_smaller = h_r < bfs_r
        bfs_bigger = h_lb < bfs_lb
        bfs_r_smaller = h_r > bfs_r
        cells[2 * i + 1][2 * j + 2] = cond_bold(h_bigger, f"{h_lb:.2f}")
        cells[2 * i + 1][2 * j + 3] = cond_bold(h_r_smaller, f"({(h_r):.2f})")
        cells[2 * i + 2][2 * j + 2] = cond_bold(bfs_bigger, f"{bfs_lb:.2f}")
        cells[2 * i + 2][2 * j + 3] = cond_bold(bfs_r_smaller, f"({(bfs_r):.2f})")

with open("results/bfs_table.tex", "w") as f:
    f.write(
        "% This file is programatically generated, so editing it manually is inadvisable.\n"
    )
    f.write("\\begin{tabular}{l" + ("r" * (width - 1)) + "}\n")
    f.write("\\toprule\n")
    f.write("\n".join(" & ".join(r) + " \\\\" for r in cells))
    f.write("\n")
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
