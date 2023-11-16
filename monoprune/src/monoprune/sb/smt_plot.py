res_marisurv = []
res_crim = []

seed = 0
for d in [
    "marisurv",
    "crim13",
]:
    if d == "marisurv":
        n_exes = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    elif d == "crim13":
        n_exes = [3, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    else:
        assert False
    for n_ex in n_exes:
        res = res_marisurv if d == "marisurv" else res_crim
        ls = {}
        for approach in ["interval", "smt"]:
            if approach == "interval":
                if d == "marisurv":
                    p = f"output_exp_smt/maritime_surveillance_a_{n_ex}_{seed}_heuristic_f1_1000000_bound"
                elif d == "crim13":
                    p = f"output_exp_smt/crim_a_{n_ex}_{seed}_heuristic_f1_1000000_bound"
                else:
                    assert False
            elif approach == "smt":
                if d == "marisurv":
                    p = f"output_exp_smt/smt_emergency_quivr_{n_ex}_{seed}"
                elif d == "crim13":
                    p = f"output_exp_smt/smt_crim13_{n_ex}_{seed}"
                else:
                    assert False
            else:
                assert False
            try:
                with open(p, "r") as f:
                    ls[approach] = f.readlines()
            except FileNotFoundError:
                ls[approach] = None
            if ls[approach] is None or len(ls[approach]) == 0:
                print(f"Warning, missing: {p}")
                ls[approach] = ["0,nan,nan,nan,nan,\n"]

        _, int_t_str, int_lb_str, int_ub_str = ls["interval"][-1].split(",")[0:4]
        int_t, int_lb, int_ub = (
            float(int_t_str),
            float(int_lb_str),
            float(int_ub_str),
        )
        _, smt_t_str, smt_lb_str, smt_ub_str = ls["smt"][-1].split(",")[0:4]
        smt_t, smt_lb, smt_ub = (
            float(smt_t_str),
            float(smt_lb_str),
            float(smt_ub_str),
        )
        print(d, n_ex, int_t, int_lb_str, int_ub_str)
        print(d, n_ex, smt_t, smt_lb_str, smt_ub_str)
        res.append((n_ex, int_t, smt_t))

import plotly.express as px
import pandas as pd

df_marisurv = pd.DataFrame(res_marisurv, columns=["n", "int_t", "smt_t"])
df_crim = pd.DataFrame(res_crim, columns=["n", "int_t", "smt_t"])

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

print(df_marisurv)
print(df_crim)

for df, disp_name, ticks in [
    (df_marisurv, "quivr_g", [5, 20, 40, 60, 80, 100]),
    (df_crim, "crim_a", [3, 5, 10, 15, 20]),
]:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df["n"],
            y=df["int_t"],
            name="Heuristic",
            mode="lines",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df["n"],
            y=df["smt_t"],
            name="SMT",
            mode="lines",
            line={"dash": "dash"},
        ),
        secondary_y=False,
    )

    fig.update_layout(
        showlegend=False,
        font_family="Linux Biolinum",
        font_color="black",
        autosize=False,
        width=350,
        height=250,
        plot_bgcolor="rgba(0, 0, 0, 0)",
        margin=go.layout.Margin(
            l=0,
            r=0,
            b=0,
            t=0,
        ),
    )
    fig.update_xaxes(
        title_text="Number of Training Examples",
        tickvals=ticks,
        griddash="dot",
        gridwidth=0.5,
        gridcolor="black",
        linewidth=0.5,
        linecolor="black",
    )
    fig.update_yaxes(
        title_text="Time to Convergence (sec)",
        secondary_y=False,
        type="log",
        griddash="dot",
        gridwidth=0.5,
        gridcolor="black",
        ticksuffix="  ",
    )
    # fig.show()
    # Plotly PDF export has an issue (https://github.com/plotly/plotly.py/issues/3469) on the first write
    fig.write_image(f"results/fig_{disp_name}.pdf")
    time.sleep(1)
    fig.write_image(f"results/fig_{disp_name}.pdf")
