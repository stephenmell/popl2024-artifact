import os


def run_and_print(c: str):
    print(f'=== running "{c}" ===')
    return os.system(c)


timeout = 1000000 # really this should be infinity
for seed in [0]:
    for d in [
        "emergency_quivr",
        "crim13",
    ]:
        if d == "emergency_quivr":
            n_exs = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        elif d == "crim13":
            n_exs = [3, 4, 6, 8, 10, 12, 14, 16, 18, 20]
            # n_exs = [3, 4, 6]
        for n_ex in n_exs:
            for approach in [
                "interval",
                "smt",
            ]:
                if approach == "interval":
                    if d == "emergency_quivr":
                        run_and_print(
                            f"python -m monoprune.sb.run_smt_torch emergency_quivr maritime_surveillance a heuristic {timeout} {n_ex} {seed}"
                        )
                    elif d == "crim13":
                        run_and_print(
                            f"python -m monoprune.sb.run_smt_torch crim crim a heuristic {timeout} {n_ex} {seed}"
                        )
                    else:
                         assert False
                else:
                    run_and_print(
                        f"python -m monoprune.exp_synth_param.run_exp smt {d} {n_ex} {seed}"
                    )
