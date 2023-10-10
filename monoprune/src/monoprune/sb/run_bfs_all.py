import os

tasks = [
    ("crim", "crim", "a"),
    ("crim", "crim", "b"),
    ("quivr", "monoprune_mabe22", "approach"),
    ("quivr", "monoprune_mabe22", "chase"),
    ("quivr", "monoprune_mabe22", "nose_ear_contact"),
    ("quivr", "monoprune_mabe22", "nose_genital_contact"),
    ("quivr", "monoprune_mabe22", "nose_nose_contact"),
    ("quivr", "monoprune_mabe22", "watching"),
    ("quivr", "maritime_surveillance", "a"),
    ("quivr", "monoprune_shibuya_one", "east_to_north"),
    ("quivr", "monoprune_shibuya_one", "north_to_west"),
    ("quivr", "monoprune_shibuya_one", "south_to_east"),
    ("quivr", "monoprune_shibuya_one", "west_to_south"),
    ("quivr", "monoprune_warsaw_one", "eastward_3_high_accel"),
    ("quivr", "monoprune_warsaw_one", "eastward_4_high_accel"),
    ("quivr", "monoprune_warsaw_two", "eastward_3_passing_eastward_4"),
    ("quivr", "monoprune_warsaw_two", "eastward_4_passing_eastward_3"),
    ("quivr", "monoprune_warsaw_two", "parallel_eastward_4_eastward_3"),
    ("quivr", "monoprune_warsaw_two", "southward_1_upper_westward_2_rightonred"),
]


def run_and_print(c: str):
    print(f'=== running "{c}" ===')
    return os.system(c)


timeout_secs = "600"
util = "f1"
for m, d, t in tasks:
    for approach in {"heuristic", "bfs"}:
        run_and_print(
            f"python -m monoprune.sb.run_bfs_exp {m} {d} {t} {approach} {util} {timeout_secs}"
        )
