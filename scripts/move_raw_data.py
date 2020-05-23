from os import path, replace, scandir


def noext(f):
    return path.splitext(f.name)[0]


raw_task1_files = list(
    sorted(scandir("raw-data/0325updated.task1train(626p)"), key=lambda f: f.name)
)
raw_task2_files = list(
    sorted(scandir("raw-data/0325updated.task2train(626p)"), key=lambda f: f.name)
)

jpg_files = [f for f in raw_task1_files if f.name.endswith("jpg")]
csv_files = [f for f in raw_task1_files if f.name.endswith("txt")]
json_files = [f for f in raw_task2_files if f.name.endswith("txt")]

for i, (f_jpg, f_csv, f_json) in enumerate(zip(jpg_files, csv_files, json_files)):
    if noext(f_jpg) != noext(f_csv) or noext(f_csv) != noext(f_json):
        raise ValueError("Raw data filenames mismatch")

    print(f"{i:03d}", f_jpg, f_csv, f_json)

    replace(f_jpg.path, f"data/{i:03d}.jpg")
    replace(f_csv.path, f"data/{i:03d}.csv")
    replace(f_json.path, f"data/{i:03d}.json")
