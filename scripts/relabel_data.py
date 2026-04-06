import os

def run_relabel():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print("This project now uses data/app_label.json as the source of app labels.")
    print("No filename-based relabeling is needed anymore.")
    print(
        "To rebuild processed features and refresh samples.npz, run "
        f"{os.path.join(project_root, 'scripts', 'main_preprocess.py')}"
    )


if __name__ == "__main__":
    run_relabel()
