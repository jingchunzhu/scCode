import yaml
from jinja2 import Template
import subprocess
from pathlib import Path

CONFIG_FILE = "run_benchmark_fixed.yaml"


def load_config(path):
    """Render Jinja2 template and load YAML."""
    with open(path) as f:
        template = Template(f.read())

    # First pass: load raw YAML to get top-level variables
    raw = yaml.safe_load(open(path).read().replace("{{", "PLACEHOLDER_OPEN").replace("}}", "PLACEHOLDER_CLOSE"))

    # Normalize K to a list
    k_values = raw.get("K", [30])
    if isinstance(k_values, (int, float)):
        k_values = [int(k_values)]

    # Normalize ontology_method to a list
    methods = raw.get("ontology_method", ["ic"])
    if isinstance(methods, str):
        methods = [methods]

    # For each K value, render the template and collect models
    all_models = []
    for k in k_values:
        rendered = template.render(
            dataset=raw.get("dataset", ""),
            ref=raw.get("ref", ""),
            FM=raw.get("FM", ""),
            K=k,
            ground_truth=raw.get("ground_truth", ""),
        )
        cfg_k = yaml.safe_load(rendered)
        for model in cfg_k["models"]:
            model["K"] = k
            model["ontology_methods"] = methods
            all_models.append(model)

    # Build final config from last rendered pass, then attach expanded models
    cfg = cfg_k
    cfg["models"] = all_models

    return cfg


def run_predict(model):
    pred_file = Path(model["prediction_file"])
    pred_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Predict: {model['name']} ===")

    subprocess.run([
        "python3", "FoundationModelBenchmarking/predict.py",
        "--index", model["index"],
        "--ref_annot", model["ref_annot"],
        "--obo", model["obo"],
        "--embeddings", model["embeddings"],
        "--k", str(model["K"]),
        "--metadata", model["metadata"],
        "--output", str(pred_file),
    ], check=True)


def run_background_distances(model):
    output_png = Path(model["outputdir"]) / "background_vs_prediction_distances.png"
    output_png.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Background Distances: {model['name']} ===")

    subprocess.run([
        "python3", "FoundationModelBenchmarking/background_distances.py",
        "--embeddings", model["ref_embedding"],
        "--predictions", str(model["prediction_file"]),
        "--output", str(output_png),
    ], check=True)


def run_background_ic(model):
    eval_dir = Path(model["outputdir"]) / "ic"
    eval_file = eval_dir / "per_cell_evaluation.tsv"
    output_png = eval_dir / "background_vs_prediction_ic.png"
    eval_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Background IC: {model['name']} ===")

    subprocess.run([
        "python3", "FoundationModelBenchmarking/background_ic.py",
        "--ref-annot", model["ref_annot"],
        "--obo", model["obo"],
        "--evaluation", str(eval_file),
        "--output", str(output_png),
    ], check=True)


def run_evaluate(model, ontology_method):
    pred_file = Path(model["prediction_file"])
    eval_dir = Path(model["outputdir"]) / ontology_method
    eval_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Evaluate: {model['name']} (method: {ontology_method}) ===")

    subprocess.run([
        "python3", "FoundationModelBenchmarking/evaluate.py",
        "--predictions", str(pred_file),
        "--ground_truth", model["ground_truth_file"],
        "--obo", model["obo"],
        "--ontology-method", ontology_method,
        "--output-dir", str(eval_dir),
    ], check=True)


def main():
    cfg = load_config(CONFIG_FILE)
    tasks = cfg.get("run", ["predict", "evaluate"])

    for model in cfg["models"]:
        outputdir = Path(model["outputdir"])
        outputdir.mkdir(parents=True, exist_ok=True)

        if "predict" in tasks:
            run_predict(model)

        if "background_distances" in tasks:
            run_background_distances(model)

        if "evaluate" in tasks:
            for method in model["ontology_methods"]:
                run_evaluate(model, method)

        if "background_ic" in tasks:
            run_background_ic(model)

    print("\nAll tasks completed.\n")


if __name__ == "__main__":
    main()
