import json

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
#    "test_regression_model"
]


# This automatically reads in the configuration
@hydra.main(version_base=None, config_name='config', config_path='.')  # Adding version_base for Python 3.13 compatibility
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                env_manager="conda",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
            # Clean the raw sample: drop price outliers and convert last_review to datetime
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning"),
                "main",
                env_manager="conda",
                parameters={
                    "input_artifact": "sample.csv:latest",
                    "output_artifact": "clean_sample.csv",
                    "output_type": "clean_sample",
                    "output_description": "Price outliers filtered and last_review converted; missing values retained",
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                },
            )

        if "data_check" in active_steps:
            # Run the data tests (pytest) on the cleaned data against the reference dataset
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "data_check"),
                "main",
                env_manager="conda",
                parameters={
                    "csv": "clean_sample.csv:latest",
                    "ref": "clean_sample.csv:reference",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                },
            )

        if "data_split" in active_steps:
            # Split the cleaned data into trainval_data.csv and test_data.csv
            _ = mlflow.run(
                f"{config['main']['components_repository']}/train_val_test_split",
                "main",
                env_manager="conda",
                parameters={
                    "input": "clean_sample.csv:latest",
                    "test_size": config["modeling"]["test_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                },
            )

        if "train_random_forest" in active_steps:

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest
            # step

            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "train_random_forest"),
                "main",
                env_manager="conda",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "val_size": config["modeling"]["val_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                    "rf_config": rf_config,
                    "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                    "output_artifact": "random_forest_export",
                },
            )

        if "test_regression_model" in active_steps:
            # Evaluate the model promoted to "prod" on the held-out test set
            _ = mlflow.run(
                f"{config['main']['components_repository']}/test_regression_model",
                "main",
                env_manager="conda",
                parameters={
                    "mlflow_model": "random_forest_export:prod",
                    "test_dataset": "test_data.csv:latest",
                },
            )

            # Explicitly compare test-set performance with validation-set performance
            # of the same "prod" model, and fail the pipeline if the model overfits
            verify_test_vs_validation(config)


def verify_test_vs_validation(config: DictConfig):
    """
    Compare the test-set metrics of the "prod" model (logged by test_regression_model)
    with the validation-set metrics logged by the train_random_forest run that produced it.

    The check passes when:
      * test MAE <= validation MAE * (1 + max_mae_increase_pct / 100)
      * test r2  >= validation r2 - max_r2_drop
    Tolerances come from the "evaluation" section of config.yaml (see README for the rationale).
    """
    api = wandb.Api()
    entity = os.environ.get("WANDB_ENTITY") or api.default_entity
    project = config["main"]["project_name"]

    # 1. Validation metrics: summary of the training run that logged the prod model
    prod_model = api.artifact(f"{entity}/{project}/random_forest_export:prod")
    train_run = prod_model.logged_by()
    val_r2 = float(train_run.summary["r2"])
    val_mae = float(train_run.summary["mae"])

    # 2. Test metrics: most recent finished test_model run that used this prod model
    test_runs = [
        r for r in prod_model.used_by()
        if r.job_type == "test_model" and r.state == "finished" and "mae" in r.summary
    ]
    if not test_runs:
        raise RuntimeError("No finished test_regression_model run found for random_forest_export:prod")
    test_run = max(test_runs, key=lambda r: r.created_at)
    test_r2 = float(test_run.summary["r2"])
    test_mae = float(test_run.summary["mae"])

    # 3. Compare against the configured tolerances
    max_mae_increase_pct = float(config["evaluation"]["max_mae_increase_pct"])
    max_r2_drop = float(config["evaluation"]["max_r2_drop"])

    mae_increase_pct = (test_mae - val_mae) / val_mae * 100
    r2_drop = val_r2 - test_r2
    mae_ok = mae_increase_pct <= max_mae_increase_pct
    r2_ok = r2_drop <= max_r2_drop
    passed = mae_ok and r2_ok

    verdict = "PASSED: no overfitting detected" if passed else "FAILED: test performance degrades beyond tolerance"
    lines = [
        "# Test vs validation performance check",
        "",
        f"Model: `random_forest_export:{prod_model.version}` (alias `prod`), "
        f"trained in run `{train_run.name}`, tested in run `{test_run.name}`",
        "",
        "| Metric | Validation | Test | Difference | Tolerance | Result |",
        "|---|---|---|---|---|---|",
        f"| MAE | {val_mae:.2f} | {test_mae:.2f} | {mae_increase_pct:+.1f}% | <= +{max_mae_increase_pct:.0f}% | {'OK' if mae_ok else 'FAIL'} |",
        f"| r2 | {val_r2:.3f} | {test_r2:.3f} | {-r2_drop:+.3f} | drop <= {max_r2_drop:.2f} | {'OK' if r2_ok else 'FAIL'} |",
        "",
        f"**{verdict}**",
    ]
    report = "\n".join(lines) + "\n"
    print("\n" + report)

    # Save the report in the repository root so it is versioned with the code
    with open(os.path.join(hydra.utils.get_original_cwd(), "test_vs_validation.md"), "w") as fp:
        fp.write(report)

    # Log the comparison to W&B as well
    run = wandb.init(project=project, job_type="test_vs_validation")
    run.summary.update({
        "val_mae": val_mae, "test_mae": test_mae, "mae_increase_pct": mae_increase_pct,
        "val_r2": val_r2, "test_r2": test_r2, "r2_drop": r2_drop, "passed": passed,
    })
    run.log({"test_vs_validation": wandb.Table(
        columns=["metric", "validation", "test", "difference", "tolerance", "ok"],
        data=[
            ["mae", val_mae, test_mae, f"{mae_increase_pct:+.1f}%", f"<= +{max_mae_increase_pct}%", mae_ok],
            ["r2", val_r2, test_r2, f"{-r2_drop:+.3f}", f"drop <= {max_r2_drop}", r2_ok],
        ],
    )})
    run.finish()

    if not passed:
        raise RuntimeError(verdict)


if __name__ == "__main__":
    go()
