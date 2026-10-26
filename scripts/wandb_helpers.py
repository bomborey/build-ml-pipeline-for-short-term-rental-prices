#!/usr/bin/env python
"""
Helpers that automate the manual W&B steps of the project.

    python scripts/wandb_helpers.py entity              -> print your W&B entity
    python scripts/wandb_helpers.py reference           -> add "reference" alias to clean_sample.csv:latest
    python scripts/wandb_helpers.py promote_best        -> add "prod" alias to the lowest-MAE model and
                                                           write its hyperparameters into config.yaml
"""
import re
import sys

import wandb

PROJECT = "nyc_airbnb"


def entity():
    return wandb.Api().default_entity


def add_reference_alias():
    api = wandb.Api()
    art = api.artifact(f"{entity()}/{PROJECT}/clean_sample.csv:latest")
    if "reference" not in art.aliases:
        art.aliases.append("reference")
        art.save()
    print(f"'reference' alias set on clean_sample.csv:{art.version}")


def promote_best():
    api = wandb.Api()
    runs = api.runs(f"{entity()}/{PROJECT}")
    scored = [r for r in runs
              if r.job_type == "train_random_forest" and r.state == "finished" and "mae" in r.summary]
    if not scored:
        sys.exit("No finished train_random_forest runs with an 'mae' value were found.")

    best = min(scored, key=lambda r: r.summary["mae"])
    max_tfidf = int(best.config["max_tfidf_features"])
    max_features = best.config["max_features"]
    print(f"Best run: {best.name}  mae={best.summary['mae']:.3f}  "
          f"max_tfidf_features={max_tfidf}  max_features={max_features}")

    # Move the "prod" alias to the model produced by the best run
    for art in best.logged_artifacts():
        if art.name.startswith("random_forest_export"):
            if "prod" not in art.aliases:
                art.aliases.append("prod")
                art.save()
            print(f"'prod' alias set on {art.name}")
            break
    else:
        sys.exit("The best run did not log a random_forest_export artifact.")

    # Save the best hyperparameters as the defaults in config.yaml (comments preserved)
    with open("config.yaml") as fp:
        cfg = fp.read()
    cfg = re.sub(r"(\n  max_tfidf_features: )\S+", rf"\g<1>{max_tfidf}", cfg)
    cfg = re.sub(r"(\n    max_features: )\S+", rf"\g<1>{max_features}", cfg)
    with open("config.yaml", "w") as fp:
        fp.write(cfg)
    print("config.yaml updated with the best hyperparameters")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "entity":
        print(entity())
    elif cmd == "reference":
        add_reference_alias()
    elif cmd == "promote_best":
        promote_best()
    else:
        sys.exit(__doc__)
