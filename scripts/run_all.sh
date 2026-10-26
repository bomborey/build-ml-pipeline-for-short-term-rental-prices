#!/usr/bin/env bash
# Runs the whole project end to end: pipeline, W&B aliases, best hyperparameters,
# releases 1.0.0 and 1.0.1, the sample2 runs, and the submission ZIP.
# Usage (from the repository root):   bash scripts/run_all.sh
set -euo pipefail

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

# ---------- preflight ----------
[ -f main.py ] || { echo "Run this from the repository root."; exit 1; }
[ -n "${WANDB_API_KEY:-}" ] || { echo "Set WANDB_API_KEY first:  export WANDB_API_KEY=..."; exit 1; }
REMOTE=$(git remote get-url origin | sed -E 's#https://[^@/]*@#https://#')
case "$REMOTE" in *udacity/build-ml-pipeline*) echo "origin points to the Udacity repo. Clone YOUR fork instead."; exit 1;; esac
REPO_URL="${REMOTE%.git}"
ENTITY=$(python scripts/wandb_helpers.py entity)
WANDB_URL="https://wandb.ai/${ENTITY}/nyc_airbnb"
echo "Repository: $REPO_URL"
echo "W&B:        $WANDB_URL"
git config user.name  >/dev/null || git config user.name  "Rey Rosales"
git config user.email >/dev/null || git config user.email "rosalesr2@macewan.ca"
BRANCH=$(git rev-parse --abbrev-ref HEAD)

# ---------- pipeline ----------
say "1/9 Download and clean"
mlflow run . -P steps=download,basic_cleaning

say "2/9 Mark clean_sample.csv:latest as 'reference'"
python scripts/wandb_helpers.py reference

say "3/9 Data checks, split and first training run"
mlflow run . -P steps=data_check,data_split,train_random_forest

say "4/9 Hyperparameter sweep (15 runs, this takes a while)"
mlflow run . -P steps=train_random_forest \
  -P hydra_options="modeling.max_tfidf_features=10,15,30 modeling.random_forest.max_features=0.1,0.33,0.5,0.75,1 -m"

say "5/9 Promote best model to 'prod' and save its hyperparameters in config.yaml"
python scripts/wandb_helpers.py promote_best

say "6/9 Evaluate the prod model on the test set"
mlflow run . -P steps=test_regression_model

# ---------- release 1.0.0 ----------
say "7/9 Commit, push and tag release 1.0.0"
sed -i "s#https://wandb.ai/<your-wandb-entity>/nyc_airbnb#${WANDB_URL}#; s#https://github.com/<your-github-username>/build-ml-pipeline-for-short-term-rental-prices#${REPO_URL}#" README.md
git add -A
git commit -m "Complete ML pipeline with best hyperparameters from the sweep"
git push origin "$BRANCH"
git tag -a 1.0.0 -m "Pipeline release 1.0.0"
git push origin 1.0.0

say "8/9 Run release 1.0.0 on sample2.csv (EXPECTED to fail on test_proper_boundaries)"
if mlflow run "$REMOTE" -v 1.0.0 -P hydra_options="etl.sample='sample2.csv'"; then
  echo "Unexpected: 1.0.0 did not fail on sample2.csv"
else
  echo "Release 1.0.0 failed on sample2.csv, as intended."
fi

# ---------- release 1.0.1 ----------
say "9/9 Add the NYC boundary filter, release 1.0.1 and run it on sample2.csv"
git apply - <<'PATCH'
diff --git a/src/basic_cleaning/run.py b/src/basic_cleaning/run.py
index c4cb5bb..065670b 100644
--- a/src/basic_cleaning/run.py
+++ b/src/basic_cleaning/run.py
@@ -33,6 +33,13 @@ def go(args):
     logger.info("Converting last_review to datetime")
     df["last_review"] = pd.to_datetime(df["last_review"])
 
+    # Drop listings outside the NYC area (added in release 1.0.1 after sample2.csv
+    # failed the test_proper_boundaries data test)
+    logger.info("Removing listings outside the NYC geographic boundaries")
+    idx = df["longitude"].between(-74.25, -73.50) & df["latitude"].between(40.5, 41.2)
+    df = df[idx].copy()
+    logger.info("%d rows remain after boundary filtering", df.shape[0])
+
     # NOTE: missing values are intentionally NOT imputed here. Imputation is part of
     # the inference pipeline so the same logic is applied at prediction time.
 
PATCH
git add src/basic_cleaning/run.py
git commit -m "Filter listings outside NYC geographic boundaries"
git push origin "$BRANCH"
git tag -a 1.0.1 -m "Pipeline release 1.0.1"
git push origin 1.0.1
mlflow run "$REMOTE" -v 1.0.1 -P hydra_options="etl.sample='sample2.csv'"

if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
  gh release create 1.0.0 --verify-tag --title "1.0.0" --notes "First release of the pipeline" || true
  gh release create 1.0.1 --verify-tag --title "1.0.1" --notes "Adds NYC geographic boundary filter" || true
fi

# ---------- submission ZIP ----------
say "Building the submission ZIP from a fresh clone"
cd ..
rm -rf nyc-airbnb-submission nyc-airbnb-submission.zip
git clone "$REMOTE" nyc-airbnb-submission
( cd nyc-airbnb-submission && git fetch --all --tags && git fsck --full && git tag -n )
zip -qr nyc-airbnb-submission.zip nyc-airbnb-submission -x "*/.venv/*" "*/__pycache__/*" "*/.DS_Store"
unzip -l nyc-airbnb-submission.zip | grep -q "/.git/HEAD" && echo "ZIP contains .git: OK"

cat <<DONE

All done. Two manual clicks left:
  1. W&B: $WANDB_URL  -> project settings -> make the project PUBLIC
  2. Download $(pwd)/nyc-airbnb-submission.zip and upload it on the Udacity page.

Paste this into "Submission Details":
Repository platform: GitHub
Repository URL: $REPO_URL
W&B project: $WANDB_URL
DONE
