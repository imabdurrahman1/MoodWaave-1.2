"""
experiments/rev3_a_stats/audit_p4.py
Audit of experiments/rev2_p4_baselines:
Examines code and fold configurations used in TASK P4.
"""

import json
import pandas as pd
import numpy as np

print("=== INNER CV AUDIT ===")
print("From experiments/rev2_p4_baselines/01_run_gtzan_baselines.py lines 62-72:")
print('''    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=grid,
        n_iter=8,
        cv=3,
        scoring="accuracy",
        random_state=inner_seed,
        n_jobs=1,
    )
    search.fit(X_tr, y_tr_fit)''')
print("When cv is integer 3 and estimator is a classifier, scikit-learn uses StratifiedKFold(n_splits=3).")
print("No groups parameter was passed to RandomizedSearchCV or fit().")
print("=> Inner tuning folds in GTZAN genre classification were PLAIN StratifiedKFold, NOT artist-grouped.")
print()

print("From experiments/rev2_p4_baselines/02_run_pmemo_baselines.py lines 54-64:")
print('''    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=grid,
        n_iter=8,
        cv=3,
        scoring="r2",
        random_state=inner_seed,
        n_jobs=1,
    )
    search.fit(X_tr, y_train)''')
print("When cv is integer 3 and estimator is a regressor, scikit-learn uses KFold(n_splits=3).")
print("No groups parameter was passed to RandomizedSearchCV or fit().")
print("=> Inner tuning folds in PMEmo regression were PLAIN KFold, NOT artist-grouped.")
print()

print("=== OUTER CV AUDIT ===")
with open("experiments/rev2_p1_splits/splits.json", "r") as f:
    splits_data = json.load(f)

gtzan_splits = splits_data["gtzan"]["fault_removed_artist_grouped"]
print(f"Total available GTZAN fault_removed_artist_grouped folds in splits.json: {len(gtzan_splits)}")
print("Repeats in splits.json:", sorted(set(f["repeat"] for f in gtzan_splits)))

df_g_p4 = pd.read_csv("experiments/rev2_p4_baselines/gtzan_baseline_folds.csv")
print(f"Records in P4 gtzan_baseline_folds.csv: {len(df_g_p4)}")
print("P4 outer repeats evaluated:", sorted(df_g_p4["repeat"].unique()))
print("P4 outer folds evaluated:", sorted(df_g_p4["fold"].unique()))
print(f"=> P4 evaluated only repeat 0 (5 outer folds) for GTZAN!")
print()

pmemo_splits = splits_data["pmemo"]["artist_grouped_5x5"]
print(f"Total available PMEmo artist_grouped_5x5 folds in splits.json: {len(pmemo_splits)}")
print("Repeats in splits.json:", sorted(set(f["repeat"] for f in pmemo_splits)))

df_p_p4 = pd.read_csv("experiments/rev2_p4_baselines/pmemo_baseline_folds.csv")
print(f"Records in P4 pmemo_baseline_folds.csv: {len(df_p_p4)}")
print("P4 outer repeats evaluated:", sorted(df_p_p4["repeat"].unique()))
print("P4 outer folds evaluated:", sorted(df_p_p4["fold"].unique()))
print(f"=> P4 evaluated repeats 0 and 1 (10 outer folds, 5x2) for PMEmo!")

