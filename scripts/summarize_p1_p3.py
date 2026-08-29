from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "outputs" / "micro_expression"
GROUP_DIR = OUT_ROOT / "group_summary"
GROUP_DIR.mkdir(parents=True, exist_ok=True)

PARTICIPANTS = ["p1", "p2", "p3"]
PROTOCOLS = ["upper", "lower"]
MODALITIES = ["semantic", "geometry", "dino"]


# ============================================================
# 1. Trial metrics
# ============================================================

trial_frames = []

for participant in PARTICIPANTS:
    for protocol in PROTOCOLS:
        path = (
            OUT_ROOT
            / participant
            / "s01"
            / f"{protocol}_trial_metrics_v3.csv"
        )

        if not path.exists():
            print(f"[MISSING] {path}")
            continue

        df = pd.read_csv(path)
        df.insert(0, "participant", participant)
        df.insert(1, "protocol", protocol)
        trial_frames.append(df)

if not trial_frames:
    raise RuntimeError("No upper/lower trial metrics found.")

trials = pd.concat(trial_frames, ignore_index=True)

trials.to_csv(
    GROUP_DIR / "all_trial_metrics.csv",
    index=False,
)


# ============================================================
# 2. Participant × action × modality summary
# ============================================================

rows = []

for participant in PARTICIPANTS:
    p_df = trials[trials["participant"] == participant]

    for action in sorted(p_df["action"].dropna().unique()):
        a_df = p_df[p_df["action"] == action]

        for modality in MODALITIES:
            sep_col = f"{modality}_hold_auc_separability"
            pol_col = f"{modality}_hold_polarity"

            if sep_col not in a_df.columns:
                continue

            sep = pd.to_numeric(
                a_df[sep_col],
                errors="coerce",
            ).dropna()

            polarities = (
                a_df[pol_col]
                .dropna()
                .astype(str)
            )

            if len(polarities):
                polarity_consistency = (
                    polarities.value_counts().iloc[0]
                    / len(polarities)
                )
                dominant_polarity = polarities.value_counts().index[0]
            else:
                polarity_consistency = np.nan
                dominant_polarity = ""

            qc = pd.to_numeric(
                a_df["qc_pass"],
                errors="coerce",
            ).dropna()

            rows.append(
                {
                    "participant": participant,
                    "action": action,
                    "modality": modality,
                    "repeat_count": len(a_df),
                    "mean_separability": sep.mean() if len(sep) else np.nan,
                    "min_separability": sep.min() if len(sep) else np.nan,
                    "std_separability": sep.std(ddof=0) if len(sep) else np.nan,
                    "dominant_polarity": dominant_polarity,
                    "polarity_consistency": polarity_consistency,
                    "qc_pass_rate": qc.mean() if len(qc) else np.nan,
                }
            )

subject_summary = pd.DataFrame(rows)

subject_summary.to_csv(
    GROUP_DIR / "subject_action_modality_summary.csv",
    index=False,
)


# ============================================================
# 3. Cross-subject summary
# ============================================================

group_rows = []

for (action, modality), g in subject_summary.groupby(
    ["action", "modality"]
):
    subject_means = pd.to_numeric(
        g["mean_separability"],
        errors="coerce",
    ).dropna()

    subject_mins = pd.to_numeric(
        g["min_separability"],
        errors="coerce",
    ).dropna()

    polarities = (
        g["dominant_polarity"]
        .replace("", np.nan)
        .dropna()
        .astype(str)
    )

    if len(polarities):
        cross_subject_polarity_consistency = (
            polarities.value_counts().iloc[0]
            / len(polarities)
        )
        dominant_polarity = polarities.value_counts().index[0]
    else:
        cross_subject_polarity_consistency = np.nan
        dominant_polarity = ""

    group_rows.append(
        {
            "action": action,
            "modality": modality,
            "subjects": len(g),
            "mean_subject_separability": (
                subject_means.mean()
                if len(subject_means)
                else np.nan
            ),
            "worst_subject_mean_separability": (
                subject_means.min()
                if len(subject_means)
                else np.nan
            ),
            "worst_single_repeat_separability": (
                subject_mins.min()
                if len(subject_mins)
                else np.nan
            ),
            "between_subject_std": (
                subject_means.std(ddof=0)
                if len(subject_means)
                else np.nan
            ),
            "dominant_polarity": dominant_polarity,
            "cross_subject_polarity_consistency": (
                cross_subject_polarity_consistency
            ),
            "mean_qc_pass_rate": pd.to_numeric(
                g["qc_pass_rate"],
                errors="coerce",
            ).mean(),
        }
    )

group_summary = pd.DataFrame(group_rows)

group_summary.to_csv(
    GROUP_DIR / "cross_subject_action_summary.csv",
    index=False,
)


# ============================================================
# 4. Repeat consistency
# ============================================================

consistency_frames = []

for participant in PARTICIPANTS:
    for protocol in PROTOCOLS:
        path = (
            OUT_ROOT
            / participant
            / "s01"
            / f"{protocol}_repeat_consistency_v3.csv"
        )

        if not path.exists():
            print(f"[MISSING] {path}")
            continue

        df = pd.read_csv(path)
        df.insert(0, "participant", participant)
        df.insert(1, "protocol", protocol)
        consistency_frames.append(df)

if consistency_frames:
    consistency = pd.concat(
        consistency_frames,
        ignore_index=True,
    )

    consistency.to_csv(
        GROUP_DIR / "all_repeat_consistency.csv",
        index=False,
    )

    repeat_group = (
        consistency
        .groupby(["action", "signal"])
        .agg(
            subjects=("participant", "nunique"),
            mean_repeat_corr=("mean_pairwise_corr", "mean"),
            worst_repeat_corr=("min_pairwise_corr", "min"),
            mean_amplitude_cv=(
                "hold_absolute_amplitude_cv",
                "mean",
            ),
            mean_polarity_consistency=(
                "hold_polarity_consistency",
                "mean",
            ),
        )
        .reset_index()
    )

    repeat_group.to_csv(
        GROUP_DIR / "cross_subject_repeat_summary.csv",
        index=False,
    )
else:
    repeat_group = pd.DataFrame()


# ============================================================
# 5. Control v3.1 matrices
# ============================================================

control_rows = []

for participant in PARTICIPANTS:
    for modality in MODALITIES:
        path = (
            OUT_ROOT
            / participant
            / "s01"
            / f"control_false_matrix_{modality}_v31.csv"
        )

        if not path.exists():
            print(f"[MISSING CONTROL] {path}")
            continue

        matrix = pd.read_csv(path, index_col=0)

        for nuisance_label, row in matrix.iterrows():
            for action, value in row.items():
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue

                if not np.isfinite(value):
                    continue

                control_rows.append(
                    {
                        "participant": participant,
                        "modality": modality,
                        "nuisance_label": str(nuisance_label),
                        "action": str(action),
                        "false_activation_rate": value,
                    }
                )

control = pd.DataFrame(control_rows)

if not control.empty:
    control.to_csv(
        GROUP_DIR / "all_control_false_activation.csv",
        index=False,
    )

    control_summary = (
        control
        .groupby(["action", "modality"])
        .agg(
            mean_false_activation=(
                "false_activation_rate",
                "mean",
            ),
            max_false_activation=(
                "false_activation_rate",
                "max",
            ),
            p95_false_activation=(
                "false_activation_rate",
                lambda x: np.quantile(x, 0.95),
            ),
        )
        .reset_index()
    )

    control_summary.to_csv(
        GROUP_DIR / "control_false_activation_summary.csv",
        index=False,
    )
else:
    control_summary = pd.DataFrame()


# ============================================================
# 6. Console report
# ============================================================

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 220)

print("\n")
print("=" * 90)
print("CROSS-SUBJECT ACTION SEPARABILITY")
print("=" * 90)

show = group_summary[
    [
        "action",
        "modality",
        "mean_subject_separability",
        "worst_subject_mean_separability",
        "worst_single_repeat_separability",
        "dominant_polarity",
        "cross_subject_polarity_consistency",
        "mean_qc_pass_rate",
    ]
].sort_values(
    ["action", "modality"]
)

print(show.to_string(index=False))


if not repeat_group.empty:
    print("\n")
    print("=" * 90)
    print("REPEAT CONSISTENCY")
    print("=" * 90)

    print(
        repeat_group.sort_values(
            ["action", "signal"]
        ).to_string(index=False)
    )


if not control_summary.empty:
    print("\n")
    print("=" * 90)
    print("CONTROL FALSE ACTIVATION")
    print("=" * 90)

    print(
        control_summary.sort_values(
            ["action", "modality"]
        ).to_string(index=False)
    )

    print("\n")
    print("=" * 90)
    print("TOP 30 CONTROL FALSE ACTIVATIONS")
    print("=" * 90)

    print(
        control.sort_values(
            "false_activation_rate",
            ascending=False,
        )
        .head(30)
        .to_string(index=False)
    )


print("\nSaved to:")
print(GROUP_DIR)
