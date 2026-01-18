import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    from scipy.stats import binomtest
except Exception:
    binomtest = None


def read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def summarize(rows, alpha=0.01):
    groups = defaultdict(list)
    for r in rows:
        key = (
            r.get("condition"),
            r.get("protocol", "independent"),
            r.get("calibration"),
            r.get("restrict_format"),
            r.get("blank_image"),
        )
        groups[key].append(r)

    summaries = []
    for key, rs in groups.items():
        condition, protocol, calibration, restrict, blank = key
        reject = []
        crvs = []
        p_specials = []
        exp = []

        joint_valid = []
        joint_all = []
        invalid_digit_rates = []
        invalid_color_rates = []
        invalid_rates = []

        total_special = 0
        total_n = 0
        total_correct = 0
        total_valid_pair = 0

        for r in rs:
            pv = r["chi2"]["p_value"]
            if pv is None or (isinstance(pv, float) and np.isnan(pv)):
                pass
            else:
                reject.append(1 if pv < alpha else 0)
            crvs.append(r["cramers_v"])
            p_specials.append(r["p_special"])
            exp.append(r["expected_p_special"])

            n = int(r["chi2"]["N"])
            total_n += n
            total_special += int(r["counts"].get(r["special_digit"], 0))

            total_correct += int(r.get("correct_pair", 0))
            total_valid_pair += int(r.get("valid_pair", 0))

            joint_valid.append(float(r.get("joint_accuracy_valid", 0.0)))
            joint_all.append(float(r.get("joint_accuracy_all", 0.0)))

            total_samples = max(int(r.get("samples_per_image", 1)), 1)
            invalid_digit_rates.append(int(r.get("invalid_digit", 0)) / total_samples)
            invalid_color_rates.append(int(r.get("invalid_color", 0)) / total_samples)
            invalid_rates.append(int(r.get("invalid", 0)) / total_samples)

        rej_rate = float(np.mean(reject)) if reject else float("nan")
        mean_crv = float(np.mean(crvs)) if crvs else float("nan")
        mean_p_special = float(np.mean(p_specials)) if p_specials else float("nan")
        mean_exp = float(np.mean(exp)) if exp else float("nan")

        mean_joint_valid = float(np.mean(joint_valid)) if joint_valid else float("nan")
        mean_joint_all = float(np.mean(joint_all)) if joint_all else float("nan")
        overall_joint_valid = total_correct / max(total_valid_pair, 1)

        mean_invalid_digit = float(np.mean(invalid_digit_rates)) if invalid_digit_rates else float("nan")
        mean_invalid_color = float(np.mean(invalid_color_rates)) if invalid_color_rates else float("nan")
        mean_invalid = float(np.mean(invalid_rates)) if invalid_rates else float("nan")

        pval_binom = None
        if binomtest is not None and total_n > 0 and mean_exp > 0:
            pval_binom = float(binomtest(k=total_special, n=total_n, p=mean_exp).pvalue)

        summaries.append(
            {
                "condition": condition,
                "protocol": protocol,
                "calibration": calibration,
                "restrict": restrict,
                "blank_image": blank,
                "n_images": len(rs),
                "alpha": alpha,
                "reject_rate_chi2": rej_rate,
                "mean_cramers_v": mean_crv,
                "mean_p_special": mean_p_special,
                "expected_p_special": mean_exp,
                "mean_joint_accuracy_valid": mean_joint_valid,
                "overall_joint_accuracy_valid": overall_joint_valid,
                "mean_joint_accuracy_all": mean_joint_all,
                "mean_invalid_digit_rate": mean_invalid_digit,
                "mean_invalid_color_rate": mean_invalid_color,
                "mean_invalid_rate": mean_invalid,
                "binom_pvalue_total": pval_binom,
            }
        )

    return pd.DataFrame(summaries).sort_values(
        ["protocol", "calibration", "blank_image", "restrict", "condition"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_jsonl", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--out_csv", type=str, default="poc_outputs/summary_choice_color.csv")
    args = parser.parse_args()

    rows = read_jsonl(args.results_jsonl)
    df = summarize(rows, alpha=args.alpha)
    print(df.to_string(index=False))

    from pathlib import Path

    Path(__import__("os").path.dirname(args.out_csv) or ".").mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] Saved summary: {args.out_csv}")


if __name__ == "__main__":
    main()
