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
            r.get("protocol"),
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
        invalid_rates = []

        total_special = 0
        total_n = 0

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

            invalid = int(r.get("invalid", 0))
            invalid_rates.append(invalid / max(r.get("samples_per_image", 1), 1))

        rej_rate = float(np.mean(reject)) if reject else float("nan")
        mean_crv = float(np.mean(crvs)) if crvs else float("nan")
        mean_p_special = float(np.mean(p_specials)) if p_specials else float("nan")
        mean_exp = float(np.mean(exp)) if exp else float("nan")
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
    parser.add_argument("--out_csv", type=str, default="poc_outputs/summary_choice_only.csv")
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
