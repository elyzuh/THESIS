import os
import re
import math

LOG_DIR = "./logs"

# Improved regex patterns for FINAL EVALUATION metrics
# Handles:
# - "Test RSE" for RMSE
# - "Test RAE" for MAE
# - "Test CORR" for CORR
# - "Test R²", "Test R2", or corrupted "Test R�" (common encoding issue for superscript ²)
PATTERNS = {
    "RMSE": re.compile(r"Test RSE\s*:\s*([0-9\.eE+-]+)"),
    "MAE":  re.compile(r"Test RAE\s*:\s*([0-9\.eE+-]+)"),
    "CORR": re.compile(r"Test CORR\s*:\s*([0-9\.eE+-]+)"),
    "R2":   re.compile(
        r"Test R[²2�]\s*:\s*([0-9\.eE+-]+)"  # Matches R², R2, or R� 
    )
}

def parse_final_evaluation(text):
    """
    Extract metrics from FINAL EVALUATION section.
    Returns dict or None if section not found.
    """
    if "FINAL EVALUATION" not in text:
        return None

    metrics = {}
    for key, pattern in PATTERNS.items():
        match = pattern.search(text)
        if match:
            value = float(match.group(1))
            metrics[key] = value

    return metrics if metrics else None


def main():
    best = {
        "RMSE": {"value": math.inf, "file": None},
        "MAE":  {"value": math.inf, "file": None},
        "CORR": {"value": -math.inf, "file": None},
        "R2":   {"value": -math.inf, "file": None},
    }

    all_results = []

    for fname in os.listdir(LOG_DIR):
        if not fname.endswith(".out"):
            continue

        path = os.path.join(LOG_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:  # 'replace' instead of 'ignore' to keep �
                text = f.read()
        except UnicodeDecodeError:
            # Fallback if utf-8 fails
            with open(path, "r", encoding="latin-1") as f:
                text = f.read()

        metrics = parse_final_evaluation(text)
        if metrics is None:
            continue

        all_results.append((fname, metrics))

        # Track best per metric (lower better for RMSE/MAE, higher for CORR/R2)
        if "RMSE" in metrics and metrics["RMSE"] < best["RMSE"]["value"]:
            best["RMSE"] = {"value": metrics["RMSE"], "file": fname}

        if "MAE" in metrics and metrics["MAE"] < best["MAE"]["value"]:
            best["MAE"] = {"value": metrics["MAE"], "file": fname}

        if "CORR" in metrics and metrics["CORR"] > best["CORR"]["value"]:
            best["CORR"] = {"value": metrics["CORR"], "file": fname}

        if "R2" in metrics and metrics["R2"] > best["R2"]["value"]:
            best["R2"] = {"value": metrics["R2"], "file": fname}

    # -------------------------------------------------
    # Output summary
    # -------------------------------------------------
    print("============== FINAL EVALUATION SUMMARY ==============\n")

    for metric in ["RMSE", "MAE", "CORR", "R2"]:
        entry = best[metric]
        if entry["file"] is not None:
            print(
                f"Best {metric:4s}: {entry['value']:.4f} | "
                f"file: {entry['file']}"
            )
        else:
            print(f"Best {metric:4s}: NOT FOUND")

    print("\n======================================================")


if __name__ == "__main__":
    main()