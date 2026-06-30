import argparse
import csv
import json
import os


def load_rows(path):
    with open(path) as f:
        payload = json.load(f)

    rows = []
    epoch = payload.get("epoch")
    global_step = payload.get("global_step")
    for split, split_payload in payload.get("splits", {}).items():
        for trace_name in ["intermediate_predictions", "oracle_noise_predictions"]:
            trace = split_payload.get(trace_name)
            if trace is None:
                continue
            for row in trace.get("rows", []):
                rows.append(
                    {
                        "source_file": os.path.basename(path),
                        "epoch": epoch,
                        "global_step": global_step,
                        "split": split,
                        "trace_type": trace_name,
                        **row,
                    }
                )
    return rows


def write_csv(rows, path):
    if not rows:
        raise ValueError("No blockwise rows found")
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows, path):
    import matplotlib.pyplot as plt

    metric_key = "accuracy_rate" if "accuracy_rate" in rows[0] else "rmse"
    ylabel = "Accuracy (%)" if metric_key == "accuracy_rate" else "RMSE"

    grouped = {}
    for row in rows:
        key = (row["split"], row["trace_type"])
        grouped.setdefault(key, []).append(row)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (split, trace_type), group_rows in sorted(grouped.items()):
        group_rows = sorted(group_rows, key=lambda row: row["step_index"])
        x = [row["block_index"] for row in group_rows]
        if metric_key == "accuracy_rate":
            y = [100.0 * row[metric_key] for row in group_rows]
        else:
            y = [row[metric_key] for row in group_rows]
        label = f"{split}: {trace_type.replace('_predictions', '')}"
        ax.plot(x, y, marker="o", linewidth=1.8, label=label)

    ax.set_xlabel("Block index")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Blockwise prediction {ylabel}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")
    parser.add_argument("--csv_path", default=None)
    parser.add_argument("--plot_path", default=None)
    args = parser.parse_args()

    rows = load_rows(args.json_path)
    stem, _ = os.path.splitext(args.json_path)
    csv_path = args.csv_path or f"{stem}_blockwise_rows.csv"
    plot_path = args.plot_path or f"{stem}_blockwise_accuracy.png"
    write_csv(rows, csv_path)
    write_plot(rows, plot_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
