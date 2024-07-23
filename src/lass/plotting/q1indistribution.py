import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Optional

import lass.plotting.shared as shared


@dataclass
class Args:
    path: Optional[Path]
    version: Optional[Literal["v1"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path)
    parser.add_argument("--version", type=str, choices=["v1"], default="v1")
    args_raw = parser.parse_args()
    run(Args(**vars(args_raw)))


def run(args: Args):
    base = Path("./artifacts/csv-results-new/")
    default = base / "q1indistribution" / "deberta-base_bs32_3sh_instance-split-07161413"  # fmt: skip
    path = args.path or default
    assert path.exists(), f"Path does not exist: {path}"

    results = shared.load_metrics(path)
    results.set_index("task", inplace=True)
    results = results[results.instance_count >= 100]

    plot_path = base / ".." / "plots" / "q1indistribution"
    if args.version == "v1":
        auroc, auroc_data = shared.plot_difference(
            results["test_conf_normalized_roc_auc"],
            results["test_roc_auc"],
            sys1_threshold=0.1,
            sort="diff",
            xlabel="Difference in AUROC",
        )
    else:
        raise ValueError(f"Invalid version: {args.version}")
    shared.save_to(auroc, auroc_data, plot_path / f"q1indistribution_auroc_{args.version}.pdf")  # fmt: skip

    if args.version == "v1":
        mcb, mcb_data = shared.plot_absolute(
            results["test_bs_mcb"],
            results["test_conf_normalized_bs_mcb"],
            sort="results",
            sort_direction="desc",
            xlabel="Assessor Brier Miscalibration (↓)",
        )
    else:
        raise ValueError(f"Invalid version: {args.version}")
    shared.save_to(mcb, mcb_data, plot_path / f"q1indistribution_mcb_{args.version}.pdf")  # fmt: skip


if __name__ == "__main__":
    main()
