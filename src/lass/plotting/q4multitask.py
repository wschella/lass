import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Optional

import lass.plotting.shared as shared


@dataclass
class Args:
    path_singltask: Optional[Path]
    path_multitask: Optional[Path]
    shots: Literal[0, 3]
    version: Optional[Literal["v1"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-single", type=Path, dest="path_singltask")
    parser.add_argument("--path-multi", type=Path, dest="path_multitask")
    parser.add_argument("--version", type=str, choices=["v1"], default="v1")
    parser.add_argument("--shots", type=int, choices=[0, 3], default=3)
    args_raw = parser.parse_args()
    run(Args(**vars(args_raw)))


def run(args: Args):
    base = Path("./artifacts/csv-results-new/")
    if args.shots == 3:
        default_multi = base / "q1indistribution" / "deberta-base_bs32_3sh_instance-split-07161413"  # fmt: skip
        default_singl = base / "q4multitask" / "deberta-base_bs32_3sh_instance-split-07231725"  # fmt: skip
        # default_singl = base / "q4multitask" / "worse-models" / "deberta-base_bs32_3sh_instance-split-07231725"  # fmt: skip
    if args.shots == 0:
        default_multi = base / "q1indistribution" / "deberta-base_bs32_0sh_instance-split-07161413"  # fmt: skip
        default_singl = base / "q4multitask" / "deberta-base_bs32_0sh_instance-split-07231725"  # fmt: skip

    path_multi = args.path_multitask or default_multi
    path_singl = args.path_singltask or default_singl
    assert path_multi.exists(), f"Path does not exist: {path_multi}"
    assert path_singl.exists(), f"Path does not exist: {path_singl}"

    results_multi = shared.load_metrics(path_multi, save=False, load=["task"])
    results_singl = shared.load_metrics(path_singl, load=["task"])
    results_multi = results_multi[results_multi.instance_count >= 100]
    results_singl = results_singl[results_singl.instance_count >= 100]
    results_multi.set_index("task", inplace=True)
    results_singl.set_index("task", inplace=True)
    print(f"There are {len(results_multi)} tasks in the multitask dataset, and {len(results_singl)} tasks in the single task dataset.")  # fmt: skip

    plot_path = base / ".." / "plots" / "q4multitask"
    if args.version == "v1":
        auroc, auroc_data = shared.plot_difference(
            results_multi["test_roc_auc"],
            results_singl["test_roc_auc"],
            sort="diff",
            xlabel="Difference in AUROC",
        )
    else:
        raise ValueError(f"Invalid version: {args.version}")
    shared.save_to(auroc, auroc_data, plot_path / f"q4multitask_auroc_{args.version}_{args.shots}sh.pdf")  # fmt: skip

    if args.version == "v1":
        mcb, mcb_data = shared.plot_absolute(
            results_multi["test_bs_mcb"],
            results_singl["test_bs_mcb"],
            sort="results",
            sort_direction="desc",
            xlabel="Assessor Brier Miscalibration (↓)",
        )
    else:
        raise ValueError(f"Invalid version: {args.version}")
    shared.save_to(mcb, mcb_data, plot_path / f"q4multitask_mcb_{args.version}_{args.shots}sh.pdf")  # fmt: skip


if __name__ == "__main__":
    main()
