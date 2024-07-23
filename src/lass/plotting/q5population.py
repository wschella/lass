import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Optional

import lass.plotting.shared as shared


@dataclass
class Args:
    path_pop: Optional[Path]
    path_ref: Optional[Path]
    version: Optional[Literal["v1"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-pop", type=Path, dest="path_pop")
    parser.add_argument("--path-ref", type=Path, dest="path_ref")
    parser.add_argument("--version", type=str, choices=["v1"], default="v1")
    args_raw = parser.parse_args()
    run(Args(**vars(args_raw)))


def run(args: Args):
    base = Path("./artifacts/csv-results-new/")
    default_ref = base / "q1indistribution" / "deberta-base_bs32_3sh_instance-split-07161413"  # fmt: skip
    default_pop = base / "q5population" / "deberta-base_bs32_3sh_pop_instance-split-07191503"  # fmt: skip
    path_ref = args.path_ref or default_ref
    path_pop = args.path_pop or default_pop
    assert path_ref.exists(), f"Path does not exist: {path_ref}"
    assert path_pop.exists(), f"Path does not exist: {path_pop}"

    results_ref = shared.load_metrics(path_ref, save=False)
    results_pop = shared.load_metrics(path_pop)
    results_ref = results_ref[results_ref.instance_count >= 100]
    results_pop = results_pop[results_pop.instance_count >= 100]
    results_ref.set_index("task", inplace=True)
    results_pop.set_index("task", inplace=True)

    plot_path = base / ".." / "plots" / "q5population"
    if args.version == "v1":
        auroc, auroc_data = shared.plot_difference(
            results_ref["test_roc_auc"],
            results_pop["test_roc_auc"],
            sort="diff",
            xlabel="Difference in AUROC",
        )
    else:
        raise ValueError(f"Invalid version: {args.version}")
    shared.save_to(auroc, auroc_data, plot_path / f"q5population_auroc_{args.version}.pdf")  # fmt: skip

    if args.version == "v1":
        mcb, mcb_data = shared.plot_difference(
            results_ref["test_bs_mcb"],
            results_pop["test_bs_mcb"],
            sort="diff",
            sort_direction="desc",
            xlabel="Assessor Brier Miscalibration (↓)",
        )
    else:
        raise ValueError(f"Invalid version: {args.version}")
    shared.save_to(mcb, mcb_data, plot_path / f"q5population_mcb_{args.version}.pdf")  # fmt: skip


if __name__ == "__main__":
    main()
