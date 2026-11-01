
import argparse
import traceback
from pathlib import Path

from DataManager import ArgConfig
from Control import Control


ACTIVATIONS = ["relu", "sigmoid", "tanh", "silu", "gelu"]
OPTIMIZERS = ["sgd", "nesterov_sgd", "adam", "adagrad", "rmsprop", "nadam"]
GROUPS = ["activation", "optimizer", "batch_size", "epochs", "learning_rate"]

# Common settings held constant unless explicitly overridden below.
RUN_13_BASE = {
    "activation": "tanh",
    "optimizer": "sgd",
    "batch_size": 128,
    "epochs": 40,
    "learning_rate": 0.001,
    "momentum": 0.9,
    "weight_decay": 0.0001,
    "use_batchnorm": False,
    "calc_meanstd": True,
    "normalize_mean": (0.5, 0.5, 0.5),  # ignored when calc_meanstd=True
    "normalize_std": (0.5, 0.5, 0.5),   # ignored when calc_meanstd=True
    "add_crop": True,
    "crop_padding": 4,
    "horizontal_flip_prob": 0.5,
    "add_jitter": False,
    "jitter_brightness": 0.15,
    "jitter_contrast": 0.15,
    "jitter_saturation": 0.15,
    "jitter_hue": 0.02,
    "add_random_erasing": False,
    "random_erasing_prob": 0.25,
    "random_erasing_scale": (0.02, 0.10),
    "random_erasing_ratio": (0.5, 2.0),
}

EXPERIMENT_RUNS = {
    "epochs": [
        ("epochs_relu_nadam_bn_100", {
            "activation": "relu", "optimizer": "nadam", "epochs": 100,
            "batch_size": 128, "learning_rate": 0.001,
            "weight_decay": 0.0001, "use_batchnorm": True
        }),
        ("epochs_relu_nadam_bn_120", {
            "activation": "relu", "optimizer": "nadam", "epochs": 120,
            "batch_size": 128, "learning_rate": 0.001,
            "weight_decay": 0.0001, "use_batchnorm": True
        }),
        ("epochs_relu_adam_bn_100", {
            "activation": "relu", "optimizer": "adam", "epochs": 100,
            "batch_size": 128, "learning_rate": 0.001,
            "weight_decay": 0.0001, "use_batchnorm": True
        }),
    ],
    "optimizer": [
        ("optimizer_relu_nadam_bn_wd5e5", {
            "activation": "relu", "optimizer": "nadam", "epochs": 80,
            "batch_size": 128, "learning_rate": 0.001,
            "weight_decay": 0.00005, "use_batchnorm": True
        }),
    ],
    "learning_rate": [
        ("learning_rate_relu_nadam_bn_5e4", {
            "activation": "relu", "optimizer": "nadam", "epochs": 80,
            "batch_size": 128, "learning_rate": 0.0005,
            "weight_decay": 0.0001, "use_batchnorm": True
        }),
    ],
}


def make_cases(selected_groups):
    cases = []

    if "activation" in selected_groups:
        for value in ACTIVATIONS:
            cases.append(("activation", f"activation_{value}", {"activation": value}))

    if "optimizer" in selected_groups:
        for value in OPTIMIZERS:
            cases.append(("optimizer", f"optimizer_{value}", {"optimizer": value}))

    if "batch_size" in selected_groups:
        for value in [64, 128, 256]:
            cases.append(("batch_size", f"batch_size_{value}", {"batch_size": value}))

    if "epochs" in selected_groups:
        for value in [20, 40, 60]:
            cases.append(("epochs", f"epochs_{value}", {"epochs": value}))

    if "learning_rate" in selected_groups:
        for value in [0.0001, 0.001, 0.01]:
            label = str(value).replace(".", "p")
            cases.append(("learning_rate", f"learning_rate_{label}", {"learning_rate": value}))

    # Include targeted configurations under their related existing groups.
    for group in ("epochs", "optimizer", "learning_rate"):
        if group in selected_groups:
            for case_id, overrides in EXPERIMENT_RUNS.get(group, []):
                cases.append((group, case_id, overrides))

    return cases


def build_config(values, args, run_output_dir, run_name):
    return ArgConfig.ArgConfig(
        activation=values["activation"],
        optimizer=values["optimizer"],
        batch_size=values["batch_size"],
        epochs=values["epochs"],
        learning_rate=values["learning_rate"],
        momentum=values["momentum"],
        weight_decay=values["weight_decay"],
        use_batchnorm=values["use_batchnorm"],
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        data_dir=args.data_dir,
        output_dir=str(run_output_dir),
        calc_meanstd=values["calc_meanstd"],
        normalize_mean=values["normalize_mean"],
        normalize_std=values["normalize_std"],
        add_crop=values["add_crop"],
        crop_padding=values["crop_padding"],
        horizontal_flip_prob=values["horizontal_flip_prob"],
        add_jitter=values["add_jitter"],
        jitter_brightness=values["jitter_brightness"],
        jitter_contrast=values["jitter_contrast"],
        jitter_saturation=values["jitter_saturation"],
        jitter_hue=values["jitter_hue"],
        add_random_erasing=values["add_random_erasing"],
        random_erasing_prob=values["random_erasing_prob"],
        random_erasing_scale=values["random_erasing_scale"],
        random_erasing_ratio=values["random_erasing_ratio"],
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        run_name=run_name,
        no_wandb=not args.wandb,
        no_download=args.no_download,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run CS6886W Q2 experiments; use W&B for plots and dashboards."
    )
    parser.add_argument("--groups", nargs="+", choices=["all"] + GROUPS, default=["all"])
    parser.add_argument("--output_dir", default="./q2_runs")
    parser.add_argument("--data_dir", default="./CIFAR10Dataset")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--wandb_project", default="Q2Solution")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb", action="store_true", help="Enable online W&B logging")
    parser.add_argument("--no_download", action="store_true", help="Do not download CIFAR-10")
    args = parser.parse_args()

    selected_groups = GROUPS if "all" in args.groups else args.groups
    cases = make_cases(selected_groups)
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("CS6886W QUESTION 2 — RUN-ONLY EXPERIMENT EXECUTION")
    print(f"Groups: {selected_groups}")
    print(f"Planned runs: {len(cases)}")
    print(f"Output directory: {output_root}")
    print(f"W&B enabled: {args.wandb}")
    print("=" * 78)

    successful = 0
    failed = 0

    for index, (group, case_id, override) in enumerate(cases, start=1):
        values = dict(RUN_13_BASE)
        values.update(override)
        case_dir = output_root / group / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        run_name = f"Q2_{case_id}_seed{args.seed}"

        print("\\n" + "=" * 78)
        print(f"[{index}/{len(cases)}] {case_id}")
        print("Configuration:")
        for key, value in values.items():
            print(f"  {key}: {value}")
        print(f"  seed: {args.seed}")
        print(f"  run_name: {run_name}")
        print("=" * 78)

        try:
            config = build_config(values, args, case_dir, run_name)
            config.validate()
            controller = Control.Control()
            controller.setconfig(config)
            controller.start()
            successful += 1
            print(f"Completed: {case_id}")
        except Exception as exc:
            failed += 1
            print(f"FAILED: {case_id} — {type(exc).__name__}: {exc}")
            traceback.print_exc()

    print("\\n" + "=" * 78)
    print(f"Run execution finished. Successful: {successful}; Failed: {failed}")
    print("Training/validation metrics and plots should be viewed in W&B.")
    print("=" * 78)


if __name__ == "__main__":
    main()