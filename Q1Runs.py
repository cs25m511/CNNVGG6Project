from DataManager import ArgConfig
from Control import Control


ACTIVATIONS = ["relu", "sigmoid", "tanh", "silu", "gelu"]


# Fixed hyperparameters for a controlled comparison.
BASE_HYPERPARAMETERS = {
    "optimizer": "sgd",
    "learning_rate": 0.001,
    "batch_size": 128,
    "epochs": 40,
    "weight_decay": 0.0001,
    "use_batchnorm": False,
    "momentum": 0.9,
    "seed": 19,
    "wandb_project": "Normalization_Q1",
}


# Five preprocessing test cases with parameters embedded in the test_id.
PREPROCESSING_CONFIGURATIONS = [
    {
        "test_id": "TC1_calc=F_jitter=F_erase=F",
        "calc_meanstd": False,
        "normalize_mean": (0.5, 0.5, 0.5),
        "normalize_std": (0.5, 0.5, 0.5),
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
    },
    {
        "test_id": "TC2_calc=T_jitter=F_erase=F",
        "calc_meanstd": True,
        "normalize_mean": (0.5, 0.5, 0.5),
        "normalize_std": (0.5, 0.5, 0.5),
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
    },
    {
        "test_id": "TC3_calc=T_jitter=T_erase=F",
        "calc_meanstd": True,
        "normalize_mean": (0.5, 0.5, 0.5),
        "normalize_std": (0.5, 0.5, 0.5),
        "add_crop": True,
        "crop_padding": 4,
        "horizontal_flip_prob": 0.5,
        "add_jitter": True,
        "jitter_brightness": 0.15,
        "jitter_contrast": 0.15,
        "jitter_saturation": 0.15,
        "jitter_hue": 0.02,
        "add_random_erasing": False,
        "random_erasing_prob": 0.25,
        "random_erasing_scale": (0.02, 0.10),
        "random_erasing_ratio": (0.5, 2.0),
    },
    {
        "test_id": "TC4_calc=T_jitter=F_erase=T",
        "calc_meanstd": True,
        "normalize_mean": (0.5, 0.5, 0.5),
        "normalize_std": (0.5, 0.5, 0.5),
        "add_crop": True,
        "crop_padding": 4,
        "horizontal_flip_prob": 0.5,
        "add_jitter": False,
        "jitter_brightness": 0.15,
        "jitter_contrast": 0.15,
        "jitter_saturation": 0.15,
        "jitter_hue": 0.02,
        "add_random_erasing": True,
        "random_erasing_prob": 0.25,
        "random_erasing_scale": (0.02, 0.10),
        "random_erasing_ratio": (0.5, 2.0),
    },
    {
        "test_id": "TC5_calc=T_jitter=T_erase=T",
        "calc_meanstd": True,
        "normalize_mean": (0.5, 0.5, 0.5),
        "normalize_std": (0.5, 0.5, 0.5),
        "add_crop": True,
        "crop_padding": 4,
        "horizontal_flip_prob": 0.5,
        "add_jitter": True,
        "jitter_brightness": 0.15,
        "jitter_contrast": 0.15,
        "jitter_saturation": 0.15,
        "jitter_hue": 0.02,
        "add_random_erasing": True,
        "random_erasing_prob": 0.25,
        "random_erasing_scale": (0.02, 0.10),
        "random_erasing_ratio": (0.5, 2.0),
    },
]


def update_config(config, settings):
    """
    Update an ArgConfig object without depending on a set_all() method.
    """
    for key, value in settings.items():
        if not hasattr(config, key):
            raise AttributeError(
                f"ArgConfig does not define '{key}'. "
                "Check that ArgConfig.py has been updated."
            )
        setattr(config, key, value)


def main():
    # Parse command-line arguments once.
    config = ArgConfig.ArgConfig.from_cli()

    # Create the controller.
    control = Control.Control()

    total_runs = len(PREPROCESSING_CONFIGURATIONS) * len(ACTIVATIONS)

    run_number = 0

    print("=" * 75)
    print("QUESTION 1: CIFAR-10 PREPROCESSING EXPERIMENTS")
    print(f"Preprocessing configurations: {len(PREPROCESSING_CONFIGURATIONS)}")
    print(f"Activation functions: {len(ACTIVATIONS)}")
    print(f"Total planned runs: {total_runs}")
    print("=" * 75)

    for preprocessing in PREPROCESSING_CONFIGURATIONS:
        test_id = preprocessing["test_id"]

        print("\n" + "#" * 75)
        print(f"PREPROCESSING TEST CASE: {test_id}")
        print("#" * 75)

        for activation in ACTIVATIONS:
            run_number += 1

            # Combine fixed training parameters, preprocessing settings,
            # and the current activation function.
            settings = {
                **BASE_HYPERPARAMETERS,
                **preprocessing,
                "activation": activation,
                "run_name": f"{test_id}_act={activation}",
            }

            # Remove the experiment label before assigning configuration.
            settings.pop("test_id")

            update_config(config, settings)

            print(
                f"\nRun {run_number}/{total_runs}: "
                f"{test_id} | activation={activation}"
            )

            # Control.start() should create a fresh model for each run.
            control.setconfig(config)
            control.start()

    print("\n" + "=" * 75)
    print(f"ALL {total_runs} EXPERIMENT RUNS FINISHED")
    print("=" * 75)


if __name__ == "__main__":
    main()