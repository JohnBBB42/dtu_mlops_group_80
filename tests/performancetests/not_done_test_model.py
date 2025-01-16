import os
import time
import torch
import wandb
import pytorch_lightning as pl

# Example import of your model
from energy.model import NeuralNetwork


def load_checkpoint_safely(checkpoint_path: str):
    """
    Load a checkpoint file from disk and ensure it has a proper Lightning format:
      - if 'state_dict' is missing, nest raw weights under 'state_dict'
      - if 'pytorch-lightning_version' is missing, set it to the installed PL version
      - if 'hyper_parameters' is missing, set it to {}

    Returns a path to a (temporarily) fixed checkpoint, so Lightning can load it.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Step 1: If 'state_dict' is missing, assume we're dealing with raw weights
    if "state_dict" not in checkpoint:
        new_checkpoint = {
            # Provide an empty dict if hyperparameters are missing
            "hyper_parameters": checkpoint.get("hyper_parameters", {}),
            # If there's no PL version, add the current installed version
            "pytorch-lightning_version": checkpoint.get(
                "pytorch-lightning_version", pl.__version__
            ),
            "state_dict": {}
        }

        # Move model weights into "state_dict"
        for key, value in checkpoint.items():
            if key not in ("hyper_parameters", "pytorch-lightning_version"):
                new_checkpoint["state_dict"][key] = value

        checkpoint = new_checkpoint

    # Step 2: If 'pytorch-lightning_version' still missing, set it
    if "pytorch-lightning_version" not in checkpoint:
        checkpoint["pytorch-lightning_version"] = pl.__version__

    # Step 3: If 'hyper_parameters' missing, set it
    if "hyper_parameters" not in checkpoint:
        checkpoint["hyper_parameters"] = {}

    # (Optional) Re-save to a new path for ease
    fixed_checkpoint_path = checkpoint_path + ".temp.ckpt"
    torch.save(checkpoint, fixed_checkpoint_path)

    return fixed_checkpoint_path


def load_model(artifact_name: str):
    """
    Load a model from a wandb artifact given the artifact name.
    Example artifact_name: "entity/project/my_model:version"
    """
    logdir = "artifacts_download"

    # Initialize wandb API with env variables
    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={
            "entity": os.getenv("WANDB_ENTITY"),
            "project": os.getenv("WANDB_PROJECT"),
        },
    )

    # Retrieve the artifact
    artifact = api.artifact(artifact_name)
    artifact_dir = artifact.download(root=logdir)

    # Assume there's only one checkpoint file in the artifact
    file_name = artifact.files()[0].name
    checkpoint_path = os.path.join(artifact_dir, file_name)

    # Patch the checkpoint if needed and get a "fixed" path
    fixed_path = load_checkpoint_safely(checkpoint_path)

    # Now safely load the model
    model = NeuralNetwork.load_from_checkpoint(fixed_path)
    return model


def test_model_speed():
    """
    Test that the model can do 100 predictions in under 1 second.
    MODEL_NAME should be something like 'entity/project/my_model:version'
    """
    artifact_name = os.getenv("MODEL_NAME", "")
    if not artifact_name:
        raise ValueError("MODEL_NAME env variable not set or empty.")

    model = load_model(artifact_name)
    model.eval()

    start_time = time.time()
    for _ in range(100):
        inputs = torch.rand(1, 1, 28, 28)
        _ = model(inputs)
    end_time = time.time()

    total_time = end_time - start_time
    print(f"Total time for 100 predictions: {total_time} seconds")
    assert total_time < 1, "Model took too long to process 100 predictions!"

