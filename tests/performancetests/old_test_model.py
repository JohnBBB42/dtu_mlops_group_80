import os
import time
import torch
import wandb

# Example import of your model
from mnist_project.lightning import MyAwesomeModel  # or your actual model location


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

    # Load the model checkpoint
    model = MyAwesomeModel.load_from_checkpoint(checkpoint_path)
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

