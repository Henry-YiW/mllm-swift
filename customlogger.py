import os
import wandb
import pickle

wandb.login(key="5134a81007c8d0794c2e7e0000df2c62c9128c5c")
# Initialize once at the module level
wandb.init(project="swift", entity="henryyi-university-of-illinois-urbana-champaign", name="experiment_1")


def log_metrics(metrics: dict, filename: str, path=None, step=None):
    wandb.log(metrics, step=step)
    with open(filename if path is None else os.path.join(path, filename), 'wb') as f:
        pickle.dump(metrics, f)
    artifact = wandb.Artifact(filename, type="dataset")
    artifact.add_file(filename if path is None else os.path.join(path, filename))
    wandb.log_artifact(artifact)

def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)