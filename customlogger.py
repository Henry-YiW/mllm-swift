import os
import wandb
import pickle

# Completely disable wandb before importing it
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_API_KEY"] = ""
#wandb.login(key="5134a81007c8d0794c2e7e0000df2c62c9128c5c")
# Initialize once at the module level
#how to disable wandb
try:
    wandb.init(mode="disabled")
except:
    pass
#wandb.init(project="swift", entity="henryyi-university-of-illinois-urbana-champaign", name="experiment_1")

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