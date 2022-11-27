import run
import datasets
import models
import torch

if __name__ == '__main__':

    hyperparameters = {
        "epochs": 10,
        "lr": 5e-4,
        "loss_func": torch.nn.CrossEntropyLoss()
    }

    data = {
        "train_set": datasets.gtzan_dataset("../data/gtzan", "train"),
        "test_set": datasets.gtzan_dataset("../data/gtzan", "test"),
        "val_set": datasets.gtzan_dataset("../data/gtzan", "val"),
    }

    model = models.CustomMLP()
    trainer = run.Trainer(model=model, hyperparameters=hyperparameters, data=data)
    trainer.train()
