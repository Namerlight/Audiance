from typing import Tuple, Any
import torch
from torch.utils.data import Dataset, DataLoader
import datasets
from tqdm import trange, tqdm
import numpy as np
import utils


class Trainer:
    def __init__(self, model: torch.nn, hyperparameters: dict, data: dict):
        self.model = model
        self.batch = hyperparameters.get("batch_size", 16)
        self.epochs = hyperparameters.get("epochs", 10)
        self.lr = hyperparameters.get("lr", 1e-3)
        self.criterion = hyperparameters.get("loss_func", torch.nn.CrossEntropyLoss())
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        self.train_loader = torch.utils.data.DataLoader(data.get("train_set"), batch_size=self.batch, shuffle=True, num_workers=0)
        self.test_loader = torch.utils.data.DataLoader(data.get("test_set"), batch_size=self.batch, shuffle=True, num_workers=0)
        self.val_loader = torch.utils.data.DataLoader(data.get("val_set"), batch_size=self.batch, shuffle=True, num_workers=0)

        self.train_loss, self.val_loss = [], []
        self.train_acc, self.val_acc = [], []

    def train(self):
        """
        Training loop

        Returns:

        """

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Computing on {device}")
        total_step = len(self.train_loader)

        self.model.to(device)

        for epoch in tqdm(range(self.epochs)):

            self.model.train()
            running_loss, correct, total = 0.0, 0, 0

            for batch_n, (data, target) in enumerate(self.train_loader):
                data, target = data.to(device), target.to(device)
                self.optimizer.zero_grad()

                outputs = self.model(data)

                loss = self.criterion(outputs, target)
                loss.backward()
                self.optimizer.step()

                # if (batch_n) % 10 == 0:
                #     print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                #           .format(epoch, self.epochs, batch_n, total_step, loss.item()))

                running_loss += loss
                _, predicted = torch.max(outputs, dim=1)
                total += target.size(0)
                correct += torch.sum(predicted == target).item()

            self.train_loss.append((running_loss / total_step).detach().cpu())
            self.train_acc.append((100 * correct / total))

            print(f'\ntrain-loss: {np.mean(self.train_loss):.4f},')
            print(f'train-acc: {np.mean(self.train_acc):.4f}')

        utils.plot_loss_accuracy(self.train_loss, self.train_acc, self.train_loss, self.train_acc)
