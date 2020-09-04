import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm


class Trainer:
    def __init__(self, args, model, model_name):
        self.model = model
        self.model_name = model_name
        self.optimizer = Adam(model.parameters(), lr=args["lr"])
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=args["gamma"])

    def train(self, args, train_loader, test_loader=None):
        train_size = len(train_loader.dataset)
        train_loss = 0
        self.model.train()

        for epoch in tqdm(range(args["num_epochs"]), desc="Epoch"):
            for data, target in tqdm(train_loader, desc="Iteration"):
                # Tensor shape: batch size x channels x height x width
                self.optimizer.zero_grad()  # Set gradients of model parameters to zero

                output = self.model(data)  # Forward pass
                loss = F.nll_loss(output, target)  # Calculate loss
                loss.backward()  # Backward pass
                train_loss += loss.item()

                self.optimizer.step()  # Update model parameters

            print(f"\nLearning rate = {self.scheduler.get_last_lr()[0]}")
            print(f"Train set:\tloss = {train_loss/train_size}")

            if test_loader:
                self.test(test_loader)  # Evaluate on test data

            self.scheduler.step()  # Update learning rate schedule

            if args["save_checkpoints"]:
                self.model.save(f"data/model_files/{self.model_name}_epoch{epoch}.pt")

        if args["save_model"]:
            self.model.save(f"data/model_files/{self.model_name}_completed.pt")

    def test(self, test_loader):
        test_size = len(test_loader.dataset)
        test_loss, correct = 0, 0

        self.model.eval()

        with torch.no_grad():
            for data, target in test_loader:
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= test_size
        accuracy = correct / test_size

        print(f"Test set:\tloss = {test_loss}, accuracy = {accuracy}")
