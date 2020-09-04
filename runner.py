import torch
from torchvision import datasets, transforms

from model import CNNForDigitRecognition
from trainer import Trainer


if __name__ == "__main__":

    args = {
        "num_epochs": 3,
        "batch_size": 4,
        "lr": 0.001,
        "gamma": 0.2,
        "save_checkpoints": False,
        "save_model": True,
    }

    # Mean and standard deviation of MNIST train data in Normalize
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train, batch_size=args["batch_size"], shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=args["batch_size"], shuffle=False, num_workers=2
    )

    model = CNNForDigitRecognition()
    print(model)

    trainer = Trainer(args, model, "model1")
    trainer.train(args, train_loader, test_loader)
    print("Training done!")
