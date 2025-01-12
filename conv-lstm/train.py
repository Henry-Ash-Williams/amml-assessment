import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from model import ImagePredictor
from tqdm import tqdm


def get_device():
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    return device


def get_args():
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=64,
        help="Batch size used while training",
    )
    arg_parser.add_argument(
        "-e", "--epochs", type=int, default=20, help="Number of training epochs"
    )
    arg_parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate used while training"
    )
    arg_parser.add_argument(
        "-l",
        "--layers",
        type=int,
        default=3,
        help="Number of layers in the ConvLSTM model",
    )
    arg_parser.add_argument(
        "-k", "--kernels", type=int, default=64, help="Number of kernels in each layer"
    )
    arg_parser.add_argument(
        "-a",
        "--activation",
        type=str,
        choices=["tanh", "relu"],
        default="relu",
        help="Activation function used by the model",
    )

    arg_parser.add_argument(
        "-o",
        "--optimizer",
        type=str,
        default="Adam",
        help="Optimisation algoritm used during training",
    )

    return arg_parser.parse_args()


def create_model(args, device):
    model = ImagePredictor(
        in_channels=1,
        num_kernels=args.kernels,
        num_layers=args.layers,
        kernel_size=(3, 3),
        padding=(1, 1),
        activation=args.activation,
        image_size=(36, 36),
        device=device,
    ).to(device)
    return model


def get_datasets():
    return torch.load(
        "/Users/henrywilliams/Documents/uni/amml/assessment/conv-lstm/train_loader.pt"
    ), torch.load(
        "/Users/henrywilliams/Documents/uni/amml/assessment/conv-lstm/test_loader.pt"
    )


def get_optimiser(name):
    return getattr(torch.optim, name)


def train(model, train, test, args):
    optim = get_optimiser(args.optimizer)(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    for epoch in range(args.epochs):
        model.train()
        torch.mps.empty_cache()
        for input, target in tqdm(
            train,
            desc=f"Train epoch {epoch + 1}/{args.epochs}",
            leave=False,
        ):
            optim.zero_grad()
            output = model(
                input, threshold=args.threshold, weight=args.weight
            ).unsqueeze(1)
            loss = criterion(output, target)
            loss.backward()
            optim.step()
            wandb.log({"train_loss": loss.item()})

        test_loss = 0
        model.eval()
        for input, target in tqdm(
            test,
            desc=f"Test epoch {epoch + 1}/{args.epochs}",
            leave=False,
        ):
            with torch.no_grad():
                output = model(
                    input, threshold=args.threshold, weight=args.weight
                ).unsqueeze(1)
            loss = criterion(output, target)
            test_loss += loss.item()
        test_loss /= len(test.dataset)
        wandb.log({"test_loss": test_loss})


def generate_image(model, test_loader, args):
    fig = plt.figure(tight_layout=True, dpi=300)

    test = np.random.choice(len(test_loader.dataset), 8)
    test_X, test_y = test_loader.dataset[test]

    test_y_hat = model(test_X.reshape(8, 1, 2, 36, 36))

    actual, pred = fig.subfigures(nrows=2, ncols=1)
    pred.suptitle("Predicted")
    actual.suptitle("Actual")
    axs = pred.subplots(1, 8)

    for ax, y_hat in zip(axs, test_y_hat):
        ax.imshow(y_hat.cpu().detach().reshape(36, 36))
        ax.set_xticks([])
        ax.set_yticks([])

    axs = actual.subplots(1, 8)
    for ax, y in zip(axs, test_y):
        ax.imshow(y.cpu().detach().reshape(36, 36))
        ax.set_xticks([])
        ax.set_yticks([])

    wandb.log({"test_images": fig})
    plt.close(fig)


def main(args, device):
    model = create_model(args, device)
    train_loader, test_loader = get_datasets(args, device)
    train(model, train_loader, test_loader, args)
    generate_image(model, test_loader, args)
    torch.save(model.state_dict(), "best-convlstm.pt")


if __name__ == "__main__":
    wandb.login()
    args = get_args()
    device = get_device()
    run = wandb.init(project="ConvLSTM", config=args.__dict__)
    try:
        main(args, device)
    except Exception as e:
        print(e)
        wandb.finish(1)
        exit(1)
    wandb.finish()
