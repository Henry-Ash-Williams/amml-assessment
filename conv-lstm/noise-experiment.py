import os
import warnings

import torch
import torch.nn as nn
import wandb
from matplotlib import pyplot as plt
from model import ImagePredictor
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)


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

    arg_parser.add_argument(
        "--exclude-outliers", choices=["True", "False"], type=str, default="False"
    )

    arg_parser.add_argument(
        "--denoise", choices=["True", "False"], type=str, default="False"
    )

    arg_parser.add_argument(
        "--dataset-path",
        type=str,
        default="/Users/henrywilliams/Documents/uni/amml/assessment/conv-lstm/data",
    )

    return arg_parser.parse_args()


def get_datasets(args):
    # Shut up idc
    denoise = args.denoise.lower() == "true"
    exclude_outliers = args.exclude_outliers.lower() == "true"

    n6_train_path = os.path.join(
        args.dataset_path,
        f"train-n6{'-denoised' if denoise else ''}{'-no-outliers' if exclude_outliers else ''}.pt",
    )

    n6_test_path = os.path.join(
        args.dataset_path,
        f"test-n6{'-denoised' if denoise else ''}{'-no-outliers' if exclude_outliers else ''}.pt",
    )

    n6_train = torch.load(n6_train_path)
    n6_test = torch.load(n6_test_path)

    n3_train_path = os.path.join(
        args.dataset_path,
        f"train-n3{'-denoised' if args.denoise else ''}{'-no-outliers' if args.exclude_outliers else ''}.pt",
    )

    n3_test_path = os.path.join(
        args.dataset_path,
        f"test-n3{'-denoised' if args.denoise else ''}{'-no-outliers' if args.exclude_outliers else ''}.pt",
    )

    n3_train = torch.load(n3_train_path)
    n3_test = torch.load(n3_test_path)

    n3 = ConcatDataset([n3_train, n3_test])

    n6_train_loader = DataLoader(n6_train, batch_size=args.batch_size, shuffle=True)
    n6_test_loader = DataLoader(n6_test, batch_size=args.batch_size, shuffle=True)

    n3_loader = DataLoader(n3, batch_size=args.batch_size, shuffle=True)

    return n6_train_loader, n6_test_loader, n3_loader


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


def create_config(args):
    config = {
        "batch-size": args.batch_size,
        "layers": args.layers,
        "kernels": args.kernels,
        "activation": args.activation,
        "epochs": args.epochs,
        "optimizer": args.optimizer,
        "lr": args.lr,
    }
    return config


def create_tags(args):
    tags = []

    denoise = args.denoise.lower() == "true"
    exclude_outliers = args.exclude_outliers.lower() == "true"

    if exclude_outliers:
        tags.append("no-outliers")
    if denoise:
        tags.append("no-noise")

    return tags


def test(model, test_loader, loss_fn, device):
    model.eval()
    test_loss = 0.0

    loop = tqdm(test_loader, desc="Testing", unit="batch", leave=False)
    i = 0
    for input, target in loop:
        input = input.to(device)
        target = target.to(device)

        with torch.no_grad():
            output = model(input).unsqueeze(1)

        loss = loss_fn(output, target)
        test_loss += loss.item()
        i += 1
        loop.set_postfix_str(f"Loss: {test_loss / i:.2}")
    model.train()

    return test_loss / len(test_loader.dataset)


def get_optimiser(name):
    return getattr(torch.optim, name)


def train(model, train_loader, test_loader, n3_loader, args, device):
    optim = get_optimiser(args.optimizer)(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    for epoch in range(args.epochs):
        model.train()
        torch.mps.empty_cache()
        for input, target in tqdm(
            train_loader,
            desc=f"Train epoch {epoch + 1}/{args.epochs}",
            leave=False,
        ):
            input = input.to(device)
            target = target.to(device)
            optim.zero_grad()
            output = model(input).unsqueeze(1)
            loss = loss_fn(output, target)
            loss.backward()
            optim.step()
            wandb.log({"train_loss": loss.item()})

        n6_test_loss = test(model, test_loader, loss_fn, device)
        n3_test_loss = test(model, n3_loader, loss_fn, device)
        wandb.log({"test_loss": {"n6": n6_test_loss, "n3": n3_test_loss}})


def generate_image(model, dataset, idx, device, name):
    fig = plt.figure(dpi=100)

    x, y = dataset.dataset[idx]
    y_hat = model(x.unsqueeze(0).to(device))

    input, output = fig.subfigures(nrows=1, ncols=2)
    input.suptitle("Input")

    axs = input.subplots(1, 2)
    axs[0].imshow(x[:, 0].cpu().detach().reshape(36, 36))
    [ax.axis("off") for ax in axs]
    axs[1].imshow(x[:, 1].cpu().detach().reshape(36, 36))
    axs = output.subplots(2, 1)
    axs[0].imshow(y.cpu().detach().reshape(36, 36))
    axs[0].title.set_text("Expected")
    axs[1].imshow(y_hat.cpu().detach().reshape(36, 36))
    axs[1].title.set_text("Predicted")

    [ax.axis("off") for ax in axs]
    wandb.log({"images": {name: fig}})
    plt.close(fig)


if __name__ == "__main__":
    args = get_args()
    device = get_device()
    run = wandb.init(
        project="ConvLSTM-Experiment",
        tags=create_tags(args),
        config=create_config(args),
    )
    model = create_model(args, device)
    n6_train, n6_test, n3 = get_datasets(args)
    train(model, n6_train, n6_test, n3, args, device)
    generate_image(model, n6_test, 100, device, "n6")
    generate_image(model, n3, 100, device, "n3")
    wandb.finish()
