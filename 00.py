import ray
import cv2
import os
import torch, torchvision
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.nn import *
from torch.utils.tensorboard import SummaryWriter

IMG_SIZE = 84
transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])


def load_data(directory, IMG_SIZE=IMG_SIZE):
    idx = -1
    data = []
    for file in tqdm(os.listdir(directory)):
        file = directory + file
        img = cv2.imread(file, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(np.array(transforms(np.array(img))))
    return data


X = load_data(directory="./data/")
np.save("./data.npy", np.array(X))
X = np.load("./data.npy")
# X = X[:500]
import matplotlib.pyplot as plt


class Desc(nn.Module):
    def __init__(self, activation=nn.LeakyReLU, starter=16):
        super().__init__()
        self.dis = nn.Sequential(
            nn.Conv2d(3, 4, 3), activation(), nn.Conv2d(4, 8, 3), activation(),
        )
        self.dis2 = nn.Sequential(
            nn.Linear(93312, starter),
            activation(),
            nn.Linear(starter, starter * 2),
            activation(),
            nn.Linear(starter * 2, starter),
            activation(),
            nn.Linear(starter, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, shape=True):
        x = x.view(-1, 3, IMG_SIZE, IMG_SIZE)
        x = self.dis(x)
        if shape:
            print("*" * 50)
            print(x.shape)
            print("*" * 50)

        x = x.view(-1, 93312)
        x = self.dis2(x)
        return x


class Gen(nn.Module):
    def __init__(self, z_dim, activation=nn.LeakyReLU, starter=256):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, starter),
            activation(),
            nn.Linear(starter, starter * 2),
            activation(),
            nn.Linear(starter * 2, starter * 4),
            activation(),
            nn.Linear(starter * 4, starter * 2),
            activation(),
            nn.Linear(starter * 2, IMG_SIZE * IMG_SIZE * 3),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)


from ray import tune

device = "cuda"
import wandb

PROJECT_NAME = "Abstract-Paiting-V2"
image_dim = IMG_SIZE * IMG_SIZE * 3
writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
step = 0


def return_loss(lossG, lossD):
    loss = lossG + lossD
    return loss


def train(config, tune_using=True):
    torch.cuda.empty_cache()
    epochs = config["epochs"]
    lr = config["lr"]
    z_dim = config["z_dim"]
    batch_size = config['batch_size']
    criterion = nn.BCELoss()
    gen = Gen(z_dim,starter=config['gen_starter']).to(device)
    desc = Desc().to(device)
    fixed_noise = torch.randn((batch_size, z_dim)).to(device)
    opt_disc = config["opt_disc"](desc.parameters(), lr=lr)
    opt_gen = config["opt_gen"](gen.parameters(), lr=lr)
    torch.cuda.empty_cache()
    for _ in tqdm(range(epochs)):
        torch.cuda.empty_cache()
        for idx in range(0, len(X), batch_size):
            X_batch = torch.tensor(np.array(X[idx : idx + batch_size]))
            real = X_batch.view(-1, IMG_SIZE * IMG_SIZE * 3).to(device)
            batch_size = real.shape[0]
            noise = torch.randn(batch_size, z_dim).to(device)
            fake = gen(noise)
            disc_real = desc(real).view(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = desc(fake).view(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            lossD = (lossD_real + lossD_fake) / 2
            desc.zero_grad()
            lossD.backward(retain_graph=True)
            opt_disc.step()
            output = desc(fake).view(-1)
            lossG = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            lossG.backward()
            opt_gen.step()
            if tune_using:
                tune.report(mean_loss=lossG.item() + lossD.item())
        if tune_using is False:
            print(lossD.item())
            print(lossG.item())
            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 3, IMG_SIZE, IMG_SIZE)
                idx = -1
                for _ in range(25):
                    fixed_noise = torch.randn((batch_size, z_dim)).to(device)
                    fake = gen(fixed_noise).reshape(-1, 3, IMG_SIZE, IMG_SIZE)
                    idx += 1
                    plt.figure(figsize=(25, 12))
                    plt.imshow(fake[0].cpu().view(IMG_SIZE, IMG_SIZE, 3) * 255)
                    plt.savefig(f"./final-results/{idx}.png")


analysis = tune.run(
    train,
    config={
        "z_dim": tune.grid_search([16, 32, 64, 128, 256, 512]),
        "batch_size": tune.grid_search([16, 32, 64, 128]),
        "gen_starter": tune.grid_search([64, 128, 256, 512]),
        "opt_disc": tune.grid_search(
            [
                torch.optim.Adam,
                torch.optim.AdamW,
                torch.optim.Adamax,
            ]
        ),
        "opt_gen": tune.grid_search(
            [
                torch.optim.Adam,
                torch.optim.AdamW,
                torch.optim.Adamax,
            ]
        ),
        "lr": tune.grid_search(
            [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 10e-4]
        ),
        "epochs": tune.grid_search([12]),
    },
    resources_per_trial={"gpu": 1},
)


print("Best config: ", analysis.get_best_config(metric="mean_loss", mode="min"))
import pandas as pd

df = analysis.results_df
df.to_csv("./logs.csv")
# config = {
#     "z_dim": 512,
#     "opt_disc": torch.optim.Adam,
#     "opt_gen": torch.optim.Adamax,
#     "lr": 0.0007,
#     "epochs": 25,
# }
# train(config, tune_using=False)

