# %% [markdown]
# # Tutorial: Generative Model Learning of Shapes
#
# ## Introduction to Generative Models
#
# This tutorial demonstrates how learn the distribution of complex objects using generative models. The metholology is inspired from these [two](https://arxiv.org/abs/2308.03662) [papers](https://arxiv.org/abs/1810.01118).
#
# We have a probability distribution of a set of ellipsis, each centered in (0,0), and we want to construct a surrogate probability distribution using generative models.
#
#
# Let's first write the code for data generation.

# %%
import torch

n = 600
density = 100
torch.manual_seed(42)


def generate_data(n, density=1000):

    theta = 2 * torch.pi * torch.linspace(0, 1, density)
    x = torch.cos(theta)
    y = torch.sin(theta)
    data_ref = torch.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
    data = torch.zeros((n, density, 2))

    for i in range(n):
        A = -0.4 + 0.8 * torch.rand(2, 2)
        A = A / torch.linalg.norm(A)
        A = A + torch.eye(2)
        A = 2 * A / torch.linalg.norm(A)
        data[i] = data_ref @ A

    data = data.reshape(n, -1)
    return data


data = generate_data(n, density)


# %% [markdown]
# Let's plot an ellipse just to check that everything is going smoothly.

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ellipse = data.reshape(-1, density, 2)[0]
ax.scatter(ellipse[:, 0], ellipse[:, 1])
ax.set(xlim=(-2, 2), ylim=(-2, 2))
plt.clf()


# %% [markdown]
# Now we are ready to define our generative model, which is composed by an AutoEncoder only.
# First we define the AutoEncoder. As all the ellipsis in the dataset have barycenter as the origin, we impose this information in the AutoEncoder, we adopt POD and BatchNorm for further regularization. We add a sinkhorn regularization to the loss function, so that we are able to directly sample from the latent space in order to get good representations.

# %%

from torch import nn
from pina.model.block import PODBlock


def barycenter_projector(x):
    x = x.reshape(-1, density, 2)
    tmp = x - torch.mean(x, axis=1).reshape(-1, 1, 2)
    return tmp.reshape(-1, 2 * density)


class LBR(torch.nn.Module):
    def __init__(self, input_size, output_size, final=False, drop_prob=0.1):
        super().__init__()
        self.lin = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(output_size)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, input):
        tmp = self.dropout(self.bn(self.relu(self.lin(input))))
        return tmp


class GenEncoder(torch.nn.Module):
    def __init__(
        self,
        input_size,
        latent_size,
        hidden_size=500,
        num_layers=5,
        drop_prob=0.1,
    ):
        super().__init__()
        self.layers = []
        self.layers.append(LBR(input_size, hidden_size, drop_prob=drop_prob))
        for _ in range(num_layers - 2):
            self.layers.append(
                LBR(hidden_size, hidden_size, drop_prob=drop_prob)
            )
        self.layers.append(nn.Linear(hidden_size, latent_size))
        self.layers.append(
            nn.BatchNorm1d(latent_size, affine=False, track_running_stats=False)
        )
        self.nn = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.nn(x)


class GenDecoder(torch.nn.Module):
    def __init__(
        self,
        latent_size,
        output_size,
        hidden_size=500,
        num_layers=5,
        drop_prob=0.1,
    ):
        super().__init__()
        self.layers = []
        self.layers.append(LBR(latent_size, hidden_size, drop_prob=drop_prob))
        for i in range(num_layers - 2):
            self.layers.append(
                LBR(hidden_size, hidden_size, drop_prob=drop_prob)
            )

        self.layers.append(nn.Linear(hidden_size, output_size))
        self.nn = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.nn(x)


class GenAutoencoder(torch.nn.Module):
    def __init__(
        self,
        pod_size,
        latent_size,
        hidden_size=500,
        num_layers=5,
        drop_prob=0.1,
    ):
        super().__init__()
        self.pod_size = pod_size
        self.latent_size = latent_size
        self.pod = PODBlock(64)
        self.encoder = GenEncoder(
            pod_size, latent_size, hidden_size, num_layers, drop_prob
        )
        self.decoder = GenDecoder(
            latent_size, pod_size, hidden_size, num_layers, drop_prob
        )

    def forward(self, x):
        tmp = self.pod(x)
        latent = self.encoder(tmp)
        tmp = self.decoder(latent)
        tmp = self.pod.expand(tmp)
        tmp = barycenter_projector(tmp)
        return (latent, tmp)

    def sample(self, n=1):
        tmp = self.decoder(torch.randn(n, self.latent_size))
        tmp = self.pod.expand(tmp)
        tmp = barycenter_projector(tmp)
        return tmp


# %% [markdown]
# We can now define the loss function, which is a combination of autoencoder loss (for learning the features) and sinkhorn (for regularizing the distribution of the latent space).

# %%
from pina.loss import LossInterface


class Loss(LossInterface):

    def __init__(self, eps, diff_conv, L, gamma):
        super().__init__()
        self.eps = eps
        self.diff_conv = diff_conv
        self.L = L
        self.gamma = gamma

    def sinkhorn_loss_single(self, a, b):
        diff = 1e8
        counter = 0
        C = torch.sum(
            (a.reshape(a.shape[0], 1, -1) - b.reshape(1, b.shape[0], -1)) ** 2,
            axis=2,
        )
        K = torch.exp(-C / self.eps)
        u = torch.ones_like(a)
        while diff > self.diff_conv and counter < self.L:
            u_old = u
            v = 1 / (C.T @ u)
            u = 1 / (C @ v)
            diff = torch.max(torch.abs(u - u_old))
            counter = counter + 1
        R = u @ v.T
        return 1 / (a.shape[0]) * torch.sum(R * C)

    def sinkhorn_loss(self, a, b):
        return (
            self.sinkhorn_loss_single(a, b)
            - 0.5 * self.sinkhorn_loss_single(a, a)
            - 0.5 * self.sinkhorn_loss_single(b, b)
        )

    def forward(self, rec_, output_):
        z, x_hat = rec_
        normal_samples = torch.randn(z.shape[0], z.shape[1])
        return torch.linalg.norm(
            x_hat - output_
        ) + self.gamma * self.sinkhorn_loss(z, normal_samples)


loss = Loss(eps=0.01, L=10000, gamma=1, diff_conv=1e-02)


# %% [markdown]
# We can finally apply the typical PINA workflow:

# %%
from pina.solver import SupervisedSolver
from pina.problem.zoo import SupervisedProblem
from pina import Trainer

problem = SupervisedProblem(data, data)
genae = GenAutoencoder(2 * density, 64, 5)
genae.pod.fit(data)
solver = SupervisedSolver(problem, genae, loss=loss, use_lt=False)

trainer = Trainer(
    solver,
    max_epochs=500,
    train_size=1.0,
    test_size=0.0,
    batch_size=100,
    accelerator="cpu",
    enable_model_summary=False,
    logger=False,
)

trainer.train()


# %% [markdown]
# Let's plot some samples to check that there is variability.

# %%
import matplotlib.pyplot as plt

genae.eval()

for _ in range(10):
    tmp = genae.sample().detach().numpy().reshape(-1, 2)
    plt.scatter(tmp[:, 0], tmp[:, 1])
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.show()
    plt.clf()


# %% [markdown]
# # Whats next?
#
# There are lot of directions to explore:
#
# *   tweak the sinkhorn parameters to study how the affect convergence.
# *   changing the latent space dimensionality or the POD dimensionality.
# * adopt measures different from sinkhorn for regularizing the latent space.
#
#
