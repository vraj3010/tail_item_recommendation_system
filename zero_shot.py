import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, a_same, i_tail):
        input_vector = torch.cat([a_same, i_tail], dim=1)  # Eq (9)
        return self.model(input_vector)


# -------------------------------
# Discriminator (D_i) for WGAN
# -------------------------------
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)


def synthetic_representation(generator, discriminator, a_same,i_head,i_tail,optimizer_G, optimizer_D,N_D):

    for _ in range(N_D):
        i_G = generator(a_same, i_tail)
        score_G = discriminator(torch.cat([a_same, i_G], dim=1))
        score_head = discriminator(torch.cat([a_same, i_head], dim=1))
        loss_D = -score_G.mean() + score_head.mean()

        # Update Discriminator
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

    # Compute Generator loss (Eq 11)
    loss_G = -discriminator(torch.cat([a_same, generator(a_same, i_tail)], dim=1)).mean()

    # Update Generator
    optimizer_G.zero_grad()
    loss_G.backward()
    optimizer_G.step()

