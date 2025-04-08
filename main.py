import pandas as pd
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import LabelEncoder
from train_test import train_test_split_per_user
import torch
from head_gcn import head_items_self_supervised_training
from tail_gcn import *
from utils import *
from model import LightGCN
from zero_shot import *
from torch_geometric.data import Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data Preparation
ratings = pd.read_csv('ml-latest-small/ratings.csv')

user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

ratings['userId'] = user_encoder.fit_transform(ratings['userId'])
ratings['movieId'] = movie_encoder.fit_transform(ratings['movieId'])

neg_samples=get_negative_items(ratings)
# negative_items = get_negative_items(ratings)

interaction_counts = ratings['movieId'].value_counts().to_dict()

# Separate Head and Tail Items
head_items, tail_items = separate_head_tail_items(interaction_counts, head_threshold=15)

num_users = ratings['userId'].nunique()
num_movies = ratings['movieId'].nunique()
head_items = torch.tensor([i + num_users for i in head_items], dtype=torch.long, device=device)
tail_items = torch.tensor([i + num_users for i in tail_items], dtype=torch.long, device=device)

# print(num_users, num_movies)

int_edges = create_interaction_edges(ratings['userId'], ratings['movieId'], ratings['rating']).to(device)
int_edges[1] += num_users
user_ids = int_edges[0].to(dtype=torch.long, device=device)
indices = torch.arange(0, int_edges.shape[1], dtype=torch.long, device=device)

# Train-Test Split (Per User)
train_idx, test_idx = train_test_split_per_user(indices, user_ids, test_size=0.2)
test_edges = int_edges[:, test_idx]  # Select test edges
int_edges = int_edges[:2,train_idx]  # Select train edges
test_set=create_test_set(test_edges)

# matching_head_items = get_matching_head_items(int_edges, head_items, tail_items).to(device)

# Initialize the model
embedding_dim = 64  # Choose the embedding dimension
model = LightGCN(num_users, num_movies, embedding_dim).to(device)  # Move model to GPU


# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 1
K = 64  # Number of batches in head item
H = 32  # Number of batches in tail item
T = 124  # Number of users per batch
input_dim = 64
output_dim = 64
discriminator_input_dim = input_dim + output_dim

generator = Generator(input_dim, output_dim).to(device)
discriminator = Discriminator(discriminator_input_dim).to(device)
learning_rate = 0.001
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)
N_D = 5
# ndcg_calculation_2(model, test_set, neg_samples, num_users,int_edges,head_items,k=10)
# ndcg_calculation_head(model, test_set, neg_samples, num_users,int_edges,head_items,k=10)
# ndcg_calculation_tail(model, test_set, neg_samples, num_users,int_edges,tail_items,k=2)

print("Training started")
for epoch in range(epochs):
    total_loss_head = head_items_self_supervised_training(
        model, optimizer, int_edges, head_items, tail_items, K=K, tau=0.1, deletion_rate=0.1
    )
    print(f"Epoch [{epoch + 1}/{epochs}], Loss (Head): {total_loss_head:.4f}")

    graph = Data(edge_index=int_edges, num_nodes=int_edges.max().item() + 1).to(device)
    with torch.no_grad():
        all_embeddings = model(graph)
        i_head2 = all_embeddings[head_items]
        repeat_factor = (len(tail_items) + len(head_items) - 1) // len(head_items)
        i_head = i_head2.repeat(repeat_factor, 1)[:len(tail_items)]  # Repeat and truncate


    total_loss_tail = tail_items_self_supervised_training(
        model, optimizer, int_edges, head_items, tail_items, H=128, tau=0.1
    )



    print(f"Epoch [{epoch + 1}/{epochs}], Loss (Tail): {total_loss_tail:.4f}")

    user_embeddings = model.user_embedding.weight
    item_embeddings = model.item_embedding.weight
    all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
    i_tail = all_embeddings[tail_items]

    # Synthetic Representation
    interaction_matrix=construct_interaction_matrix(int_edges,head_items,tail_items,num_users,num_movies)
    graph=construct_interaction_graph(interaction_matrix,head_items,tail_items)
    a_same = model(graph)

    a_same=a_same[tail_items]
    loss_G, loss_D = synthetic_representation(generator, discriminator, a_same, i_head, i_tail, optimizer_G, optimizer_D, N_D)
    print(f"Epoch [{epoch + 1}/{epoch}] : Loss D = {loss_D}, Loss G = {loss_G}")

    #Main_Loss

    z_i = generator(a_same, i_tail)

    with torch.no_grad():
        new_embeddings = all_embeddings.clone()
        new_embeddings[tail_items] = z_i.detach()
        all_embeddings = new_embeddings

    users=torch.randperm(num_users,device=device)
    for t in range(0, num_users, T):
        batch_users = users[t:min(t+T,num_users)]

        mask = torch.isin(int_edges[0], batch_users)
        pos_items = int_edges[1][mask]
        corr_pos_users = int_edges[0][mask]

        batch_user_embeddings = model.get_embeddings(corr_pos_users)

        neg_items = torch.tensor(
            [random.choice(neg_samples[user.item()])+num_users for user in corr_pos_users],
            dtype=torch.long, device=device
        )

        pos_embeddings = model.get_embeddings(pos_items)  # Instead of all_embeddings[pos_items]
        neg_embeddings = model.get_embeddings(neg_items)  # Instead of all_embeddings[neg_items]
        y_ui = (batch_user_embeddings * pos_embeddings).sum(dim=1, keepdim=True)
        y_uj = (batch_user_embeddings * neg_embeddings).sum(dim=1, keepdim=True)
        model.train()
        # Compute BPR Loss
        loss_main = -torch.log(torch.sigmoid(y_ui - y_uj)).mean()
        # Update Model Parameters
        optimizer.zero_grad()
        loss_main.backward()
        optimizer.step()

        # print(f"Batch {t // T + 1}: Loss Main = {loss_main.item()}")

    ndcg_calculation_2(model, test_set, neg_samples, num_users,int_edges,head_items,k=10)
    ndcg_calculation_head(model, test_set, neg_samples, num_users,int_edges,head_items,k=10)
    ndcg_calculation_tail(model, test_set, neg_samples, num_users,int_edges,tail_items,k=2)
















