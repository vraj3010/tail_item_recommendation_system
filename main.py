import pandas as pd
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from train_test import train_test_split_per_user
import torch
from utils import separate_head_tail_items,create_interaction_edges,get_matching_head_items
from head_gcn import head_items_self_supervised_training
from tail_gcn import *
from model import LightGCN
from synthetic_rep_utils import compute_a_same
from zero_shot import *
from torch_geometric.data import Data
# Data Preparation
ratings = pd.read_csv('ml-latest-small/ratings.csv')

user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

ratings['userId'] = user_encoder.fit_transform(ratings['userId'])
ratings['movieId'] = movie_encoder.fit_transform(ratings['movieId'])

interaction_counts = ratings['movieId'].value_counts().to_dict()

# Separate Head and Tail Items
head_items, tail_items = separate_head_tail_items(interaction_counts, head_threshold=40)

# print(len(tail_items))
num_users = ratings['userId'].nunique()
num_movies = ratings['movieId'].nunique()
head_items=[i+num_users for i in head_items]

tail_items=[i+num_users for i in tail_items]

print(num_users,num_movies)
int_edges = create_interaction_edges(ratings['userId'], ratings['movieId'], ratings['rating'])
print(int_edges.shape)
int_edges[1] += num_users
user_ids = int_edges[0].to(dtype=torch.long)
indices = torch.arange(0, int_edges.shape[1], dtype=torch.long)

matching_head_items=get_matching_head_items(int_edges,head_items,tail_items)
print(matching_head_items.shape)
# Train-Test Split (Per User)
# train_idx, test_idx = train_test_split_per_user(indices, user_ids, test_size=0.2)



# Initialize the model
embedding_dim = 64  # Choose the embedding dimension
model = LightGCN(num_users, num_movies,embedding_dim)
# Training the Model
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 1

K = 64  # Number of head items per batch
H = 64  # Number of tail items per batch
T = 32  # Number of users per batch
input_dim = 64
output_dim = 64
discriminator_input_dim=input_dim+output_dim
generator = Generator(input_dim, output_dim)
discriminator = Discriminator(discriminator_input_dim)
learning_rate = 0.001
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)
N_D = 5


for epoch in range(epochs):





    # total_loss_head = head_items_self_supervised_training(model, optimizer, int_edges, head_items,
    #                                                       tail_items, K=K, tau=0.1, deletion_rate=0.1)
    #
    # print(f"Epoch [{epoch + 1}/{epochs}], Loss (Head): {total_loss_head:.4f}")
    #
    user_embeddings = model.user_embedding.weight  # Shape: (num_users, embedding_dim)
    item_embeddings = model.item_embedding.weight  # Shape: (num_items, embedding_dim)
    all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)  # Shape: (num_users + num_items, embedding_dim)

    i_head=all_embeddings[head_items]
    #
    # total_loss_tail = tail_items_self_supervised_training(model, optimizer, int_edges, head_items,
    #                                                       tail_items, H=H, tau=0.1)
    # print(f"Epoch [{epoch + 1}/{epochs}], Loss (Tail): {total_loss_tail:.4f}")
    #
    # user_embeddings = model.user_embedding.weight  # Shape: (num_users, embedding_dim)
    # item_embeddings = model.item_embedding.weight  # Shape: (num_items, embedding_dim)
    # all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)  # Shape: (num_users + num_items, embedding_dim)
    #
    i_tail=all_embeddings[tail_items]

    # Synthetic Representation

    # R_head_matched = all_embeddings[matching_head_items]  # (num_tail_items, embedding_dim)
    # print(R_head_matched.shape)
    # R_tail = all_embeddings[tail_items]  # (num_tail_items, embedding_dim)
    # # Element-wise multiplication
    # a_same = R_head_matched * R_tail
    # # a_same = compute_a_same(model,int_edges, head_items, tail_items)
    # print(a_same.shape)
    # loss_G,loss_D=synthetic_representation(generator,discriminator,a_same,i_head,i_tail,optimizer_G,optimizer_D,N_D)
    # print(f"Epoch {epoch + 1}: Loss D = {loss_D.item()}, Loss G = {loss_G.item()}")
    # batch_item_embeddings=
    # === Predictions and Parameter Updates ===
    k=1
    for t in range(0, num_users, T):  # Process in batches of T users

        k-=1
        if k<0:
            break
        batch_users = torch.arange(t, min(t + T, num_users), dtype=torch.long)
        batch_user_embeddings = user_embeddings[batch_users]

        # Compute predicted scores \hat{y} (Eq. 13)
        # batch_item_embeddings = item_embeddings
        # predictions = torch.matmul(batch_user_embeddings, batch_item_embeddings.T)

        # Select positive and negative samples
        mask = torch.isin(int_edges[0], batch_users)
        pos_items = int_edges[1][mask]
        pos_items-=num_users
        neg_items = torch.randint(0, num_movies, (len(pos_items),), dtype=torch.long)
        neg_items-=num_users
        print(pos_items.shape)
        y_ui = torch.matmul(batch_user_embeddings, item_embeddings[pos_items].T)
        y_uj = torch.matmul(batch_user_embeddings, item_embeddings[neg_items].T)

        # Compute BPR Loss (Eq. 14)
        loss_main = -torch.log(torch.sigmoid(y_ui - y_uj)).mean()

        # Update Model Parameters
        optimizer.zero_grad()
        loss_main.backward()
        optimizer.step()

        print(f"Batch {t // T + 1}: Loss Main = {loss_main.item()}")