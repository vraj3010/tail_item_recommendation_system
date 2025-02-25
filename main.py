import pandas as pd
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import LabelEncoder
from train_test import train_test_split_per_user
import torch
from utils import separate_head_tail_items,create_interaction_edges,get_matching_head_items,get_negative_items,ndcg_at_k
from head_gcn import head_items_self_supervised_training
from tail_gcn import *
from model import LightGCN
from zero_shot import *
from torch_geometric.data import Data
# Data Preparation
ratings = pd.read_csv('ml-latest-small/ratings.csv')

user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

ratings['userId'] = user_encoder.fit_transform(ratings['userId'])
ratings['movieId'] = movie_encoder.fit_transform(ratings['movieId'])

negative_items = get_negative_items(ratings)

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
int_edges[1] += num_users
user_ids = int_edges[0].to(dtype=torch.long)
indices = torch.arange(0, int_edges.shape[1], dtype=torch.long)
# Train-Test Split (Per User)
train_idx, test_idx = train_test_split_per_user(indices, user_ids, test_size=0.2)
test_edges = int_edges[:, test_idx]  # Select test edges
int_edges = int_edges[:, train_idx]  # Select train edges


# Identify negative interactions (ratings < 4)
neg_mask = ratings['rating'] < 4
neg_user_ids = ratings.loc[neg_mask, 'userId'].values
neg_movie_ids = ratings.loc[neg_mask, 'movieId'].values + num_users  # Adjust movie IDs

neg_user_ids = np.array(neg_user_ids)  # Convert list to numpy array
neg_movie_ids = np.array(neg_movie_ids)  # Convert list to numpy array

neg_edges = torch.tensor(np.stack([neg_user_ids, neg_movie_ids]), dtype=torch.long)

# Perform 20% per-user split for negative samples
neg_user_ids_tensor = torch.tensor(neg_user_ids, dtype=torch.long)
neg_indices = torch.arange(0, neg_edges.shape[1], dtype=torch.long)
neg_train_idx, neg_test_idx = train_test_split_per_user(neg_indices, neg_user_ids_tensor, test_size=0.2)

# Select test negative edges
test_neg_edges = neg_edges[:, neg_test_idx]

# Append negative edges to test_edges
test_edges = torch.cat([test_edges, test_neg_edges], dim=1)

matching_head_items=get_matching_head_items(int_edges,head_items,tail_items)

# Initialize the model
embedding_dim = 64  # Choose the embedding dimension
model = LightGCN(num_users, num_movies,embedding_dim)
# Training the Model
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5

K = 64  # Number of head items per batch
H = 64  # Number of tail items per batch
T = 124  # Number of users per batch
input_dim = 64
output_dim = 64
discriminator_input_dim=input_dim+output_dim
generator = Generator(input_dim, output_dim)
discriminator = Discriminator(discriminator_input_dim)
learning_rate = 0.001
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)
N_D = 5


# for epoch in range(epochs):
#
#     total_loss_head = head_items_self_supervised_training(model, optimizer, int_edges, head_items,
#                                                           tail_items, K=K, tau=0.1, deletion_rate=0.1)
#     print(f"Epoch [{epoch + 1}/{epochs}], Loss (Head): {total_loss_head:.4f}")
#
#     user_embeddings = model.user_embedding.weight  # Shape: (num_users, embedding_dim)
#     item_embeddings = model.item_embedding.weight  # Shape: (num_items, embedding_dim)
#     all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)  # Shape: (num_users + num_items, embedding_dim)
#
#     i_head=all_embeddings[matching_head_items]
#
#     total_loss_tail = tail_items_self_supervised_training(model, optimizer, int_edges, head_items,
#                                                           tail_items, H=H, tau=0.1)
#     print(f"Epoch [{epoch + 1}/{epochs}], Loss (Tail): {total_loss_tail:.4f}")
#
#     user_embeddings = model.user_embedding.weight  # Shape: (num_users, embedding_dim)
#     item_embeddings = model.item_embedding.weight  # Shape: (num_items, embedding_dim)
#     all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)  # Shape: (num_users + num_items, embedding_dim)
#
#     i_tail=all_embeddings[tail_items]
#
#     # Synthetic Representation
#
#     R_head_matched = all_embeddings[matching_head_items]  # (num_tail_items, embedding_dim)
#     R_tail = all_embeddings[tail_items]  # (num_tail_items, embedding_dim)
#     a_same = R_head_matched * R_tail
#     loss_G,loss_D=synthetic_representation(generator,discriminator,a_same,i_head,i_tail,optimizer_G,optimizer_D,N_D)
#     print(f"Epoch {epoch + 1}: Loss D = {loss_D.item()}, Loss G = {loss_G.item()}")
#
#     for t in range(0, num_users, T):
#
#         batch_users = torch.arange(t, min(t + T, num_users), dtype=torch.long)
#         mask = torch.isin(int_edges[0], batch_users)
#         pos_items = int_edges[1][mask]
#         corr_pos_users= int_edges[0][mask]
#         batch_user_embeddings = user_embeddings[corr_pos_users]
#         pos_items-=num_users
#         z_i = generator(a_same[corr_pos_users], item_embeddings[pos_items])
#         neg_items=[]
#         for user in corr_pos_users.tolist():
#             neg_list = negative_items.get(user, [])  # Get negative items for this user
#             if len(neg_list) > 0:
#                 neg_item = torch.tensor(neg_list)[torch.randint(0, len(neg_list), (1,))].item()
#             else:
#                 neg_item = torch.randint(0, num_movies, (1,), dtype=torch.long).item()
#
#             neg_items.append(neg_item)
#
#         y_ui = (batch_user_embeddings * item_embeddings[pos_items]).sum(dim=1, keepdim=True)
#         y_uj = (batch_user_embeddings * item_embeddings[neg_items]).sum(dim=1, keepdim=True)
#
#         # Compute BPR Loss (Eq. 14)
#         loss_main = -torch.log(torch.sigmoid(y_ui - y_uj)).mean()
#
#         # Update Model Parameters
#         optimizer.zero_grad()
#         loss_main.backward()
#         optimizer.step()
#
#         print(f"Batch {t // T + 1}: Loss Main = {loss_main.item()}")

# ----- NDCG Calculation -----

# Get user & item embeddings
user_embeddings = model.user_embedding.weight  # Shape: (num_users, embedding_dim)
item_embeddings = model.item_embedding.weight  # Shape: (num_items, embedding_dim)

test_user_ids = test_edges[0]
test_item_ids = test_edges[1]

test_item_ids-=num_users
# Compute predicted scores: dot product of user & item embeddings
predicted_scores = (user_embeddings[test_user_ids] * item_embeddings[test_item_ids]).sum(dim=1)

# True relevance labels: 1 for positive items, 0 for negative items
num_pos = test_edges.shape[1] - test_neg_edges.shape[1]  # Count of positive ratings
num_neg = test_neg_edges.shape[1]  # Count of negative ratings
true_relevance = torch.cat([torch.ones(num_pos), torch.zeros(num_neg)])

ndcg_score = ndcg_at_k(true_relevance, predicted_scores, k=10)
print(f"NDCG@10: {ndcg_score.item()}")