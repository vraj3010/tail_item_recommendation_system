import pandas as pd
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import LabelEncoder
from train_test import train_test_split_per_user
import torch
from DNN import DNN
from head_gcn import head_items_self_supervised_training
from tail_gcn import *
from utils import *
from model import LightGCN
from zero_shot import *
from torch_geometric.data import Data
import gaussian_diffusion as gd
from tqdm import tqdm
from torch.utils.data import DataLoader
###
torch.set_printoptions(sci_mode=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data Preparation
ratings = pd.read_csv('ml-latest-small/ratings.csv')

user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

ratings['userId'] = user_encoder.fit_transform(ratings['userId'])
ratings['movieId'] = movie_encoder.fit_transform(ratings['movieId'])

neg_samples=get_negative_items(ratings)

interaction_counts = ratings['movieId'].value_counts().to_dict()

# Separate Head and Tail Items
head_items, tail_items = separate_head_tail_items(interaction_counts, head_threshold=15)
print(len(head_items))
num_users = ratings['userId'].nunique()
num_movies = ratings['movieId'].nunique()
print(num_users,num_movies)
head_items = torch.tensor([i + num_users for i in head_items], dtype=torch.long, device=device)
tail_items = torch.tensor([i + num_users for i in tail_items], dtype=torch.long, device=device)

int_edges = create_interaction_edges(ratings['userId'], ratings['movieId'], ratings['rating']).to(device)
# int_edges[1] += num_users

user_ids = int_edges[0].to(dtype=torch.long, device=device)
indices = torch.arange(0, int_edges.shape[1], dtype=torch.long, device=device)

# Train-Test Split (Per User)
train_idx, test_idx = train_test_split_per_user(indices, user_ids, test_size=0.2)

####
train_edges = int_edges[:, train_idx]
train_adj = create_adj_matrix(train_edges, num_users, num_movies)
# train_r = adj_to_r_mat(train_adj, num_users, num_movies)
####

int_edges[1] += num_users
test_edges = int_edges[:, test_idx]  # Select test edges
int_edges = int_edges[:2,train_idx]  # Select train edges
test_set=create_test_set(test_edges)

# Initialize the model
embedding_dim = 64  # Choose the embedding dimension
model = LightGCN(num_users, num_movies, embedding_dim).to(device)  # Move model to GPU


# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 1
K = 128 # Number of batches in head item
H = 128  # Number of batches in tail item
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

#Diffusion Recommendation
mean_type = gd.ModelMeanType.START_X
lr=5e-5
weight_decay=0
batch_size=400
dims=str([500])
emb_size=10
steps=5
noise_scale=0.0001
noise_min=0.0005
noise_max=0.005
sampling_steps=0
reweight=1
round=1
noise_schedule='linear-var'
diffusion = gd.GaussianDiffusion(mean_type, noise_schedule, \
                                 noise_scale, noise_min, noise_max, steps, device).to(device)
out_dims = [64,64]   #64,64
in_dims = out_dims[::-1]  #9724,500
model2 = DNN(in_dims, out_dims, emb_size, time_type="cat", norm=False).to(device)

optimizer2 = optim.AdamW(model2.parameters(), lr=lr, weight_decay=weight_decay)
print("models ready.")
mlp_num = sum([param.nelement() for param in model2.parameters()])
diff_num = sum([param.nelement() for param in diffusion.parameters()])  # 0
param_num = mlp_num + diff_num
print("Number of all parameters:", param_num)
train_dataset = InteractionDataset(int_edges)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#################


print("Training started")
interaction_matrix=construct_interaction_matrix(int_edges,head_items,tail_items,num_users,num_movies)
iterator = tqdm(range(epochs))
for epoch in iterator:
    total_loss_head = head_items_self_supervised_training(
        model, optimizer, int_edges, head_items, K=K, tau=0.1, deletion_rate=0.1
    )


    print(f"Epoch [{epoch + 1}/{epochs}], Loss (Head): {total_loss_head:.4f}")
    user_embeddings = model.user_embedding.weight
    item_embeddings = model.item_embedding.weight
    all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
    i_head2 = all_embeddings[head_items]
    repeat_factor = (len(tail_items) + len(head_items) - 1) // len(head_items)
    i_head = i_head2.repeat(repeat_factor, 1)[:len(tail_items)]  # Repeat and truncate

    total_loss_tail = tail_items_self_supervised_training(
        model, optimizer, int_edges, head_items, tail_items, H=128, tau=0.1
    )

    print(f"Epoch [{epoch + 1}/{epochs}], Loss (Tail): {total_loss_tail:.4f}")
    #
    user_embeddings = model.user_embedding.weight
    item_embeddings = model.item_embedding.weight
    all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
    i_tail = all_embeddings[tail_items]

    # # Synthetic Representation
    # interaction_matrix=construct_interaction_matrix(int_edges,head_items,tail_items,num_users,num_movies)
    # edge_index2,edge_weights2=construct_interaction_graph(interaction_matrix,head_items,tail_items)
    # a_same = model(edge_index2,edge_weights=edge_weights2)
    # #
    # a_same=a_same[tail_items]
    # loss_G, loss_D = synthetic_representation(generator, discriminator, a_same, i_head, i_tail, optimizer_G, optimizer_D, N_D)
    # print(f"Epoch [{epoch + 1}/{epoch}] : Loss D = {loss_D}, Loss G = {loss_G}")
    # #
    # # #Main_Loss
    # #
    # z_i = generator(a_same, i_tail)
    #
    # with torch.no_grad():
    #     new_embeddings = all_embeddings.clone()
    #     new_embeddings[tail_items] = z_i.detach()
    #     all_embeddings = new_embeddings




    #difuusion Recommendation
    print("hii")
    if epoch%100==0:

        n_epoch=100
        for epoched in range(n_epoch):
            tot=torch.randperm(num_users+num_movies,device=device)
            model2.train()
            total_loss=0
            for t in range(0, num_users+num_movies, batch_size):
                batch = tot[t:min(t+batch_size,num_users+num_movies)]
                batch_emb=all_embeddings[batch]
                optimizer2.zero_grad()
                # print(batch_emb.shape)
                losses = diffusion.training_losses(model2, batch_emb, reweight)

                loss = losses["loss"].mean()
                total_loss += loss
                loss.backward()
                optimizer2.step()
            print(epoched)

        prediction = diffusion.p_sample(model2, all_embeddings, sampling_steps, sampling_noise=False)
        print(prediction.shape)
        with torch.no_grad():
            model.user_embedding.weight.copy_(prediction[:num_users])
            model.item_embedding.weight.copy_(prediction[num_users:num_users + num_movies])

    #bpr loss
    users = torch.randperm(num_users, device=device)
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

    # if epoch%10==0 and epoch!=0:
    #     ndcg_calculation_2(model, test_set, neg_samples, num_users, int_edges, head_items, k=10)
    #     ndcg_calculation_head(model, test_set, neg_samples, num_users, int_edges, head_items, k=10)
    #     ndcg_calculation_tail(model, test_set, neg_samples, num_users, int_edges, tail_items, k=2)
    #     ndcg_calculation_2(model, test_set, neg_samples, num_users, int_edges, head_items, k=10, N=80)
    #     ndcg_calculation_head(model, test_set, neg_samples, num_users, int_edges, head_items, k=10, N=80)
    #     ndcg_calculation_tail(model, test_set, neg_samples, num_users, int_edges, tail_items, k=10, N=80)


ndcg_calculation_2(model, test_set, neg_samples, num_users,int_edges,head_items,k=10)
ndcg_calculation_head(model, test_set, neg_samples, num_users,int_edges,head_items,k=10)
ndcg_calculation_tail(model, test_set, neg_samples, num_users,int_edges,tail_items,k=2)
ndcg_calculation_2(model, test_set, neg_samples, num_users,int_edges,head_items,k=10,N=80)
ndcg_calculation_head(model, test_set, neg_samples, num_users,int_edges,head_items,k=10,N=80)
ndcg_calculation_tail(model, test_set, neg_samples, num_users,int_edges,tail_items,k=10,N=80)