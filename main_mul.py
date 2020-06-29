import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tensorboardX import SummaryWriter
from scipy import sparse
import models_mul
import data
import metric

import torch.nn.functional as F
import pandas as pd
import os
import pickle
from sklearn.cluster import KMeans
from tqdm import tqdm

torch.backends.cudnn.enabled = False
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoders for Collaborative Filtering')
parser.add_argument('--data', type=str, default='amazon',
                    help='Movielens-20m dataset location')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--wd', type=float, default=0.001,
                    help='weight decay coefficient')
parser.add_argument('--batch_size', type=int, default=100,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
# parser.add_argument('--total_anneal_steps', type=int, default=200000,
#                     help='the total number of gradient updates for annealing')
parser.add_argument('--anneal_cap', type=float, default=0.2,
                    help='largest annealing parameter')
parser.add_argument('--seed', type=int, default=98765,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--keep', type=float, default=0.5,
                    help='Keep probability for dropout, in (0,1].')
parser.add_argument('--tau', type=float, default=0.1,
                    help='Temperature of sigmoid/softmax, in (0,oo).')
parser.add_argument('--std', type=float, default=0.075,
                    help='Standard deviation of the Gaussian prior.')
parser.add_argument('--kfac', type=int, default=7,
                    help='Number of facets (macro concepts).')
parser.add_argument('--dfac', type=int, default=100,
                    help='Dimension of each facet.')
parser.add_argument('--nogb', action='store_true', default=False,
                    help='Disable Gumbel-Softmax sampling.')
# args = parser.parse_args()
args = parser.parse_known_args()
args = args[0]

# Set the random seed manually for reproductibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################
# args.data = 'alishop-7c'
loader = data.DataLoader(args.data)

n_items = loader.load_n_items()
train_data = loader.load_data('train')
vad_data_tr, vad_data_te = loader.load_data('validation')
test_data_tr, test_data_te = loader.load_data('test')

N = train_data.shape[0]
idxlist = list(range(N))

num_batches = int(np.ceil(float(N) / args.batch_size))
total_anneal_steps = 5 * num_batches

###############################################################################
# load item title
###############################################################################
# with open(os.path.join(args.data, 'meta_encoded.pickle'), 'rb') as f:
#     dataset = pickle.load(f)
#
# item_title = dataset['data']
# vocab2index = dataset['vocab2index']
# cat2index = dataset['cat2index']
# with open(os.path.join(args.data, 'pro_sg/unique_sid.txt'), "r") as f:
#     unique_sid = []
#     for l in f:
#         unique_sid.append(l.rstrip("\n"))
#
# # filter items
# item_title = item_title[item_title['asin'].isin(unique_sid)]
# titles = []
# for i in range(len(item_title)):
#     titles.append(item_title['encoded'].iloc[i][0])
# titles = np.array(titles)

with open(os.path.join(args.data, 'meta_encoded.pickle'), 'rb') as f:
    dataset = pickle.load(f)

item_title = dataset['data_mat']
vocab2index = dataset['vocab2index']
cat2index = dataset['cat2index']
item2index = dataset['item2index']
category_id = np.array(dataset['category_id'])


embeddings = torch.load('embeddings.pt')

# kmeans = KMeans(n_clusters=args.kfac, random_state=args.seed).fit(embeddings)
# init_kmeans = torch.FloatTensor(kmeans.cluster_centers_)

titles = torch.from_numpy(embeddings).float().contiguous().to(device)

num_batches = int(np.ceil(float(N) / args.batch_size))
total_anneal_steps = 5 * num_batches

# define parameters for titles
embedding_dim = 100
hidden_dim = 100

###############################################################################
# load image
###############################################################################
img_features_filtered = torch.load('images_filtered.pt')
img_features_filtered = torch.from_numpy(img_features_filtered).float().contiguous().to(device)
###############################################################################
# Build the model
###############################################################################

p_dims = [2048, args.dfac, n_items]

# model = models_mul.MultiVAE(p_dims, tau=args.tau, std=args.std, kfac=args.kfac,
#                             vocab_size=embeddings.shape[1], embedding_dim=embedding_dim, hidden_dim=hidden_dim,
#                             title_data=titles, image_data=img_features_filtered,
#                             dropout=args.keep, nogb=args.nogb, q_dims=None).to(device)

model = models_mul.MultiVAE_Title(p_dims, title_data=titles, image_data=img_features_filtered,
                                  q_dims=None, dropout=args.keep, tau=args.tau, std=args.std, kfac=args.kfac, nogb=args.nogb).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
criterion = models_mul.loss_function

###############################################################################
# Training code
###############################################################################

# TensorboardX Writer

writer = SummaryWriter()


def sparse2torch_sparse(data):
    """
    Convert scipy sparse matrix to torch sparse tensor with L2 Normalization
    This is much faster than naive use of torch.FloatTensor(data.toarray())
    https://discuss.pytorch.org/t/sparse-tensor-use-cases/22047/2
    """
    samples = data.shape[0]
    features = data.shape[1]
    coo_data = data.tocoo()
    indices = torch.LongTensor([coo_data.row, coo_data.col])
    row_norms_inv = 1 / np.sqrt(data.sum(1))
    row2val = {i: row_norms_inv[i].item() for i in range(samples)}
    values = np.array([row2val[r] for r in coo_data.row])
    t = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), [samples, features])
    return t


def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())


def set_rng_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def train():
    set_rng_seed(args.seed)

    # Turn on training mode
    model.train()
    train_loss = 0.0
    start_time = time.time()
    global update_count

    np.random.shuffle(idxlist)

    for batch_idx, start_idx in enumerate(range(0, N, args.batch_size)):
        end_idx = min(start_idx + args.batch_size, N)
        data = train_data[idxlist[start_idx:end_idx]]
        data = naive_sparse2tensor(data).to(device)

        if total_anneal_steps > 0:
            anneal = min(args.anneal_cap,
                         1. * update_count / total_anneal_steps)
        else:
            anneal = args.anneal_cap

        optimizer.zero_grad()
        # recon_batch, mu, logvar = model(data)
        std_list, recon_batch = model(data)
        loss = criterion(data, std_list, recon_batch, anneal)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        update_count += 1

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | '
                  'loss {:4.2f}'.format(
                        epoch, batch_idx, len(range(0, N, args.batch_size)),
                        elapsed * 1000 / args.log_interval,
                        train_loss / args.log_interval))

            # Log loss to tensorboard
            n_iter = (epoch - 1) * len(range(0, N, args.batch_size)) + batch_idx
            writer.add_scalars('data/loss', {'train': train_loss / args.log_interval}, n_iter)

            start_time = time.time()
            train_loss = 0.0


def evaluate(data_tr, data_te):
    set_rng_seed(args.seed)
    # Turn on evaluation mode
    model.eval()
    total_loss = 0.0
    global update_count
    e_idxlist = list(range(data_tr.shape[0]))
    e_N = data_tr.shape[0]
    n100_list = []
    r20_list = []
    r50_list = []

    with torch.no_grad():
        for start_idx in range(0, e_N, args.batch_size):
            end_idx = min(start_idx + args.batch_size, N)
            data = data_tr[e_idxlist[start_idx:end_idx]]
            heldout_data = data_te[e_idxlist[start_idx:end_idx]]

            data_tensor = naive_sparse2tensor(data).to(device)

            if total_anneal_steps > 0:
                anneal = min(args.anneal_cap,
                             1. * update_count / total_anneal_steps)
            else:
                anneal = args.anneal_cap

            # recon_batch, mu, logvar = model(data_tensor)
            std_list, recon_batch = model(data_tensor)
            # loss = criterion(recon_batch, data_tensor, mu, logvar, anneal)
            loss = criterion(data_tensor, std_list, recon_batch, anneal)
            total_loss += loss.item()

            # Exclude examples from training set
            # recon_batch = F.log_softmax(recon_batch, 1) # TODO: not sure
            recon_batch = recon_batch.cpu().numpy()
            recon_batch[data.nonzero()] = -np.inf

            n100 = metric.NDCG_binary_at_k_batch(recon_batch, heldout_data, 100)
            r20 = metric.Recall_at_k_batch(recon_batch, heldout_data, 20)
            r50 = metric.Recall_at_k_batch(recon_batch, heldout_data, 50)

            n100_list.append(n100)
            r20_list.append(r20)
            r50_list.append(r50)

    total_loss /= len(range(0, e_N, args.batch_size))
    n100_list = np.concatenate(n100_list)
    r20_list = np.concatenate(r20_list)
    r50_list = np.concatenate(r50_list)

    return total_loss, np.mean(n100_list), np.mean(r20_list), np.mean(r50_list)


best_n100 = -np.inf
update_count = 0

# At any point you can hit Ctrl + C to break out of training early.
try:
    # train for items

    # disentangled
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        # train
        train()
        # evaluate
        val_loss, n100, r20, r50 = evaluate(vad_data_tr, vad_data_te)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:4.2f}s | valid loss {:4.2f} | '
              'n100 {:5.3f} | r20 {:5.3f} | r50 {:5.3f}'.format(
                epoch, time.time() - epoch_start_time, val_loss,
                n100, r20, r50))
        print('-' * 89)

        n_iter = epoch * len(range(0, N, args.batch_size))
        writer.add_scalars('data/loss', {'valid': val_loss}, n_iter)
        writer.add_scalar('data/n100', n100, n_iter)
        writer.add_scalar('data/r20', r20, n_iter)
        writer.add_scalar('data/r50', r50, n_iter)

        # Save the model if the n100 is the best we've seen so far.
        if n100 > best_n100:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_n100 = n100

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss, n100, r20, r50 = evaluate(test_data_tr, test_data_te)
print('=' * 89)
print('| End of training | test loss {:4.5f} | n100 {:4.5f} | r20 {:4.5f} | '
      'r50 {:4.5f}'.format(test_loss, n100, r20, r50))
print('=' * 89)