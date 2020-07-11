import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tensorboardX import SummaryWriter
from scipy import sparse
import models_mvae
import data
import metric

import torch.nn.functional as F
import pandas as pd
import os
import pickle
from sklearn.cluster import KMeans
from tqdm import tqdm
import logging

# torch.backends.cudnn.enabled = False
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoders for Collaborative Filtering')
parser.add_argument('--data', type=str, default='data/amazon',
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
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--keep', type=float, default=0.5,
                    help='Keep probability for dropout, in (0,1].')
parser.add_argument('--tau', type=float, default=0.1,
                    help='Temperature of sigmoid/softmax, in (0,oo).')
parser.add_argument('--std', type=float, default=0.075,
                    help='Standard deviation of the Gaussian prior.')
parser.add_argument('--kfac', type=int, default=3,
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

logging.basicConfig(filename='train_logs_paper',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
logger = logging.getLogger()

###############################################################################
# Load data
###############################################################################
# args.data = 'alishop-7c'
loader = data.DataLoader(args.data)

n_items = loader.load_n_items()
train_data, train_buy = loader.load_data('train')

vad_data_tr, vad_data_te, vad_buy = loader.load_data('validation')
test_data_tr, test_data_te, test_buy = loader.load_data('test')

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

item_mat = dataset['meta_mat']
vocab2index = dataset['vocab2index']
cat2index = dataset['cat2index']
item2index = dataset['item2index']
category_id = np.array(dataset['category_id'])
item_title = dataset['meta_titles']

max_item = max([len(v) for k,v in train_buy.items()])
max_word = 100
# embeddings = torch.load('embeddings.pt')

# kmeans = KMeans(n_clusters=args.kfac, random_state=args.seed).fit(embeddings)
# init_kmeans = torch.FloatTensor(kmeans.cluster_centers_)

# titles = torch.from_numpy(embeddings).float().contiguous().to(device)

# define parameters for titles
embedding_dim = 100
hidden_dim = 100

###############################################################################
# load image
###############################################################################
# img_features_filtered = torch.load('images_filtered.pt')
# img_features_filtered = torch.from_numpy(img_features_filtered).float().contiguous().to(device)

###############################################################################
# Build the model
###############################################################################

p_dims = [args.dfac, args.dfac, n_items]

model = models_mvae.MultiVAE(p_dims, q_dims=None, dropout=args.keep, tau=args.tau, std=args.std, kfac=args.kfac,
                             nogb=args.nogb).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
criterion = models_mvae.loss_function

###############################################################################
# Training code
###############################################################################

# TensorboardX Writer

writer = SummaryWriter()

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

        # seq
        data = train_data[idxlist[start_idx:end_idx]]
        data = naive_sparse2tensor(data).to(device)

        # title
        data_title = []
        for i in range(start_idx, end_idx):
            if i in train_buy:
                data_title.append(train_buy[i])
            else:
                data_title.append([])
        # data_title = [data_buy[i] for i in range(start_idx, end_idx)]
        data_title_word = []
        for i in data_title:
            tmp = []
            if not i:
                tmp.append([0] * 100)
            else:
                for j in i:
                    tmp.append(item_title[j].tolist())
            data_title_word.append(tmp)
        # data_title = [train_buy[i] for i in range(start_idx, end_idx)]
        data_title_word = []
        for i in data_title:
            tmp = []
            for j in i:
                tmp.append(item_title[j].tolist())
            data_title_word.append(tmp)
        # max_item = max([len(i) for i in data_title])
        data_title_mask = np.zeros((len(data_title), max_item, max_word), dtype=int)
        for i, c in enumerate(data_title_word):
            data_title_mask[i,:len(c),:] = c
        data_title_mask = torch.LongTensor(data_title_mask).to(device)

        # anneal
        if total_anneal_steps > 0:
            anneal = min(args.anneal_cap,
                         1. * update_count / total_anneal_steps)
        else:
            anneal = args.anneal_cap

        # model
        optimizer.zero_grad()
        # recon_batch, mu, logvar = model(data)
        recon_batch, std_list = model(data, data_title_mask)
        # print(start_idx)
        loss = criterion(data, std_list, recon_batch, anneal)
        # print(loss)
        loss.backward()
        train_loss += loss.item()
        # we use gradient clipping to avoid exploding gradients
        # torch.nn.utils.clip_grad_norm(model.parameters(), 5)
        optimizer.step()

        update_count += 1

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | '
                  'loss {:4.2f}'.format(
                        epoch, batch_idx, len(range(0, N, args.batch_size)),
                        elapsed * 1000 / args.log_interval,
                        train_loss / args.log_interval))
            # for n,p in model.named_parameters():
            #     if n == 'cores_title.weight' or n == 'cores.weight':
            #         print(n)
            #         print(p)
            print(model.cores.weight)
            print(model.cores_title.weight)
            # Log loss to tensorboard
            n_iter = (epoch - 1) * len(range(0, N, args.batch_size)) + batch_idx
            writer.add_scalars('data/loss', {'train': train_loss / args.log_interval}, n_iter)

            start_time = time.time()
            train_loss = 0.0


def evaluate(data_tr, data_te, data_buy):
    # data_tr = vad_data_tr
    # data_te = vad_data_te
    # data_buy = vad_buy

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

    n10_list, n20_list, n30_list, n40_list, n50_list, n60_list, n70_list, n80_list, n90_list = ([] for i in range(9))
    r10_list, r30_list, r40_list, r60_list, r70_list, r80_list, r90_list, r100_list = ([] for i in range(8))

    with torch.no_grad():
        for start_idx in range(0, e_N, args.batch_size):
            end_idx = min(start_idx + args.batch_size, N)

            # seq
            data = data_tr[e_idxlist[start_idx:end_idx]]
            heldout_data = data_te[e_idxlist[start_idx:end_idx]]
            data_tensor = naive_sparse2tensor(data).to(device)

            # title
            data_title = []
            for i in range(start_idx, end_idx):
                if i in data_buy:
                    data_title.append(data_buy[i])
                else:
                    data_title.append([])
            # data_title = [data_buy[i] for i in range(start_idx, end_idx)]
            data_title_word = []
            for i in data_title:
                tmp = []
                if not i:
                    tmp.append([0]*100)
                else:
                    for j in i:
                        tmp.append(item_title[j].tolist())
                data_title_word.append(tmp)
            # max_item = max([len(i) for i in data_title])
            data_title_mask = np.zeros((len(data_title), max_item, max_word), dtype=int)
            for i, c in enumerate(data_title_word):
                data_title_mask[i, :len(c), :] = c
            data_title_mask = torch.LongTensor(data_title_mask).to(device)

            if total_anneal_steps > 0:
                anneal = min(args.anneal_cap,
                             1. * update_count / total_anneal_steps)
            else:
                anneal = args.anneal_cap

            # recon_batch, mu, logvar = model(data_tensor)
            recon_batch, std_list = model(data_tensor, data_title_mask)
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

            n10 = metric.NDCG_binary_at_k_batch(recon_batch, heldout_data, 10)
            n20 = metric.NDCG_binary_at_k_batch(recon_batch, heldout_data, 20)
            n30 = metric.NDCG_binary_at_k_batch(recon_batch, heldout_data, 30)
            n40 = metric.NDCG_binary_at_k_batch(recon_batch, heldout_data, 40)
            n50 = metric.NDCG_binary_at_k_batch(recon_batch, heldout_data, 50)
            n60 = metric.NDCG_binary_at_k_batch(recon_batch, heldout_data, 60)
            n70 = metric.NDCG_binary_at_k_batch(recon_batch, heldout_data, 70)
            n80 = metric.NDCG_binary_at_k_batch(recon_batch, heldout_data, 80)
            n90 = metric.NDCG_binary_at_k_batch(recon_batch, heldout_data, 90)

            n10_list.append(n10)
            n20_list.append(n20)
            n30_list.append(n30)
            n40_list.append(n40)
            n50_list.append(n50)
            n60_list.append(n60)
            n70_list.append(n70)
            n80_list.append(n80)
            n90_list.append(n90)

            r10 = metric.Recall_at_k_batch(recon_batch, heldout_data, 10)
            r30 = metric.Recall_at_k_batch(recon_batch, heldout_data, 30)
            r40 = metric.Recall_at_k_batch(recon_batch, heldout_data, 40)
            r60 = metric.Recall_at_k_batch(recon_batch, heldout_data, 60)
            r70 = metric.Recall_at_k_batch(recon_batch, heldout_data, 70)
            r80 = metric.Recall_at_k_batch(recon_batch, heldout_data, 80)
            r90 = metric.Recall_at_k_batch(recon_batch, heldout_data, 90)
            r100 = metric.Recall_at_k_batch(recon_batch, heldout_data, 100)

            r10_list.append(r10)
            r30_list.append(r30)
            r40_list.append(r40)
            r60_list.append(r60)
            r70_list.append(r70)
            r80_list.append(r80)
            r90_list.append(r90)
            r100_list.append(r100)

    total_loss /= len(range(0, e_N, args.batch_size))
    n100_list = np.concatenate(n100_list)
    r20_list = np.concatenate(r20_list)
    r50_list = np.concatenate(r50_list)

    n10_list = np.concatenate(n10_list)
    n20_list = np.concatenate(n20_list)
    n30_list = np.concatenate(n30_list)
    n40_list = np.concatenate(n40_list)
    n50_list = np.concatenate(n50_list)
    n60_list = np.concatenate(n60_list)
    n70_list = np.concatenate(n70_list)
    n80_list = np.concatenate(n80_list)
    n90_list = np.concatenate(n90_list)

    r10_list = np.concatenate(r10_list)
    r30_list = np.concatenate(r30_list)
    r40_list = np.concatenate(r40_list)
    r60_list = np.concatenate(r60_list)
    r70_list = np.concatenate(r70_list)
    r80_list = np.concatenate(r80_list)
    r90_list = np.concatenate(r90_list)
    r100_list = np.concatenate(r100_list)

    # return total_loss, np.mean(n100_list), np.mean(r20_list), np.mean(r50_list)
    return total_loss, np.mean(n10_list), np.mean(n20_list), np.mean(n30_list), np.mean(n40_list), np.mean(n50_list), \
           np.mean(n60_list), np.mean(n70_list), np.mean(n80_list), np.mean(n90_list), np.mean(n100_list), \
           np.mean(r10_list), np.mean(r20_list), np.mean(r30_list), np.mean(r40_list), np.mean(r50_list), \
           np.mean(r60_list), np.mean(r70_list), np.mean(r80_list), np.mean(r90_list), np.mean(r100_list),

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
        # val_loss, n100, r20, r50 = evaluate(vad_data_tr, vad_data_te)
        val_loss, n10, n20, n30, n40, n50, n60, n70, n80, n90, n100, \
        r10, r20, r30, r40, r50, r60, r70, r80, r90, r100 = evaluate(vad_data_tr, vad_data_te, vad_buy)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:4.2f}s | valid loss {:4.2f} | '
              'n100 {:5.3f} | r20 {:5.3f} | r50 {:5.3f}'.format(
                epoch, time.time() - epoch_start_time, val_loss,
                n100, r20, r50))
        print('-' * 89)
        logger.info('| end of epoch {:3d} | time: {:4.2f}s | valid loss {:4.2f} | '
              'n10 {:5.5f} | n20 {:5.5f} | n30 {:5.5f}| n40 {:5.5f} | n50 {:5.5f}| n60 {:5.5f} | n70 {:5.5f}| '
                    'n80 {:5.5f} | n90 {:5.5f}| n100 {:5.5f} | r10 {:5.5f} | r20 {:5.5f} | r30 {:5.5f}'
                    '| r40 {:5.5f}| r50 {:5.5f}| r60 {:5.5f}| r70 {:5.5f}| r80 {:5.5f}| r90 {:5.5f}| r100 {:5.5f}'.format(
                epoch, time.time() - epoch_start_time, val_loss,
                 n10, n20, n30, n40, n50, n60, n70, n80, n90, n100,
                    r10, r20, r30, r40, r50, r60, r70, r80, r90, r100))

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
# test_loss, n100, r20, r50 = evaluate(test_data_tr, test_data_te)
test_loss, n10, n20, n30, n40, n50, n60, n70, n80, n90, n100, \
        r10, r20, r30, r40, r50, r60, r70, r80, r90, r100 = evaluate(test_data_tr, test_data_te, test_buy)
print('=' * 89)
print('| End of training | test loss {:4.5f} | n100 {:4.5f} | r20 {:4.5f} | '
      'r50 {:4.5f}'.format(test_loss, n100, r20, r50))
print('=' * 89)
logger.info('=' * 89)
logger.info('test loss')
logger.info('| test loss {:4.2f} | '
            'n10 {:5.5f} | n20 {:5.5f} | n30 {:5.5f}| n40 {:5.5f} | n50 {:5.5f}| n60 {:5.5f} | n70 {:5.5f}| '
            'n80 {:5.5f} | n90 {:5.5f}| n100 {:5.5f} | r10 {:5.5f} | r20 {:5.5f} | r30 {:5.5f}'
            '| r40 {:5.5f}| r50 {:5.5f}| r60 {:5.5f}| r70 {:5.5f}| r80 {:5.5f}| r90 {:5.5f}| r100 {:5.5f}'.format(
    test_loss,
    n10, n20, n30, n40, n50, n60, n70, n80, n90, n100,
    r10, r20, r30, r40, r50, r60, r70, r80, r90, r100))