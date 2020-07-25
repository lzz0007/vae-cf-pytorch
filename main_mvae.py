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
import logging
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter
from k_medoid import KMedoids
from torch.autograd import Variable
# torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoders for Collaborative Filtering')
parser.add_argument('--data', type=str, default='data/amazon',
                    help='Movielens-20m dataset location')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate')
parser.add_argument('--wd', type=float, default=0.001,
                    help='weight decay coefficient')
parser.add_argument('--batch_size', type=int, default=1,
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
parser.add_argument('--log-interval', type=int, default=80, metavar='N',
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
parser.add_argument('--mvae', action='store_true', default=False,
                    help='whether to run mvae.')
# args = parser.parse_args()
args = parser.parse_known_args()
args = args[0]
args.mvae = True
# Set the random seed manually for reproductibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda:1" if args.cuda else "cpu")

logging.basicConfig(filename='train_logs',
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
# kmeans for interaction data
###############################################################################
# Standardize the data to have a mean of ~0 and a variance of 1
X_std = StandardScaler().fit_transform(train_data.transpose().todense())
# Create a PCA instance: pca
pca = PCA(n_components=100, random_state=args.seed)
principalComponents = pca.fit_transform(X_std)
# Save components to a DataFrame
PCA_components = pd.DataFrame(principalComponents)

print('start doing Kmedoids for interaction')
kmedoids = KMedoids(n_clusters=6, random_state=args.seed).fit(PCA_components)
print(Counter(kmedoids.labels_).keys())
print(Counter(kmedoids.labels_).values())
kmedoids_center = kmedoids.cluster_centers_
labels = np.array(list(set(kmedoids.labels_)))
kmedoids_center = kmedoids_center[labels]
centers = torch.FloatTensor(kmedoids_center).to(device)
kfac = len(labels)
print('no of clusters:', kfac)

###############################################################################
# load item title
###############################################################################
with open(os.path.join(args.data, 'meta_encoded.pickle'), 'rb') as f:
    dataset = pickle.load(f)

item_mat = dataset['meta_mat']
vocab2index = dataset['vocab2index']
cat2index = dataset['cat2index']
item2index = dataset['item2index']
category_id = np.array(dataset['category_id'])
item_title = dataset['meta_titles']

max_item = max([len(v) for k, v in train_buy.items()])
max_word = max([len(i['words']) for i in item_title])

# embeddings = torch.load('embeddings.pt')

# load pretrained word embeddings
if args.mvae:
    # import codecs
    # all_word_embeds = {}
    # for i, line in enumerate(codecs.open('data/glove.6B.100d.txt', 'r', 'utf-8')):
    #     s = line.strip().split()
    #     if len(s) == args.dfac + 1:
    #         all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])
    #
    # # Intializing Word Embedding Matrix
    # word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(vocab2index), args.dfac))
    # for w in vocab2index:
    #     if w in all_word_embeds:
    #         word_embeds[vocab2index[w]] = all_word_embeds[w]
    #     elif w.lower() in all_word_embeds:
    #         word_embeds[vocab2index[w]] = all_word_embeds[w.lower()]
    #
    # print('Loaded %i pretrained embeddings.' % len(word_embeds))
    # del all_word_embeds
    #
    # title_emb = np.zeros((len(item_title), max_word, 100))
    # for k, v in item_title.items():
    #     for i, idx in enumerate(v):
    #         title_emb[k, i, :] = word_embeds[idx, :]

    title_emb = item_mat.todense()
    X_std = StandardScaler().fit_transform(title_emb)
    # Create a PCA instance: pca
    pca = PCA(n_components=100, random_state=args.seed)
    principalComponents = pca.fit_transform(X_std)
    # Save components to a DataFrame
    PCA_components = pd.DataFrame(principalComponents)

    print('start doing Kmedoids for title')
    kmedoids = KMedoids(n_clusters=kfac, random_state=args.seed).fit(PCA_components)
    print(Counter(kmedoids.labels_).keys())
    print(Counter(kmedoids.labels_).values())
    kmedoids_center = kmedoids.cluster_centers_
    labels = np.array(list(set(kmedoids.labels_)))
    kmedoids_center = kmedoids_center[labels]
    centers_title = torch.FloatTensor(kmedoids_center).to(device) # 6x100
else:
    centers_title = None

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
# Training code
###############################################################################

# TensorboardX Writer

writer = SummaryWriter()
decay_rate = 0.05


def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())


def set_rng_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def adjust_learning_rate(optimizer, lr):
    """
    shrink learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
        data_tensor = naive_sparse2tensor(data).to(device) # batchx13015

        # title
        # purchased_items = []
        # for i in range(data.shape[0]):
        #     purchased_items.append(list(data[i].nonzero()[1]))
        # purchased_items_title = [] # word index that each item has
        # for items in purchased_items:
        #     w = []
        #     for item in items:
        #         w.append(item_title[item]['words'])
        #     purchased_items_title.append(w)
        purchased_items = list(data.nonzero()[1])
        purchased_items_mask = np.zeros(max_item)
        purchased_items_mask[:len(purchased_items)] = purchased_items

        purchased_items_title = []
        for idx, item in enumerate(list(purchased_items_mask)):
            if np.sum(item) == 0 and idx >= len(purchased_items):
                purchased_items_title.append([0])
            else:
                purchased_items_title.append(item_title[int(item)]['words'])

        purchased_items_title_sorted = sorted(purchased_items_title, key=lambda p: len(p), reverse=True)
        purchased_items_mask = Variable(torch.LongTensor(purchased_items_mask).to(device))

        d = {}
        for i, ci in enumerate(purchased_items_title):
            for j, cj in enumerate(purchased_items_title_sorted):
                if ci == cj and not j in d and not i in d.values():
                    d[j] = i
                    continue
        title_length = [len(c) for c in purchased_items_title_sorted]
        title_maxl = max(title_length)
        title_mask = np.zeros((len(purchased_items_title_sorted), title_maxl), dtype='int')
        for i, c in enumerate(purchased_items_title_sorted):
            title_mask[i, :title_length[i]] = c
        title_mask = Variable(torch.LongTensor(title_mask).to(device))

        purchased_items_title_onehot = np.zeros((len(purchased_items_title), len(vocab2index)))
        for i, c in enumerate(purchased_items_title):
            purchased_items_title_onehot[i, c] = 1
        purchased_items_title_onehot = torch.FloatTensor(purchased_items_title_onehot).to(device)

        # anneal
        if total_anneal_steps > 0:
            anneal = min(args.anneal_cap,
                         1. * update_count / total_anneal_steps)
        else:
            anneal = args.anneal_cap

        # model
        optimizer.zero_grad()

        if args.mvae:
            recon_batch_1, std_list_1, items, recon_title, cluster_loss, std_list_title = \
                model(data_tensor, purchased_items_mask.to(device), title_mask.to(device), title_length, d)
            loss_joint = criterion(data_tensor, std_list_1, recon_batch_1, anneal,
                                   purchased_items_title_onehot, recon_title, std_list_title)
            # recon_batch_2, std_list_2, _, _, _ = model(data, None)
            # loss_seq = criterion(data, std_list_2, recon_batch_2, anneal, title=None, recon_title=None)
            # loss = loss_joint + loss_seq + cluster_loss
            loss = loss_joint + cluster_loss
        else:
            recon_batch_2, std_list_2, items, _ = model(data, None)
            loss_seq = criterion(data, std_list_2, recon_batch_2, anneal, title=None, recon_title=None)
            loss = loss_seq

        loss.backward()
        train_loss += loss.item()

        # we use gradient clipping to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
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
            #     if n == 'cores_title.weight':
            #         print(n)
            #         print(p)
            # tmp = model.cores_title.grad
            # print(tmp)
            # print(model.cores_title)
            # Log loss to tensorboard
            n_iter = (epoch - 1) * len(range(0, N, args.batch_size)) + batch_idx
            writer.add_scalars('data/loss', {'train': train_loss / args.log_interval}, n_iter)

            start_time = time.time()
            train_loss = 0.0

    # title word embeddings
    # titles_words = model.fc1_enc.weight
    # print(titles_words.shape)

    return items


def evaluate(data_tr, data_te, data_buy, centers, centers_title):
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
            purchased_items = list(data.nonzero()[1])
            purchased_items_mask = np.zeros(max_item)
            purchased_items_mask[:len(purchased_items)] = purchased_items

            purchased_items_title = []
            for idx, item in enumerate(list(purchased_items_mask)):
                if np.sum(item) == 0 and idx >= len(purchased_items):
                    purchased_items_title.append([0])
                else:
                    purchased_items_title.append(item_title[int(item)]['words'])

            purchased_items_title_sorted = sorted(purchased_items_title, key=lambda p: len(p), reverse=True)
            purchased_items_mask = Variable(torch.LongTensor(purchased_items_mask))

            d = {}
            for i, ci in enumerate(purchased_items_title):
                for j, cj in enumerate(purchased_items_title_sorted):
                    if ci == cj and not j in d and not i in d.values():
                        d[j] = i
                        continue
            title_length = [len(c) for c in purchased_items_title_sorted]
            title_maxl = max(title_length)
            title_mask = np.zeros((len(purchased_items_mask), title_maxl), dtype='int')
            for i, c in enumerate(purchased_items_title_sorted):
                title_mask[i, :title_length[i]] = c
            title_mask = Variable(torch.LongTensor(title_mask))

            purchased_items_title_onehot = np.zeros((len(purchased_items_title), len(vocab2index)))
            for i, c in enumerate(purchased_items_title):
                purchased_items_title_onehot[i, c] = 1
            purchased_items_title_onehot = torch.FloatTensor(purchased_items_title_onehot).to(device)

            if total_anneal_steps > 0:
                anneal = min(args.anneal_cap,
                             1. * update_count / total_anneal_steps)
            else:
                anneal = args.anneal_cap

            if args.mvae:
                recon_batch_1, std_list_1, items, recon_title, cluster_loss, std_list_title = \
                    model(data_tensor, purchased_items_mask.to(device), title_mask.to(device), title_length, d)
                loss_joint = criterion(data_tensor, std_list_1, recon_batch_1, anneal,
                                       purchased_items_title_onehot, recon_title, std_list_title)
                # recon_batch_2, std_list_2, _, _, _ = model(data_tensor, None)
                # loss_seq = criterion(data_tensor, std_list_2, recon_batch_2, anneal, title=None, recon_title=None)
                # loss = loss_joint + loss_seq + cluster_loss
                # recon_batch = (recon_batch_2 + recon_batch_1) / 2
                loss = loss_joint + cluster_loss
                recon_batch = recon_batch_1
            else:
                recon_batch_2, std_list_2, _, _ = model(data_tensor, None)
                loss_seq = criterion(data_tensor, std_list_2, recon_batch_2, anneal, title=None, recon_title=None)
                loss = loss_seq
                recon_batch = recon_batch_2

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
    ###############################################################################
    # Build the model
    ###############################################################################
    p_dims = [args.dfac, args.dfac, n_items]

    if args.mvae:
        model = models_mvae.MultiVAE(p_dims, q_dims=None, dropout=args.keep, tau=args.tau, std=args.std, kfac=kfac,
                                     nogb=args.nogb, pre_word_embeds=None, centers=centers, centers_title=None,
                                     char_to_ix=vocab2index, vocab_size=len(item2index)).to(device)
    else:
        model = models_mvae.MultiVAE(p_dims, q_dims=None, dropout=args.keep, tau=args.tau, std=args.std, kfac=kfac,
                                     nogb=args.nogb, pre_word_embeds=None, centers=centers, centers_title=None).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = models_mvae.loss_function
    # disentangled
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        # train
        items = train()

        # Performing decay on the learning rate
        if epoch % 100 == 0:
            adjust_learning_rate(optimizer, lr=args.lr / (1 + decay_rate * epoch / 20))

        # evaluate
        # val_loss, n100, r20, r50 = evaluate(vad_data_tr, vad_data_te)
        if args.mvae:
            val_loss, n10, n20, n30, n40, n50, n60, n70, n80, n90, n100, \
            r10, r20, r30, r40, r50, r60, r70, r80, r90, r100 = evaluate(vad_data_tr, vad_data_te, vad_buy,
                                                                         centers, centers_title)
        else:
            val_loss, n10, n20, n30, n40, n50, n60, n70, n80, n90, n100, \
            r10, r20, r30, r40, r50, r60, r70, r80, r90, r100 = evaluate(vad_data_tr, vad_data_te, vad_buy,
                                                                         centers, None)
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

        # kmeans centers
        # items_array = items.cpu().detach().numpy()
        # kmeans = KMeans(n_clusters=kfac).fit(items_array)
        # centers = torch.FloatTensor(kmeans.cluster_centers_).to(device)
        # predict = kmeans.predict(items_array)

        # # kmeans centers for title
        # if args.mvae:
        #     title_emb = np.zeros((len(item_title), max_word, 100))
        #     titles_words = titles_words.cpu().detach().numpy()
        #     for k, v in item_title.items():
        #         for i, idx in enumerate(v):
        #             title_emb[k, i, :] = titles_words[idx, :]
        #
        #     title_emb_reshape = title_emb.reshape(len(item_title), max_word * 100)
        #     kmeans_title = KMeans(n_clusters=kfac, random_state=args.seed).fit(title_emb_reshape)
        #     centers_title = torch.FloatTensor(kmeans_title.cluster_centers_).to(device)
        #     predict_title = kmeans_title.predict(title_emb_reshape)
    # print(Counter(predict).keys())
    # print(Counter(predict).values())
    # if args.mvae:
    #     print(Counter(predict_title).keys())
    #     print(Counter(predict_title).values())

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
# test_loss, n100, r20, r50 = evaluate(test_data_tr, test_data_te)
if args.mvae:
    test_loss, n10, n20, n30, n40, n50, n60, n70, n80, n90, n100, \
            r10, r20, r30, r40, r50, r60, r70, r80, r90, r100 = evaluate(test_data_tr, test_data_te, test_buy, centers, centers_title)
else:
    test_loss, n10, n20, n30, n40, n50, n60, n70, n80, n90, n100, \
            r10, r20, r30, r40, r50, r60, r70, r80, r90, r100 = evaluate(test_data_tr, test_data_te, test_buy, centers, None)

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