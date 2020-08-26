import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tensorboardX import SummaryWriter
from scipy import sparse
import models_mix
import dataloader
import metric
import os
import pickle

parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoders for Collaborative Filtering')
parser.add_argument('--data', type=str, default='data/amazon',
                    help='Movielens-20m dataset location')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--wd', type=float, default=0.001,
                    help='weight decay coefficient')
parser.add_argument('--batch_size', type=int, default=2,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--total_anneal_steps', type=int, default=200000,
                    help='the total number of gradient updates for annealing')
parser.add_argument('--anneal_cap', type=float, default=0.2,
                    help='largest annealing parameter')
parser.add_argument('--seed', type=int, default=98765,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
args = parser.parse_args()
# args = parser.parse_known_args()
# args = args[0]

# Set the random seed manually for reproductibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

###############################################################################
# Load data
###############################################################################
# args.data = 'ml-latest-small'
loader = dataloader.DataLoader(args.data)

n_items = loader.load_n_items()
train_data = loader.load_data('train')
vad_data_tr, vad_data_te = loader.load_data('validation')
# test_data_tr, test_data_te = loader.load_data('test')

N = train_data.shape[0]
idxlist = list(range(N))

num_batches = int(np.ceil(float(N) / args.batch_size))
total_anneal_steps = 5 * num_batches
###############################################################################
# load item title
###############################################################################
with open(os.path.join(args.data, 'meta_encoded.pickle'), 'rb') as f:
    dataset = pickle.load(f)

item_mat = dataset['meta_mat']
vocab2index = dataset['vocab2index']
item2index = dataset['item2index']
item_title = dataset['meta_titles']

max_item = int(max(train_data.sum(axis=1)))
max_word = max([len(i['words']) for i in item_title])

###############################################################################
# Build the model
###############################################################################

p_dims = [100, 100, n_items]
model = models_mix.MultiVAE(p_dims, q_dims=None, dropout=0.5, vocab_size=len(vocab2index), tot_items=len(item2index),
                        max_item=max_item).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=args.wd)
criterion = models_mix.loss_function_title

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

        # seq
        data = train_data[idxlist[start_idx:end_idx]]
        data_tensor = naive_sparse2tensor(data).to(device)

        # title
        purchased_items = []
        for i in range(data.shape[0]):
            purchased_items.append(list(data[i].nonzero()[1]))

        ulist = []
        ilist = []
        for i, items in enumerate(purchased_items):
            w = set()
            for item in items:
                w.update(item_title[int(item)]['words'])
            ulist.extend([i]*len(w))
            ilist.extend(w)
        title = sparse.csr_matrix((np.ones_like(ulist), (ulist, ilist)), dtype='float64',
                                  shape=(data.shape[0], len(vocab2index)))
        title_tensor = naive_sparse2tensor(title).to(device)

        # anneal
        if total_anneal_steps > 0:
            anneal = min(args.anneal_cap,
                         1. * update_count / total_anneal_steps)
        else:
            anneal = args.anneal_cap

        # train model
        optimizer.zero_grad()

        # recon_batch, mu, logvar, recon_title = model(data_tensor, title_tensor)
        recon_batch_1, logvar_1, recon_title_1 = model(data_tensor, title_tensor)
        # recon_batch_2, logvar_2, recon_title_2 = model(data_tensor, None)
        # _, mu_3, logvar_3, recon_title_3 = model(None, title_tensor)

        # loss = criterion(recon_batch, data_tensor, mu, logvar, recon_title, title_tensor, anneal)
        joint_loss = criterion(recon_batch_1, data_tensor, logvar_1, recon_title_1, title_tensor, anneal)
        # seq_loss = criterion(recon_batch_2, data_tensor, logvar_2, None, None, anneal)
        # title_loss = criterion(None, None, mu_3, logvar_3, recon_title_3, title_tensor, anneal)
        loss = joint_loss

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
    n20_list = []
    n50_list = []
    r20_list = []
    r50_list = []

    recon_input = []
    with torch.no_grad():
        for start_idx in range(0, e_N, args.batch_size):
            end_idx = min(start_idx + args.batch_size, N)
            data = data_tr[e_idxlist[start_idx:end_idx]]
            heldout_data = data_te[e_idxlist[start_idx:end_idx]]
            # print(heldout_data.sum(axis=1))
            data_tensor = naive_sparse2tensor(data).to(device)

            # title
            purchased_items = []
            for i in range(data.shape[0]):
                purchased_items.append(list(data[i].nonzero()[1]))

            ulist = []
            ilist = []
            for i, items in enumerate(purchased_items):
                w = set()
                for item in items:
                    w.update(item_title[int(item)]['words'])
                ulist.extend([i] * len(w))
                ilist.extend(w)
            title = sparse.csr_matrix((np.ones_like(ulist), (ulist, ilist)), dtype='float64',
                                      shape=(data.shape[0], len(vocab2index)))
            title_tensor = naive_sparse2tensor(title).to(device)

            if total_anneal_steps > 0:
                anneal = min(args.anneal_cap,
                             1. * update_count / total_anneal_steps)
            else:
                anneal = args.anneal_cap

            # recon_batch, mu, logvar, recon_title = model(data_tensor, title_tensor)
            recon_batch_1, logvar_1, recon_title_1 = model(data_tensor, title_tensor)
            # recon_batch_2, logvar_2, recon_title_2 = model(data_tensor, None)
            # _, mu_3, logvar_3, recon_title_3 = model(None, title_tensor)

            # loss = criterion(recon_batch, data_tensor, mu, logvar, recon_title, title_tensor, anneal)
            joint_loss = criterion(recon_batch_1, data_tensor, logvar_1, recon_title_1, title_tensor, anneal)
            # seq_loss = criterion(recon_batch_2, data_tensor, logvar_2, None, None, anneal)
            # title_loss = criterion(None, None, mu_3, logvar_3, recon_title_3, title_tensor, anneal)
            loss = joint_loss
            total_loss += loss.item()

            # Exclude examples from training set
            # recon_batch = 0.5*(recon_batch_1 + recon_batch_2)
            recon_batch = recon_batch_1.cpu().numpy()
            recon_batch[data.nonzero()] = -np.inf

            n20 = metric.NDCG_binary_at_k_batch(recon_batch, heldout_data, 20)
            n50 = metric.NDCG_binary_at_k_batch(recon_batch, heldout_data, 50)
            r20 = metric.Recall_at_k_batch(recon_batch, heldout_data, 20)
            r50 = metric.Recall_at_k_batch(recon_batch, heldout_data, 50)

            n20_list.append(n20)
            n50_list.append(n50)
            r20_list.append(r20)
            r50_list.append(r50)

            recon_input.append(recon_batch.tolist())

    total_loss /= len(range(0, e_N, args.batch_size))
    n20_list = np.concatenate(n20_list)
    n50_list = np.concatenate(n50_list)
    r20_list = np.concatenate(r20_list)
    r50_list = np.concatenate(r50_list)

    return total_loss, np.mean(n20_list), np.mean(n50_list), np.mean(r20_list), np.mean(r50_list)


best_n20 = -np.inf
best_n50 = -np.inf
best_r20 = -np.inf
best_r50 = -np.inf
best_val = -np.inf

update_count = 0

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        # train
        train()
        # evaluate
        val_loss, n20, n50, r20, r50 = evaluate(vad_data_tr, vad_data_te)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:4.2f}s | valid loss {:4.2f} | '
              'n20 {:5.3f} | n50 {:5.3f} | r20 {:5.3f} | r50 {:5.3f}'.format(
            epoch, time.time() - epoch_start_time, val_loss,
            n20, n50, r20, r50))
        print('-' * 89)

        n_iter = epoch * len(range(0, N, args.batch_size))
        writer.add_scalars('data/loss', {'valid': val_loss}, n_iter)
        writer.add_scalar('data/n20', n20, n_iter)
        writer.add_scalar('data/n50', n50, n_iter)
        writer.add_scalar('data/r20', r20, n_iter)
        writer.add_scalar('data/r50', r50, n_iter)

        # Save the model if the n100 is the best we've seen so far.
        if r50 > best_r50:
            # with open(args.save, 'wb') as f:
            #     torch.save(model, f)
            # best_model = copy.deepcopy(model)
            best_n20 = n20
            best_n50 = n50
            best_r20 = r20
            best_r50 = r50
            best_val = val_loss

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# # Load the best saved model.
# with open(args.save, 'rb') as f:
#     model = torch.load(f)

# # Run on test data.
# test_loss, n100, r20, r50, recon = evaluate(test_data_tr, test_data_te)
# print('=' * 89)
# print('| End of training | test loss {:4.5f} | n100 {:4.5f} | r20 {:4.5f} | '
#         'r50 {:4.5f}'.format(test_loss, n100, r20, r50))
# print('=' * 89)
#
# checks = {'recon': recon, 'test_tr': test_data_tr, 'test_te': test_data_te}
# with open('checks.pickle', 'wb') as f:
#     pickle.dump(checks, f, protocol=pickle.HIGHEST_PROTOCOL)

print('=' * 89)
print('| End of training | val loss {:4.5f} | n20 {:4.5f} | n50 {:4.5f} | r20 {:4.5f} | '
      'r50 {:4.5f}'.format(best_val, best_n20, best_n50, best_r20, best_r50))
print('=' * 89)