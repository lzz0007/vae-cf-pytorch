import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tensorboardX import SummaryWriter
from scipy import sparse
import models_dis
import dataloader
import metric

import torch.nn.functional as F
import logging

parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoders for Collaborative Filtering')
parser.add_argument('--data', type=str, default='data/amazon',
                    help='Movielens-20m dataset location')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--wd', type=float, default=0.001,
                    help='weight decay coefficient')
parser.add_argument('--batch_size', type=int, default=100,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=1,
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
parser.add_argument('--kfac', type=int, default=1,
                    help='Number of facets (macro concepts).')
parser.add_argument('--dfac', type=int, default=100,
                    help='Dimension of each facet.')
parser.add_argument('--nogb', action='store_true', default=False,
                    help='Disable Gumbel-Softmax sampling.')
args = parser.parse_args()

# Set the random seed manually for reproductibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda:2" if args.cuda else "cpu")

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
loader = dataloader.DataLoader(args.data)

n_items = loader.load_n_items()
train_data = loader.load_data('train')
vad_data_tr, vad_data_te = loader.load_data('validation')
# test_data_tr, test_data_te, _ = loader.load_data('test')

N = train_data.shape[0]
idxlist = list(range(N))

num_batches = int(np.ceil(float(N) / args.batch_size))
total_anneal_steps = 5 * num_batches
###############################################################################
# Build the model
###############################################################################

p_dims = [args.dfac, 800, n_items]
model = models_dis.MultiVAE(p_dims, q_dims=None, dropout=args.keep, tau=args.tau,
                            std=args.std, kfac=args.kfac, nogb=args.nogb).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
criterion = models_dis.loss_function

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
    n5_list = []
    r20_list = []
    r50_list = []

    n10_list, n20_list, n30_list, n40_list, n50_list, n60_list, n70_list, n80_list, n90_list = ([] for i in range(9))
    r10_list, r30_list, r40_list, r60_list, r70_list, r80_list, r90_list, r5_list = ([] for i in range(8))

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

            n5 = metric.NDCG_binary_at_k_batch(recon_batch, heldout_data, 5)
            r20 = metric.Recall_at_k_batch(recon_batch, heldout_data, 20)
            r50 = metric.Recall_at_k_batch(recon_batch, heldout_data, 50)

            n5_list.append(n5)
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
            r5 = metric.Recall_at_k_batch(recon_batch, heldout_data, 5)

            r10_list.append(r10)
            r30_list.append(r30)
            r40_list.append(r40)
            r60_list.append(r60)
            r70_list.append(r70)
            r80_list.append(r80)
            r90_list.append(r90)
            r5_list.append(r5)

    total_loss /= len(range(0, e_N, args.batch_size))
    n5_list = np.concatenate(n5_list)
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
    r5_list = np.concatenate(r5_list)

    return total_loss, np.mean(n10_list), np.mean(n20_list), np.mean(n30_list), np.mean(n40_list), np.mean(n50_list), \
           np.mean(n60_list), np.mean(n70_list), np.mean(n80_list), np.mean(n90_list), np.mean(n5_list), \
           np.mean(r10_list), np.mean(r20_list), np.mean(r30_list), np.mean(r40_list), np.mean(r50_list), \
           np.mean(r60_list), np.mean(r70_list), np.mean(r80_list), np.mean(r90_list), np.mean(r5_list),


best_n20 = -np.inf
best_n50 = -np.inf
best_r20 = -np.inf
best_r50 = -np.inf
best_n5 = -np.inf
best_n10 = -np.inf
best_r5 = -np.inf
best_r10 = -np.inf
best_val = -np.inf
update_count = 0

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        # train
        train()
        # evaluate
        # val_loss, n100, r20, r50 = evaluate(vad_data_tr, vad_data_te)
        val_loss, n10, n20, n30, n40, n50, n60, n70, n80, n90, n5, \
        r10, r20, r30, r40, r50, r60, r70, r80, r90, r5 = evaluate(vad_data_tr, vad_data_te)

        print('-' * 89)
        print('|End of epoch {:3d}|val loss {:4.5f}|time: {:4.2f}s|n5 {:4.5f}|n10 {:4.5f}|n20 {:4.5f}|n50 {:4.5f}|r5 {:4.5f}|'
              'r10 {:4.5f}|r20 {:4.5f}|r50 {:4.5f}'.format(epoch, val_loss, time.time() - epoch_start_time,
                                                           n5, n10, n20, n50, r5, r10, r20, r50))
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

            best_n5 = n5
            best_n10 = n10
            best_r5 = r5
            best_r10 = r10

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# # Load the best saved model.
# with open(args.save, 'rb') as f:
#     model = torch.load(f)
#
# # Run on test data.
# # test_loss, n100, r20, r50 = evaluate(test_data_tr, test_data_te)
# test_loss, n10, n20, n30, n40, n50, n60, n70, n80, n90, n100, r10, r20, r30, r40, \
# r50, r60, r70, r80, r90, r100 = evaluate(test_data_tr, test_data_te)
#
# print('=' * 89)
# print('| End of training | test loss {:4.5f} | n100 {:4.5f} | r20 {:4.5f} | '
#       'r50 {:4.5f}'.format(test_loss, n100, r20, r50))
# print('=' * 89)
# logger.info('|script: {}|kfac: {}|epochs: {:3d}|lr: {:5.5f}'.format('main_disentangled.py', args.kfac, args.epochs, args.lr))
# logger.info('|{:4.2f}| '
#             '{:5.5f}|{:5.5f}|{:5.5f}|{:5.5f}|{:5.5f}|{:5.5f}|{:5.5f}| '
#             '{:5.5f}|{:5.5f}|{:5.5f}|{:5.5f}|{:5.5f}|{:5.5f}'
#             '|{:5.5f}|{:5.5f}|{:5.5f}|{:5.5f}|{:5.5f}|{:5.5f}|{:5.5f}'.format(
#     test_loss,
#     n10, n20, n30, n40, n50, n60, n70, n80, n90, n100,
#     r10, r20, r30, r40, r50, r60, r70, r80, r90, r100))

print('=' * 89)
print('|End of training|test loss {:4.5f}|n5 {:4.5f}|n10 {:4.5f}|n20 {:4.5f}|n50 {:4.5f}|r5 {:4.5f}|'
      'r10 {:4.5f}|r20 {:4.5f}|r50 {:4.5f}'.format(best_val, best_n5, best_n10, best_n20, best_n50,
                                                   best_r5, best_r10, best_r20, best_r50))
print('=' * 89)