import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tensorboardX import SummaryWriter
from scipy import sparse
import models_mix_nips
import models_mix
import dataloader
import metric
import os
import pickle
import copy
import logging
# from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoders for Collaborative Filtering')
parser.add_argument('--data', type=str, default='data/amazon',
                    help='Movielens-20m dataset location')
# parser.add_argument('--lr', type=float, default=1e-3,
#                     help='initial learning rate')
# parser.add_argument('--wd', type=float, default=0.001,
#                     help='weight decay coefficient')
# parser.add_argument('--batch_size', type=int, default=2,
#                     help='batch size')
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
parser.add_argument('--save', type=str, default='train_mix_logs',
                    help='path to save the final model')
# parser.add_argument('--dfac1', type=int, default=100,
#                     help='Dimension of each facet.')
# parser.add_argument('--dfac2', type=int, default=100,
#                     help='Dimension of each facet.')
# parser.add_argument('--drop', type=float, default=0.5,
#                     help='Keep probability for dropout, in (0,1].')

args = parser.parse_args()
# args = parser.parse_known_args()
# args = args[0]
# parameters = {'lr': [0.001, 0.01, 0.1], 'wd': [0.001, 0.01, 0.1], 'batch_size': [100, 500, 1000],
#               'dfac1': [100, 400, 800], 'dfac2': [100, 400, 800], 'drop': [0, 0.4, 0.6, 0.8],
#               'std': [0.025, 0.075, 0.1]}
parameters = {'lr': [0.1], 'wd': [0.01, 0.1], 'batch_size': [100],
              'dfac1': [100], 'dfac2': [100], 'drop': [0.5], 'std': [0.075]}
# parameters = {'lr': [0.001], 'wd': [0.001], 'batch_size': [100],
#               'dfac1': [100], 'dfac2': [100], 'drop': [0.5]}

# Set the random seed manually for reproductibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

logging.basicConfig(filename=args.save, filemode='a', format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S', level=logging.DEBUG)
logger = logging.getLogger()
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

    for batch_idx, start_idx in enumerate(range(0, N, batch_size)):
        end_idx = min(start_idx + batch_size, N)

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
        recon_batch_1, mu_1, logvar_1, recon_title_1 = model(data_tensor, title_tensor)
        recon_batch_2, mu_2, logvar_2, recon_title_2 = model(data_tensor, None)
        recon_batch_3, mu_3, logvar_3, recon_title_3 = model(None, title_tensor)

        # loss = criterion(recon_batch, data_tensor, mu, logvar, recon_title, title_tensor, anneal)
        joint_loss = criterion(recon_batch_1, data_tensor, mu_1, logvar_1, recon_title_1, title_tensor, anneal)
        seq_loss = criterion(recon_batch_2, data_tensor, mu_2, logvar_2, recon_title_2, title_tensor, anneal)
        title_loss = criterion(recon_batch_3, data_tensor, mu_3, logvar_3, recon_title_3, title_tensor, anneal)
        loss = joint_loss + seq_loss + title_loss

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        update_count += 1

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | '
                  'loss {:4.2f}'.format(epoch, batch_idx, len(range(0, N, batch_size)),
                                        elapsed * 1000 / args.log_interval,
                                        train_loss / args.log_interval))

            # Log loss to tensorboard
            n_iter = (epoch - 1) * len(range(0, N, batch_size)) + batch_idx
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

    n5_list = []
    n10_list = []
    r5_list = []
    r10_list = []
    # recon_input = []
    with torch.no_grad():
        for start_idx in range(0, e_N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            data = data_tr[e_idxlist[start_idx:end_idx]]
            heldout_data = data_te[e_idxlist[start_idx:end_idx]]

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
            recon_batch_1, mu_1, logvar_1, recon_title_1 = model(data_tensor, title_tensor)
            recon_batch_2, mu_2, logvar_2, recon_title_2 = model(data_tensor, None)
            recon_batch_3, mu_3, logvar_3, recon_title_3 = model(None, title_tensor)

            # loss = criterion(recon_batch, data_tensor, mu, logvar, recon_title, title_tensor, anneal)
            joint_loss = criterion(recon_batch_1, data_tensor, mu_1, logvar_1, recon_title_1, title_tensor, anneal)
            seq_loss = criterion(recon_batch_2, data_tensor, mu_2, logvar_2, recon_title_2, title_tensor, anneal)
            title_loss = criterion(recon_batch_3, data_tensor, mu_3, logvar_3, recon_title_3, title_tensor, anneal)
            loss = title_loss + joint_loss + seq_loss

            total_loss += loss.item()

            # Exclude examples from training set
            recon_batch = (recon_batch_3 + recon_batch_1 + recon_batch_2)/3
            recon_batch = recon_batch.cpu().numpy()
            recon_batch[data.nonzero()] = -np.inf

            n20 = metric.NDCG_binary_at_k_batch(recon_batch, heldout_data, 20)
            n50 = metric.NDCG_binary_at_k_batch(recon_batch, heldout_data, 50)
            r20 = metric.Recall_at_k_batch(recon_batch, heldout_data, 20)
            r50 = metric.Recall_at_k_batch(recon_batch, heldout_data, 50)

            n5 = metric.NDCG_binary_at_k_batch(recon_batch, heldout_data, 5)
            n10 = metric.NDCG_binary_at_k_batch(recon_batch, heldout_data, 10)
            r5 = metric.Recall_at_k_batch(recon_batch, heldout_data, 5)
            r10 = metric.Recall_at_k_batch(recon_batch, heldout_data, 10)

            n20_list.append(n20)
            n50_list.append(n50)
            r20_list.append(r20)
            r50_list.append(r50)

            n5_list.append(n5)
            n10_list.append(n10)
            r5_list.append(r5)
            r10_list.append(r10)

            # recon_input.append(recon_batch.tolist())

    total_loss /= len(range(0, e_N, batch_size))
    n20_list = np.concatenate(n20_list) # concat into an array
    n50_list = np.concatenate(n50_list)
    r20_list = np.concatenate(r20_list)
    r50_list = np.concatenate(r50_list)

    n5_list = np.concatenate(n5_list)
    n10_list = np.concatenate(n10_list)
    r5_list = np.concatenate(r5_list)
    r10_list = np.concatenate(r10_list)

    return total_loss, np.mean(n5_list), np.mean(n10_list), np.mean(n20_list), np.mean(n50_list), \
           np.mean(r5_list), np.mean(r10_list), np.mean(r20_list), np.mean(r50_list)


# At any point you can hit Ctrl + C to break out of training early.
if __name__ == '__main__':
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

    for lr in parameters['lr']:
        for wd in parameters['wd']:
            for batch_size in parameters['batch_size']:
                for dfac1 in parameters['dfac1']:
                    for dfac2 in parameters['dfac2']:
                        if dfac2 < dfac1:
                            continue
                        for drop in parameters['drop']:
                            for std in parameters['std']:
                                best_r50 = -np.inf
                                best_model = None
                                best_params = {}
                                update_count = 0
                                num_batches = int(np.ceil(float(N) / batch_size))
                                total_anneal_steps = 5 * num_batches

                                best_n5 = -np.inf
                                best_n10 = -np.inf
                                best_n20 = -np.inf
                                best_n50 = -np.inf
                                best_r5 = -np.inf
                                best_r10 = -np.inf
                                best_r20 = -np.inf
                                best_val = -np.inf
                                ###############################################################################
                                # Build the model
                                ###############################################################################
                                p_dims = [dfac1, dfac2, n_items]
                                model = models_mix_nips.MultiVAE(len(vocab2index), p_dims, None, drop, std).to(device)
                                # model = models_mix.MultiVAE(len(vocab2index), p_dims, None, drop).to(device)
                                optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)
                                criterion = models_mix_nips.loss_function

                                # logger.info('=' * 89)
                                # logger.info('|lr: {:4.5f}|wd: {:4.5f}|batch size: {:d}|dfac1: {:d}|dfac2: {:d}|'
                                #             'drop: {:4.2f}|std: {:4.5f}'.format(lr, wd, batch_size, dfac1, dfac2, drop, std))
                                # logger.info('=' * 89)
                                # print('=' * 89)
                                # print('|lr: {:4.5f}|wd: {:4.5f}|batch size: {:d}|dfac1: {:d}|dfac2: {:d}|'
                                #       'drop: {:4.2f}|std: {:4.5f}'.format(lr, wd, batch_size, dfac1, dfac2, drop, std))
                                # print('=' * 89)
                                for epoch in range(1, args.epochs + 1):
                                    epoch_start_time = time.time()
                                    # train
                                    train()
                                    # evaluate
                                    val_loss, n5, n10, n20, n50, r5, r10, r20, r50 = evaluate(vad_data_tr, vad_data_te)
                                    # print('epoch: {:3d} | time: {:4.5f}'.format(epoch, time.time() - epoch_start_time))
                                    # logger.info(epoch)
                                    logger.info('-' * 89)
                                    logger.info('|end of epoch {:3d}|time: {:4.2f}s|valid loss {:4.2f}|n5 {:5.5f}|n10 {:5.5f}|n20 {:5.5f}|'
                                                'n50 {:5.5f}|r5 {:5.5f}|r10 {:5.5f}|r20 {:5.5f}|r50 {:5.5f}'.format(
                                                epoch, time.time() - epoch_start_time, val_loss, n5, n10, n20, n50, r5, r10, r20, r50))
                                    # logger.info('-' * 89)
                                    # print('-' * 89)
                                    print(
                                        '|end of epoch {:3d}|time: {:4.2f}s|valid loss {:4.2f}|n5 {:5.5f}|n10 {:5.5f}|n20 {:5.5f}|'
                                        'n50 {:5.5f}|r5 {:5.5f}|r10 {:5.5f}|r20 {:5.5f}|r50 {:5.5f}'.format(
                                            epoch, time.time() - epoch_start_time, val_loss, n5, n10, n20, n50, r5, r10,
                                            r20, r50))
                                    # print('-' * 89)

                                    n_iter = epoch * len(range(0, N, batch_size))
                                    writer.add_scalars('data/loss', {'valid': val_loss}, n_iter)
                                    writer.add_scalar('data/n20', n20, n_iter)
                                    writer.add_scalar('data/n50', n50, n_iter)
                                    writer.add_scalar('data/r20', r20, n_iter)
                                    writer.add_scalar('data/r50', r50, n_iter)

                                    # Save the model if the n100 is the best we've seen so far.
                                    if r50 > best_r50:
                                        # with open(args.save, 'wb') as f:
                                        #     torch.save(model, f)
                                        best_model = copy.deepcopy(model)
                                        best_params['lr'] = lr
                                        best_params['wd'] = wd
                                        best_params['batch_size'] = batch_size
                                        best_params['dfac1'] = dfac1
                                        best_params['dfac2'] = dfac2
                                        best_params['drop'] = drop
                                        best_params['std'] = std
                                        best_params['epochs'] = epoch

                                        best_n20 = n20
                                        best_n50 = n50
                                        best_r20 = r20
                                        best_r50 = r50
                                        best_val = val_loss

                                        best_n5 = n5
                                        best_n10 = n10
                                        best_r5 = r5
                                        best_r10 = r10

                                # Load the best saved model.
                                # with open(args.save, 'rb') as f:
                                #     model = torch.load(f)
                                # model = copy.deepcopy(best_model)
                                # parameter settings
                                print('=' * 89)
                                print('|lr: {:4.5f}|wd: {:4.5f}|batch size: {:5f}|dfac1: {:5f}|dfac2: {:5f}|'
                                      'drop: {:4.2f}|std: {:4.5f}|epoch: {:4.5f}'.
                                      format(best_params['lr'], best_params['wd'], best_params['batch_size'],
                                             best_params['dfac1'], best_params['dfac2'], best_params['drop'],
                                             best_params['std'], best_params['epoch']))
                                # # print('=' * 89)
                                # # Run on test data.
                                # test_loss, n5, n10, n20, n50, r5, r10, r20, r50 = evaluate(vad_data_tr, vad_data_te)
                                # # print('=' * 89)
                                # print('|End of training|test loss {:4.5f}|n5 {:4.5f}|n10 {:4.5f}|n20 {:4.5f}|n50 {:4.5f}|r5 {:4.5f}|'
                                #     'r10 {:4.5f}|r20 {:4.5f}|r50 {:4.5f}'.format(test_loss, n5, n10, n20, n50, r5, r10, r20, r50))
                                # # print('=' * 89)
                                #
                                logger.info('=' * 89)
                                logger.info(best_params)
                                # # logger.info('=' * 89)
                                # # logger.info('=' * 89)
                                # logger.info('|End of training|test loss {:4.5f}|n5 {:4.5f}|n10 {:4.5f}|n20 {:4.5f}|n50 {:4.5f}|r5 {:4.5f}|'
                                #       'r10 {:4.5f}|r20 {:4.5f}|r50 {:4.5f}'.format(test_loss, n5, n10, n20, n50, r5, r10, r20, r50))
                                # # logger.info('=' * 89)

                                print(
                                    '|End of training|test loss {:4.5f}|n5 {:4.5f}|n10 {:4.5f}|n20 {:4.5f}|n50 {:4.5f}|r5 {:4.5f}|'
                                    'r10 {:4.5f}|r20 {:4.5f}|r50 {:4.5f}'.format(best_val, best_n5, best_n10, best_n20,
                                                                                 best_n50, best_r5, best_r10, best_r20,
                                                                                 best_r50))
                                logger.info(
                                    '|End of training|test loss {:4.5f}|n5 {:4.5f}|n10 {:4.5f}|n20 {:4.5f}|n50 {:4.5f}|r5 {:4.5f}|'
                                    'r10 {:4.5f}|r20 {:4.5f}|r50 {:4.5f}'.format(best_val, best_n5, best_n10, best_n20,
                                                                                 best_n50, best_r5, best_r10, best_r20,
                                                                                 best_r50))
