import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
import random

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class MultiVAE(nn.Module):
    """
    Container module for Multi-VAE.

    Multi-VAE : Variational Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, q_dims=None, dropout=0.5, tau=0.1, std=0.075, kfac=7, nogb=False, pre_word_embeds=None,
                 centers=None, centers_title=None, char_to_ix=None, vocab_size=None):
        super(MultiVAE, self).__init__()
        self.p_dims = p_dims
        if q_dims: # 13015x100x100
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]  # make the copy of the list in reverse order

        # Last dimension of q- network is for mean and variance
        # :-1 everything except last one. -1 last one
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]  # same as q_dims but last element*2
        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        temp_t_dims = [10200] + [self.q_dims[1]] + [self.q_dims[-1] * 2]
        self.t_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(temp_t_dims[:-1], temp_t_dims[1:])])
        temp_c_dims = [self.q_dims[0]+10200] + [self.q_dims[1]] + [self.q_dims[-1] * 2]
        self.c_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(temp_c_dims[:-1], temp_c_dims[1:])])

        dfac = self.q_dims[-1]
        self.kfac = kfac
        num_items = self.q_dims[0]
        self.cores = nn.Parameter(torch.empty(self.kfac, dfac))
        self.cores.data = centers
        self.items = nn.Parameter(torch.empty(num_items, dfac))
        # nn.init.xavier_normal_(self.cores.data)
        nn.init.xavier_normal_(self.items.data)
        self.tau = tau
        self.std = std  # Standard deviation of the Gaussian prior
        self.save_emb = False
        self.nogb = nogb

        self.dropout = nn.Dropout(dropout)
        self.init_weights()

        self.hidden_dim = 100

        # center for title
        self.cores_title = nn.Parameter(torch.empty(self.kfac, self.hidden_dim))
        # self.cores_title.data = centers_title
        nn.init.xavier_normal_(self.cores_title.data)

        # # for title encoder
        # # self.fc1_enc = nn.Embedding(17424, hidden_dim)
        # self.fc1_enc = nn.Linear(17424, hidden_dim)
        # if pre_word_embeds is not None:
        #     self.fc1_enc.weight = nn.Parameter(pre_word_embeds)
        # self.fc2_enc = nn.Linear(hidden_dim, hidden_dim)
        # self.fc31_enc = nn.Linear(hidden_dim*102, dfac)
        # self.fc32_enc = nn.Linear(hidden_dim*102, dfac)

        # for title decoder
        self.fc1_dec = nn.Linear(100, self.hidden_dim)
        self.fc2_dec = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3_dec = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc4_dec = nn.Linear(self.hidden_dim, len(char_to_ix))
        # self.swish = Swish()

        # lstm for title words
        self.char_embeds = nn.Embedding(len(char_to_ix), self.hidden_dim)
        init_embedding(self.char_embeds.weight)
        self.char_lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, bidirectional=True)
        init_lstm(self.char_lstm)

        # lstm encoding for item
        self.word_embeds = nn.Embedding(vocab_size, self.hidden_dim)
        self.lstm = nn.LSTM(self.hidden_dim + self.hidden_dim * 2, self.hidden_dim, bidirectional=True)
        init_lstm(self.lstm)

        # lstm decoding for item


        # linear layers
        self.linear_title = nn.Linear(self.hidden_dim*2, self.hidden_dim)

        self.experts = ProductOfExperts()

    def forward(self, input, purchased_items, title_mask, title_length, d):  # data_title: 100x102x100
        batch_size = input.shape[0]

        # clustering
        # cores = F.normalize(self.cores)  # 7*100
        cores = F.normalize(self.cores)
        items = F.normalize(self.items)  # 13015*100

        cates_logits = torch.mm(items, cores.t()) / self.tau  # 13015*7
        cates = self.cate_softmax(cates_logits)  # 13015x7

        title_emb = None
        if purchased_items is not None:
            # # print(torch.max(data_title))
            # # title_emb = self.fc1_enc(data_title).view(batch_size, data_title.shape[1], -1)  # 100x102x51200
            # title_emb = self.fc1_enc(data_title) # 100x102x100
            # title_emb = F.normalize(title_emb)
            # # cores_title = F.normalize(self.cores_title)
            # cores_title = F.normalize(self.cores_title) # 7x100
            # cates_logits_title = title_emb.matmul(cores_title.t()) / self.tau  # 100x102x7
            # cates_title = self.cate_softmax(cates_logits_title)  # 100x102x7

            chars_embeds = self.char_embeds(title_mask).transpose(0, 1) # title_mask: itemxword; wordxitemx100
            packed = torch.nn.utils.rnn.pack_padded_sequence(chars_embeds, title_length)
            lstm_out, _ = self.char_lstm(packed)
            outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out) # outputs: wordxitemx200
            outputs = outputs.transpose(0, 1) # itemxwordx200
            chars_embeds_temp = Variable(torch.FloatTensor(torch.zeros((outputs.size(0), outputs.size(2))))).to(device)
            for i, index in enumerate(output_lengths):
                chars_embeds_temp[i] = torch.cat((outputs[i, index - 1, :self.hidden_dim],
                                                  outputs[i, 0, self.hidden_dim:]))
            chars_embeds = chars_embeds_temp.clone() # itemx200
            for i in range(chars_embeds.size(0)):
                chars_embeds[d[i]] = chars_embeds_temp[i]

            embeds = self.word_embeds(purchased_items) # itemx100
            concat_embeds = torch.cat((embeds, chars_embeds), 1) # itemx300
            concat_embeds = concat_embeds.unsqueeze(1)
            concat_embeds = self.dropout(concat_embeds) # itemx1x300 - batch size is 1

            lstm_out, _ = self.lstm(concat_embeds) # itemx1x200
            lstm_out = lstm_out.view(len(purchased_items), self.hidden_dim * 2) # itemx200

            title_emb = torch.tanh(self.linear_title(lstm_out))
            title_emb = self.dropout(title_emb)

            title_emb = F.normalize(title_emb)
            cores_title = F.normalize(self.cores_title)  # kx200
            cates_logits_title = title_emb.matmul(cores_title.t()) / self.tau  # itemxk
            cates_title = self.cate_softmax(cates_logits_title) # itemxk

        z_list = []
        probs, probs_title = None, None
        std_list, std_list_title = [], []

        for k in range(self.kfac):
            # use_cuda = next(self.parameters()).is_cuda  # check if CUDA
            # initialize the universal prior expert
            # mu_k, std_k = prior_expert((1, batch_size, 100), use_cuda=use_cuda)

            # for seq
            cates_k = cates[:, k].unsqueeze(0)  # 1x13015
            x_k = input * cates_k  # 1x13015
            mu_seq_k, std_seq_k, lnvarq_seq_k = self.encode(x_k)  # 1x100

            if purchased_items is not None:
                cates_k_t = cates_title[:, k].unsqueeze(1)  # itemx1
                title_k = title_emb * cates_k_t  # itemx200
                # combined_k = torch.cat((x_k, title_k.view(batch_size, -1)), dim=1)
                # mu_k, std_k, lnvarq_k = self.encode_combined(combined_k)
                mu_title, std_title, lnvarq_title = self.encode_title(title_k.view(1, -1))  # 1x100

                mu_k = torch.cat((mu_seq_k, mu_title), dim=0)  # 3x100x100
                std_k = torch.cat((std_seq_k, std_title), dim=0)

                # product of gaussians
                mu_k, std_k = self.experts(mu_k, std_k)  # 1x100
            else:
                mu_k, std_k = mu_seq_k, std_seq_k

            # zk embedding
            z_k = self.reparameterize(mu_k, std_k).unsqueeze(0)  # 1x100

            # seq decoder
            z_k = F.normalize(z_k)  # 1x100
            logits_k = torch.mm(z_k, items.t()) / self.tau  # 1x13015
            probs_k = torch.exp(logits_k)
            probs_k = probs_k * cates_k  # 1x13015
            probs = (probs_k if (probs is None) else (probs + probs_k))

            # title decoder
            if purchased_items is not None:
                title_k_decode = z_k.repeat(len(purchased_items), 1, 1)
                title_k = self.title_decoder(title_k_decode)
                title_k = torch.exp(title_k)
                title_k = title_k * cates_k_t # 100x102x17424
                probs_title = (title_k if (probs_title is None) else (probs_title + title_k)) # 100x102x17424
                std_list_title.append(lnvarq_title)

            std_list.append(lnvarq_seq_k)
            if self.save_emb:
                z_list.append(z_k)

        logits = torch.log(probs) # 100x13015
        if purchased_items is not None:
            logits_title = torch.log(probs_title) # 100x102x17424
        else:
            logits_title = None
        # logits = F.log_softmax(logits, dim=-1)

        cluster_loss = None
        if purchased_items is not None:
            cluster_loss = self.cluster_similarity()

        return logits, std_list, items, logits_title, cluster_loss, std_list_title

    def encode(self, input):
        h = F.normalize(input)
        h = self.dropout(h)

        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = torch.tanh(h)
            else:  # for last layer
                mu = h[:, :self.q_dims[-1]]
                mu = F.normalize(mu)
                # logvar = h[:, self.q_dims[-1]:]
                lnvarq_sub_lnvar0 = -h[:, self.q_dims[-1]:]
                std0 = self.std
                std_q = torch.exp(0.5 * lnvarq_sub_lnvar0) * std0
        return mu, std_q, lnvarq_sub_lnvar0

    def encode_combined(self, input):
        h = F.normalize(input)
        h = self.dropout(h)

        for i, layer in enumerate(self.c_layers):
            h = layer(h)
            if i != len(self.c_layers) - 1:
                h = torch.tanh(h)
            else:  # for last layer
                mu = h[:, :self.q_dims[-1]]
                mu = F.normalize(mu)
                # logvar = h[:, self.q_dims[-1]:]
                lnvarq_sub_lnvar0 = -h[:, self.q_dims[-1]:]
                std0 = self.std
                std_q = torch.exp(0.5 * lnvarq_sub_lnvar0) * std0
        return mu, std_q, lnvarq_sub_lnvar0

    def encode_title(self, input):
        h = F.normalize(input)
        h = self.dropout(h)

        for i, layer in enumerate(self.t_layers):
            h = layer(h)
            if i != len(self.c_layers) - 1:
                h = torch.tanh(h)
            else:  # for last layer
                mu = h[:, :self.q_dims[-1]]
                mu = F.normalize(mu)
                # logvar = h[:, self.q_dims[-1]:]
                lnvarq_sub_lnvar0 = -h[:, self.q_dims[-1]:]
                std0 = self.std
                std_q = torch.exp(0.5 * lnvarq_sub_lnvar0) * std0
        return mu, std_q, lnvarq_sub_lnvar0

    # def title_encoder(self, x):
    #     h = self.swish(x)
    #     h = self.swish(self.fc2_enc(h)).view(x.shape[0], -1) # 100x10200
    #     return self.fc31_enc(h), self.fc32_enc(h)

    def title_decoder(self, x):
        # h = self.swish(self.fc1_dec(x)) # 100x100
        # h = self.swish(self.fc2_dec(h))
        # h = self.swish(self.fc3_dec(h))
        h = torch.tanh(self.fc1_dec(x))
        h = torch.tanh(self.fc2_dec(h))
        h = torch.tanh(self.fc3_dec(h))
        return self.fc4_dec(h)

    def reparameterize(self, mu, std):
        if self.training:
            # std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu
            # return eps.mul(std).add_(mu)
        else:
            return mu

    def cate_softmax(self, cates_logits):
        if self.nogb:
            cates = F.softmax(cates_logits, dim=1)
        else:
            cates_dist = F.gumbel_softmax(cates_logits, tau=1)
            cates_mode = F.softmax(cates_logits, dim=1)
            if self.training:
                cates = cates_dist
            else:
                cates = cates_mode
        return cates

    # def decode(self, z):
    #     h = z
    #     for i, layer in enumerate(self.p_layers):
    #         h = layer(h)
    #         if i != len(self.p_layers) - 1:
    #             h = F.tanh(h)
    #     return h

    def init_weights(self):
        for layer in self.q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
        for layer in self.c_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

    def cluster_similarity(self):
        k = self.cores.shape[0]
        cores = F.normalize(self.cores)
        cores_title = F.normalize(self.cores_title)
        tot_loss = 0
        for i in range(k):
            pos_similarity = torch.mm(cores[i, :].unsqueeze(0), cores_title[i, :].unsqueeze(1))
            neg = np.random.randint(k)
            while neg == i:
                neg = np.random.randint(k)
            neg_similarity = torch.mm(cores[neg, :].unsqueeze(0), cores_title[i, :].unsqueeze(1))
            loss = max(0, neg_similarity - pos_similarity)
            tot_loss += loss
        return tot_loss


class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""

    def forward(self, x):
        return x * torch.sigmoid(x)


# class TitleEncoder(nn.Module):
#     """Parametrizes q(z|y).
#
#     @param n_latents: integer
#                       number of latent dimensions
#     """
#
#     def __init__(self, n_latents):
#         super(TitleEncoder, self).__init__()
#         self.fc1 = nn.Embedding(17424, 512)
#         self.fc2 = nn.Linear(512 * 100 * 102, 512)
#         self.fc31 = nn.Linear(512, n_latents)
#         self.fc32 = nn.Linear(512, n_latents)
#         self.swish = Swish()
#
#     def forward(self, x):
#         h = self.fc1(x).view(x.shape[0], -1)
#         h = self.swish(h)
#         h = self.swish(self.fc2(h))
#         return self.fc31(h), self.fc32(h)
#
#
# class TitleDecoder(nn.Module):
#     """Parametrizes p(y|z).
#
#     @param n_latents: integer
#                       number of latent dimensions
#     """
#
#     def __init__(self, n_latents):
#         super(TitleDecoder, self).__init__()
#         self.fc1 = nn.Linear(n_latents, 512)
#         self.fc2 = nn.Linear(512, 512)
#         self.fc3 = nn.Linear(512, 512)
#         self.fc4 = nn.Linear(512, 10)
#         self.swish = Swish()
#
#     def forward(self, z):
#         h = self.swish(self.fc1(z))
#         h = self.swish(self.fc2(h))
#         h = self.swish(self.fc3(h))
#         return self.fc4(h)


# def binary_cross_entropy_with_logits(input, target):
#     """Sigmoid Activation + Binary Cross Entropy
#
#     @param input: torch.Tensor (size N)
#     @param target: torch.Tensor (size N)
#     @return loss: torch.Tensor (size N)
#     """
#     if not (target.size() == input.size()):
#         raise ValueError("Target size ({}) must be the same as input size ({})".format(
#             target.size(), input.size()))
#
#     return (torch.clamp(input, 0) - input * target
#             + torch.log(1 + torch.exp(-torch.abs(input))))


def cross_entropy(input, target, eps=1e-6):
    """k-Class Cross Entropy (Log Softmax + Log Loss)

    @param input: torch.Tensor (size N x K)
    @param target: torch.Tensor (size N x K)
    @param eps: error to add (default: 1e-6)
    @return loss: torch.Tensor (size N)
    """
    if not (target.size(0) == input.size(0)):
        raise ValueError(
            "Target size ({}) must be the same as input size ({})".format(
                target.size(0), input.size(0)))

    log_input = F.log_softmax(input + eps, dim=1)
    y_onehot = Variable(log_input.data.new(log_input.size()).zero_())
    y_onehot = y_onehot.scatter(1, target.unsqueeze(1), 1)
    loss = y_onehot * log_input
    return -loss


def loss_function(x, std_list, recon_x, anneal, title, recon_title, std_list_title):
    # BCE = F.binary_cross_entropy(recon_x, x)
    # BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    # KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    recon_loss = torch.mean(torch.sum(-F.log_softmax(recon_x, 1) * x, -1))
    kl, kl_t = None, None
    for i in range(len(std_list)):
        lnvarq_sub_lnvar0 = std_list[i]
        kl_k = torch.mean(torch.sum(0.5 * (-lnvarq_sub_lnvar0 + torch.exp(lnvarq_sub_lnvar0) - 1.), dim=1))
        kl = (kl_k if (kl is None) else (kl + kl_k))
    # neg_elbo = recon_loss + anneal * kl
    recon_loss_title = 0
    if recon_title is not None:
        # recon_loss_title = torch.sum(cross_entropy(recon_title, title), dim=1)
        recon_loss_title = torch.sum(torch.mean(torch.sum(-F.log_softmax(recon_title, -1), 1) * title, -1))
    for i in range(len(std_list_title)):
        lnvarq_sub_lnvar0 = std_list_title[i]
        kl_k_t = torch.mean(torch.sum(0.5 * (-lnvarq_sub_lnvar0 + torch.exp(lnvarq_sub_lnvar0) - 1.), dim=1))
        kl_t = (kl_k_t if (kl is None) else (kl + kl_k_t))

    return recon_loss + anneal * kl + 0.1*recon_loss_title + anneal * kl_t


# def prior_expert(size, use_cuda=False):
#     """Universal prior expert. Here we use a spherical
#     Gaussian: N(0, 1).
#
#     @param size: integer
#                  dimensionality of Gaussian
#     @param use_cuda: boolean [default: False]
#                      cast CUDA on variables
#     """
#     mu = Variable(torch.zeros(size))
#     logvar = Variable(torch.zeros(size))
#     # std = Variable(torch.zeros(size))
#     cuda = torch.device('cuda:2')
#     if use_cuda:
#         mu, logvar = mu.cuda(cuda), logvar.cuda(cuda)
#     return mu, logvar


class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """
    def forward(self, mu, logvar, eps=1e-8):
        var = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / (var + eps)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + eps)
        return pd_mu, pd_logvar


def init_lstm(input_lstm):
    """
    Initialize lstm

    PyTorch weights parameters:

        weight_ih_l[k]: the learnable input-hidden weights of the k-th layer,
            of shape `(hidden_size * input_size)` for `k = 0`. Otherwise, the shape is
            `(hidden_size * hidden_size)`

        weight_hh_l[k]: the learnable hidden-hidden weights of the k-th layer,
            of shape `(hidden_size * hidden_size)`
    """

    # Weights init for forward layer
    for ind in range(0, input_lstm.num_layers):
        ## Gets the weights Tensor from our model, for the input-hidden weights in our current layer
        weight = eval('input_lstm.weight_ih_l' + str(ind))

        # Initialize the sampling range
        sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))

        # Randomly sample from our samping range using uniform distribution and apply it to our current layer
        nn.init.uniform_(weight, -sampling_range, sampling_range)

        # Similar to above but for the hidden-hidden weights of the current layer
        weight = eval('input_lstm.weight_hh_l' + str(ind))
        sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform_(weight, -sampling_range, sampling_range)

    # We do the above again, for the backward layer if we are using a bi-directional LSTM (our final model uses this)
    if input_lstm.bidirectional:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.weight_ih_l' + str(ind) + '_reverse')
            sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform_(weight, -sampling_range, sampling_range)
            weight = eval('input_lstm.weight_hh_l' + str(ind) + '_reverse')
            sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform_(weight, -sampling_range, sampling_range)

    # Bias initialization steps

    # We initialize them to zero except for the forget gate bias, which is initialized to 1
    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            bias = eval('input_lstm.bias_ih_l' + str(ind))

            # Initializing to zero
            bias.data.zero_()

            # This is the range of indices for our forget gates for each LSTM cell
            bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

            # Similar for the hidden-hidden layer
            bias = eval('input_lstm.bias_hh_l' + str(ind))
            bias.data.zero_()
            bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

        # Similar to above, we do for backward layer if we are using a bi-directional LSTM
        if input_lstm.bidirectional:
            for ind in range(0, input_lstm.num_layers):
                bias = eval('input_lstm.bias_ih_l' + str(ind) + '_reverse')
                bias.data.zero_()
                bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
                bias = eval('input_lstm.bias_hh_l' + str(ind) + '_reverse')
                bias.data.zero_()
                bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1


def init_embedding(input_embedding):
    """
    Initialize embedding
    """
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -bias, bias)