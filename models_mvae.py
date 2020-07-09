import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable


class MultiVAE(nn.Module):
    """
    Container module for Multi-VAE.

    Multi-VAE : Variational Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, q_dims=None, dropout=0.5, tau=0.1, std=0.075, kfac=7, nogb=False):
        super(MultiVAE, self).__init__()
        self.p_dims = p_dims
        if q_dims:
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
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])

        dfac = self.q_dims[-1]
        self.kfac = kfac
        num_items = self.q_dims[0]
        self.cores = nn.Parameter(torch.empty(self.kfac, dfac))
        self.items = nn.Parameter(torch.empty(num_items, dfac))
        nn.init.xavier_normal_(self.cores.data)
        nn.init.xavier_normal_(self.items.data)
        self.tau = tau
        self.std = std  # Standard deviation of the Gaussian prior
        self.save_emb = False
        self.nogb = nogb

        self.drop = nn.Dropout(dropout)
        self.init_weights()

        # center for title
        self.cores_title = nn.Parameter(torch.empty(self.kfac, 100*512))
        nn.init.xavier_normal_(self.cores_title.data)
        # for title
        self.fc1 = nn.Embedding(17424, 512)
        self.fc2 = nn.Linear(512 * 100 * 102, 512)
        self.fc31 = nn.Linear(512, 100)
        self.fc32 = nn.Linear(512, 100)
        self.swish = Swish()

        self.experts = ProductOfExperts()

    def forward(self, input, data_title=None):
        batch_size = input.shape[0]

        # clustering
        cores = F.normalize(self.cores) # 7*100
        items = F.normalize(self.items) # 13015*100

        cates_logits = torch.mm(items, cores.t()) / self.tau # 13015*7
        cates = self.cate_softmax(cates_logits) # 13015x7

        title_emb = self.fc1(data_title).view(batch_size, data_title.shape[1], -1) # 100x102x51200
        cates_logits_title = title_emb.matmul(self.cores_title.t())/self.tau # 100x102x7
        cates_title = self.cate_softmax(cates_logits_title) # 100x102x7

        z_list = []
        probs = None
        std_list = []

        for k in range(self.kfac):
            use_cuda = next(self.parameters()).is_cuda  # check if CUDA
            # initialize the universal prior expert
            mu_k, std_k = prior_expert((1, batch_size, 100), use_cuda=use_cuda)

            # for seq
            cates_k = cates[:, k].unsqueeze(0) # 1x13015
            x_k = input * cates_k # 100x13015
            mu_seq_k, std_seq_k, lnvarq_seq_k = self.encode(x_k) # 100x100
            mu_k = torch.cat((mu_k, mu_seq_k.unsqueeze(0)), dim=0) # 2x100x100
            std_k = torch.cat((std_k, std_seq_k.unsqueeze(0)), dim=0) # 2x100x100

            if data_title is not None:
                cates_k_t = cates_title[:, :, k].unsqueeze(2) # 100x102x1
                title_k = title_emb * cates_k_t # 100x102x51200
                mu_title, std_title = self.title_encoder(title_k) # 100x100
                mu_k = torch.cat((mu_k, mu_title.unsqueeze(0)), dim=0) # 3x100x100
                std_k = torch.cat((std_k, std_title.unsqueeze(0)), dim=0)

            # product of gaussians
            mu_k, std_k = self.experts(mu_k, std_k) # 100x100

            # zk embedding
            z_k = self.reparameterize(mu_k, std_k)


            # seq decoder
            z_k = F.normalize(z_k)
            logits_k = torch.mm(z_k, items.t()) / self.tau
            probs_k = torch.exp(logits_k)
            probs_k = probs_k * cates_k
            probs = (probs_k if (probs is None) else (probs + probs_k))

            # title decoder
            # title_recon = self.title_decoder(z_k)

            std_list.append(lnvarq_seq_k)
            if self.save_emb:
                z_list.append(z_k)

        logits = torch.log(probs)
        # logits = F.log_softmax(logits, dim=-1)

        return logits, std_list

    def encode(self, input):
        h = F.normalize(input)
        h = self.drop(h)

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

    def title_encoder(self, x):
        h = torch.tanh(x.view(x.shape[0], -1))
        h = torch.tanh(self.fc2(h))
        return self.fc31(h), self.fc32(h)

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

        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class TitleEncoder(nn.Module):
    """Parametrizes q(z|y).

    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents):
        super(TitleEncoder, self).__init__()
        self.fc1 = nn.Embedding(17424, 512)
        self.fc2 = nn.Linear(512*100*102, 512)
        self.fc31 = nn.Linear(512, n_latents)
        self.fc32 = nn.Linear(512, n_latents)
        self.swish = Swish()

    def forward(self, x):
        h = self.fc1(x).view(x.shape[0], -1)
        h = self.swish(h)
        h = self.swish(self.fc2(h))
        return self.fc31(h), self.fc32(h)


class TitleDecoder(nn.Module):
    """Parametrizes p(y|z).

    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents):
        super(TitleDecoder, self).__init__()
        self.fc1 = nn.Linear(n_latents, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 10)
        self.swish = Swish()

    def forward(self, z):
        h = self.swish(self.fc1(z))
        h = self.swish(self.fc2(h))
        h = self.swish(self.fc3(h))
        return self.fc4(h)


def loss_function(x, std_list, recon_x, anneal=1.0):
    # BCE = F.binary_cross_entropy(recon_x, x)
    # BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    # KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    recon_loss = torch.mean(torch.sum(-F.log_softmax(recon_x, 1) * x, -1))
    kl = None
    for i in range(len(std_list)):
        lnvarq_sub_lnvar0 = std_list[i]
        kl_k = torch.mean(torch.sum(0.5 * (-lnvarq_sub_lnvar0 + torch.exp(lnvarq_sub_lnvar0) - 1.), dim=1))
        kl = (kl_k if (kl is None) else (kl + kl_k))
    # neg_elbo = recon_loss + anneal * kl

    return recon_loss + anneal * kl


def prior_expert(size, use_cuda=False):
    """Universal prior expert. Here we use a spherical
    Gaussian: N(0, 1).

    @param size: integer
                 dimensionality of Gaussian
    @param use_cuda: boolean [default: False]
                     cast CUDA on variables
    """
    mu     = Variable(torch.zeros(size))
    logvar = Variable(torch.zeros(size))
    # std = Variable(torch.zeros(size))
    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar


class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """
    def forward(self, mu, logvar, eps=1e-8):
        var       = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T         = 1. / (var + eps)
        pd_mu     = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var    = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + eps)
        return pd_mu, pd_logvar