import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class MultiVAE(nn.Module):
    """
    Container module for Multi-VAE.

    Multi-VAE : Variational Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, q_dims=None, dropout=0.5, vocab_size=None, tot_items=None, max_item=None):
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
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]  # same as q_dims but last element *2
        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        # self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
        #                                d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])

        self.hidden_dim = 100
        # title
        # self.t_dims = [vocab_size, self.hidden_dim, self.hidden_dim]
        # self.t_d_dims = self.t_dims[::-1]
        # temp_t_dims = self.t_dims[:-1] + [self.t_dims[-1] * 2]  # same as q_dims but last element *2
        # self.t_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
        #                                d_in, d_out in zip(temp_t_dims[:-1], temp_t_dims[1:])])
        # self.t_d_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
        #                                  d_in, d_out in zip(self.t_d_dims[:-1], self.t_d_dims[1:])])

        # direct concat
        self.a_dims = [vocab_size+self.q_dims[0]] + self.q_dims[1:]
        temp_a_layers = [vocab_size+self.q_dims[0]] + [self.q_dims[1]] + [self.q_dims[-1] * 2]
        self.a_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(temp_a_layers[:-1], temp_a_layers[1:])])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

        # self.experts = ProductOfExperts()

        ##############################################
        # parameters
        self.tau = 0.1
        self.std = 0.075
        self.kfac = 1
        self.nogb = False

        # clustering cores
        self.cores = nn.Parameter(torch.empty(self.kfac, self.hidden_dim))
        self.items = nn.Parameter(torch.empty(tot_items, self.hidden_dim))
        nn.init.xavier_normal_(self.cores.data)
        nn.init.xavier_normal_(self.items.data)

        self.cores_title = nn.Parameter(torch.empty(self.kfac, self.hidden_dim))
        self.words = nn.Parameter(torch.empty(vocab_size, self.hidden_dim))
        nn.init.xavier_normal_(self.cores_title.data)
        nn.init.xavier_normal_(self.words.data)

    def forward(self, input, title):
        # distribution of items over clusters
        cores = F.normalize(self.cores)
        items = F.normalize(self.items)
        cates_logits = torch.mm(items, cores.t()) / self.tau

        # gumbel softmax
        if self.nogb:
            cates = F.softmax(cates_logits, dim=1)
        else:
            cates_dist = F.gumbel_softmax(cates_logits, tau=1)
            cates_mode = F.softmax(cates_logits, dim=1)
            if self.training:
                cates = cates_dist
            else:
                cates = cates_mode

        # prob over k clusters
        probs, probs_title = None, None
        std_list = []

        if input is not None and title is None:
            for k in range(self.kfac):
                cates_k = cates[:, k].unsqueeze(0)

                # encode
                x_k = input * cates_k
                mu_k, std_k, logvar_k = self.encode(x_k)

                z_k = self.reparameterize(mu_k, std_k)
                z_k = F.normalize(z_k)

                # decode
                logits_k = torch.mm(z_k, items.t()) / self.tau
                probs_k = torch.exp(logits_k)
                probs_k = probs_k * cates_k
                probs = (probs_k if (probs is None) else (probs + probs_k))

                std_list.append(logvar_k)

            logits = torch.log(probs)
            # mu, logvar = self.encode(input)

        if input is not None and title is not None:
            # mu_s, logvar_s = self.encode(input)
            # mu_t, logvar_t = self.encode_title(title)
            #
            # # POE
            # mu = torch.cat((mu_t.unsqueeze(0), mu_s.unsqueeze(0)), dim=0)  # 2x100
            # logvar = torch.cat((logvar_t.unsqueeze(0), logvar_s.unsqueeze(0)), dim=0)  # 2x100
            # mu, logvar = self.experts(mu, logvar)

            # # direct concatenate
            # combined = torch.cat((input, title), dim=1)
            # mu, logvar = self.encode_all(combined)

            cores_title = F.normalize(self.cores_title)
            words = F.normalize(self.words)
            cates_logits_title = torch.mm(words, cores_title.t()) / self.tau

            # gumbel softmax
            if self.nogb:
                cates_title = F.softmax(cates_logits_title, dim=1)
            else:
                cates_dist = F.gumbel_softmax(cates_logits_title, tau=1)
                cates_mode = F.softmax(cates_logits_title, dim=1)
                if self.training:
                    cates_title = cates_dist
                else:
                    cates_title = cates_mode

            for k in range(self.kfac):
                cates_k = cates[:, k].unsqueeze(0)
                cates_title_k = cates_title[:, k].unsqueeze(0)

                # encode
                x_k = input * cates_k
                title_k = title * cates_title_k
                combined = torch.cat((x_k, title_k), dim=1)
                mu_k, std_k, logvar_k = self.encode_all(combined)

                z_k = self.reparameterize(mu_k, std_k)
                z_k = F.normalize(z_k)

                # decode
                logits_k = torch.mm(z_k, items.t()) / self.tau
                probs_k = torch.exp(logits_k)
                probs_k = probs_k * cates_k
                probs = (probs_k if (probs is None) else (probs + probs_k))

                logits_title_k = torch.mm(z_k, words.t()) / self.tau
                probs_title_k = torch.exp(logits_title_k)
                probs_title_k = probs_title_k * cates_title_k
                probs_title = (probs_title_k if (probs_title is None) else (probs_title + probs_title_k))

                std_list.append(logvar_k)

            logits = torch.log(probs)
            logits_title = torch.log(probs_title)

        if input is None and title is not None:
            mu, logvar = self.encode_title(title)

        # z = self.reparameterize(mu, logvar)

        return logits, std_list, (logits_title if title is not None else None)

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
                logvar = h[:, self.q_dims[-1]:]
                std = torch.exp(0.5 * -logvar) * self.std
        return mu, std, -logvar

    # def encode_title(self, input):
    #     h = F.normalize(input)
    #     h = self.drop(h)
    #
    #     for i, layer in enumerate(self.t_layers):
    #         h = layer(h)
    #         if i != len(self.t_layers) - 1:
    #             h = torch.tanh(h)
    #         else:  # for last layer
    #             mu = h[:, :self.t_dims[-1]]
    #             mu = F.normalize(mu)
    #             logvar = h[:, self.t_dims[-1]:]
    #             std = torch.exp(0.5 * -logvar) * self.std
    #     return mu, std, -logvar

    def encode_all(self, input):
        h = F.normalize(input)
        h = self.drop(h)

        for i, layer in enumerate(self.a_layers):
            h = layer(h)
            if i != len(self.a_layers) - 1:
                h = torch.tanh(h)
            else:  # for last layer
                mu = h[:, :self.a_dims[-1]]
                mu = F.normalize(mu)
                logvar = h[:, self.a_dims[-1]:]
                std = torch.exp(0.5 * -logvar) * self.std
        return mu, std, -logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    # def decode(self, z):
    #     h = z
    #     for i, layer in enumerate(self.p_layers):
    #         h = layer(h)
    #         if i != len(self.p_layers) - 1:
    #             h = torch.tanh(h)
    #     return h

    # def decode_title(self, z):
    #     h = z
    #     for i, layer in enumerate(self.t_d_layers):
    #         h = layer(h)
    #         if i != len(self.t_d_layers) - 1:
    #             h = torch.tanh(h)
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

        # for layer in self.p_layers:
        #     # Xavier Initialization for weights
        #     size = layer.weight.size()
        #     fan_out = size[0]
        #     fan_in = size[1]
        #     std = np.sqrt(2.0 / (fan_in + fan_out))
        #     layer.weight.data.normal_(0.0, std)
        #
        #     # Normal Initialization for Biases
        #     layer.bias.data.normal_(0.0, 0.001)

        # for layer in self.t_layers:
        #     # Xavier Initialization for weights
        #     size = layer.weight.size()
        #     fan_out = size[0]
        #     fan_in = size[1]
        #     std = np.sqrt(2.0 / (fan_in + fan_out))
        #     layer.weight.data.normal_(0.0, std)
        #
        #     # Normal Initialization for Biases
        #     layer.bias.data.normal_(0.0, 0.001)

        # for layer in self.t_d_layers:
        #     # Xavier Initialization for weights
        #     size = layer.weight.size()
        #     fan_out = size[0]
        #     fan_in = size[1]
        #     std = np.sqrt(2.0 / (fan_in + fan_out))
        #     layer.weight.data.normal_(0.0, std)
        #
        #     # Normal Initialization for Biases
        #     layer.bias.data.normal_(0.0, 0.001)

        for layer in self.a_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


# def loss_function(recon_x, x, mu, logvar, anneal=1.0):
#     # BCE = F.binary_cross_entropy(recon_x, x)
#     BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
#     KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
#
#     return BCE + anneal * KLD


def loss_function_title(recon_x, x, mu, logvar, recon_t, t, anneal=1.0):
    # BCE = F.binary_cross_entropy(recon_x, x)
    bce, t_bce = 0, 0
    if recon_x is not None and x is not None:
        bce = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    if recon_t is not None and t is not None:
        t_bce = -torch.mean(torch.sum(F.log_softmax(recon_t, 1) * t, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return bce + anneal * KLD + t_bce


def loss_function(recon_x, x, std_list, recon_t, t, anneal=1.0):
    recon_loss, recon_loss_t = 0, 0
    if recon_x is not None and x is not None:
        recon_loss = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    if recon_t is not None and t is not None:
        recon_loss_t = -torch.mean(torch.sum(F.log_softmax(recon_t, 1) * t, -1))
    kl = None
    for i in range(len(std_list)):
        logvar = std_list[i]
        kl_k = torch.mean(torch.sum(0.5 * (-logvar + torch.exp(logvar) - 1.), dim=1))
        kl = (kl_k if (kl is None) else (kl + kl_k))
    return recon_loss + anneal * kl + recon_loss_t


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