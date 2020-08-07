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

    def __init__(self, vocab_size, p_dims, q_dims=None, dropout=0.5, std=0.075, tau=0.1, kfac=1):
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
        # self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
        #                                d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])

        dfac = self.q_dims[-1]
        self.kfac = kfac
        num_items = self.q_dims[0]
        self.items = nn.Parameter(torch.empty(num_items, dfac))
        nn.init.xavier_normal_(self.items.data)
        self.tau = tau
        self.std = std  # Standard deviation of the Gaussian prior

        # direct concat
        self.a_dims = [vocab_size + self.q_dims[0]] + self.q_dims[1:]
        temp_a_layers = [vocab_size + self.q_dims[0]] + [self.q_dims[1]] + [self.q_dims[-1] * 2]
        self.a_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(temp_a_layers[:-1], temp_a_layers[1:])])

        # title
        self.hidden_dim = 100
        self.t_dims = [vocab_size, self.hidden_dim, self.hidden_dim]
        self.t_d_dims = self.t_dims[::-1]
        temp_t_dims = self.t_dims[:-1] + [self.t_dims[-1] * 2]  # same as q_dims but last element *2
        self.t_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(temp_t_dims[:-1], temp_t_dims[1:])])
        self.t_d_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                         d_in, d_out in zip(self.t_d_dims[:-1], self.t_d_dims[1:])])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, input, title):
        logits, logits_title = None, None
        if input is not None and title is None:
            # clustering
            items = F.normalize(self.items)
            # q-network
            mu, std, logvar = self.encode(input)
            z = self.reparameterize(mu, std)
            # p-network
            z = F.normalize(z)
            logits = torch.mm(z, items.t()) / self.tau
            logits_title = self.decode_title(z)

        if input is not None and title is not None:
            # clustering
            items = F.normalize(self.items)
            # q-network
            combined = torch.cat((input, title), dim=1)
            mu, std, logvar = self.encode_all(combined)
            z = self.reparameterize(mu, std)
            logits = torch.mm(F.normalize(z), items.t()) / self.tau
            logits_title = self.decode_title(z)

        if input is None and title is not None:
            mu, logvar = self.encode_title(title)
            z = self.reparameterize(mu, logvar)
            logits_title = self.decode_title(z)

            items = F.normalize(self.items)
            logits = torch.mm(F.normalize(z), items.t()) / self.tau

        return logits, mu, logvar, logits_title

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

    def encode_all(self, input):
        h = F.normalize(input)
        h = self.drop(h)

        for i, layer in enumerate(self.a_layers):
            h = layer(h)
            if i != len(self.a_layers) - 1:
                h = torch.tanh(h)
            else:  # for last layer
                mu = h[:, :self.a_dims[-1]]
                logvar = -h[:, self.a_dims[-1]:]
                std0 = self.std
                std_q = torch.exp(0.5 * logvar) * std0
        return mu, std_q, logvar

    def encode_title(self, input):
        h = F.normalize(input)
        h = self.drop(h)

        for i, layer in enumerate(self.t_layers):
            h = layer(h)
            if i != len(self.t_layers) - 1:
                h = torch.tanh(h)
            else:  # for last layer
                mu = h[:, :self.t_dims[-1]]
                logvar = h[:, self.t_dims[-1]:]
        return mu, logvar

    def decode_title(self, z):
        h = z
        for i, layer in enumerate(self.t_d_layers):
            h = layer(h)
            if i != len(self.t_d_layers) - 1:
                h = torch.tanh(h)
        return h

    def reparameterize(self, mu, std):
        if self.training:
            # std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu
            # return eps.mul(std).add_(mu)
        else:
            return mu

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

        for layer in self.a_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.t_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.t_d_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


def loss_function(recon_x, x, mu, logvar, recon_t, t, anneal=1.0):
    # BCE = F.binary_cross_entropy(recon_x, x)
    # BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    # KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    bce, t_bce, kl = 0, 0, 0
    if recon_x is not None and x is not None:
        bce = torch.mean(torch.sum(-F.log_softmax(recon_x, 1) * x, -1))
        kl = torch.mean(torch.sum(0.5 * (-logvar + torch.exp(logvar) - 1.), dim=1))
    if recon_t is not None and t is not None:
        t_bce = -torch.mean(torch.sum(F.log_softmax(recon_t, 1) * t, -1))
    if recon_x is None and x is None:
        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    return bce + anneal * kl + t_bce
