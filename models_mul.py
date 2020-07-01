import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence


class MultiVAE(nn.Module):
    """
    Container module for Multi-VAE.

    Multi-VAE : Variational Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, title_data, image_data,
                 q_dims=None, dropout=0.5, tau=0.1, std=0.075, kfac=7, nogb=False):
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
        # nn.init.xavier_normal_(self.items.data)

        # self.items.data = title_data # replace item with title
        # self.items.data = image_data # replace item with image

        # # concat random with title
        # self.title = nn.Parameter(torch.empty(num_items, dfac))
        # self.title.data = title_data
        # self.linear = nn.Linear(dfac+dfac, dfac)

        # # concate random with image
        # self.image = nn.Parameter(torch.empty(num_items, 2048))
        # self.image.data = image_data
        # self.linear = nn.Linear(dfac+2048, dfac)

        # concate title with image
        self.title = nn.Parameter(torch.empty(num_items, dfac))
        self.title.data = title_data
        self.image = nn.Parameter(torch.empty(num_items, 2048))
        self.image.data = image_data
        self.linear = nn.Linear(dfac+2048, dfac)

        self.tau = tau
        self.std = std  # Standard deviation of the Gaussian prior
        self.save_emb = False
        self.nogb = nogb

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, input):
        # clustering
        cores = F.normalize(self.cores)
        # items = F.normalize(self.items)

        # # concate random with title
        # title = torch.cat((self.items, self.title), dim=1)
        # items = self.linear(title)
        # items = F.normalize(items)

        # concate random with image
        items = torch.cat((self.title, self.image), dim=1)
        items = self.linear(items)
        items = F.normalize(items)
        cates_logits = torch.mm(items, cores.t()) / self.tau

        if self.nogb:
            cates = F.softmax(cates_logits, dim=1)
        else:
            cates_dist = F.gumbel_softmax(cates_logits, tau=1)
            cates_mode = F.softmax(cates_logits, dim=1)
            if self.training:
                cates = cates_dist
            else:
                cates = cates_mode

        z_list = []
        probs = None
        std_list = []
        for k in range(self.kfac):
            # cates_k = torch.flatten(cates[:, k])
            cates_k = cates[:, k].unsqueeze(0)

            # q-network
            x_k = input * cates_k
            mu_k, std_k, lnvarq_sub_lnvar0_k = self.encode(x_k)
            z_k = self.reparameterize(mu_k, std_k)
            std_list.append(lnvarq_sub_lnvar0_k)

            if self.save_emb:
                z_list.append(z_k)

            # p-network
            z_k = F.normalize(z_k)
            logits_k = torch.mm(z_k, items.t()) / self.tau
            probs_k = torch.exp(logits_k)
            probs_k = probs_k * cates_k
            probs = (probs_k if (probs is None) else (probs + probs_k))

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

        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


class MultiVAE_CNN(nn.Module):
    """
    Container module for Multi-VAE.

    Multi-VAE : Variational Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, tau, std, kfac,
                 vocab_size, embedding_dim, hidden_dim, title_data, image_data,
                 dropout, nogb=False, q_dims=None):
        super(MultiVAE_CNN, self).__init__()

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
        self.cores = nn.Parameter(torch.empty(self.kfac, dfac), requires_grad=True)
        self.items = nn.Parameter(torch.empty(num_items, dfac), requires_grad=True)
        nn.init.xavier_normal_(self.cores.data)
        nn.init.xavier_normal_(self.items.data)
        # self.cores.data = init_kmeans
        # self.items.data = init_kmeans

        self.tau = tau
        self.std = std  # Standard deviation of the Gaussian prior
        self.save_emb = False
        self.nogb = nogb

        self.drop = nn.Dropout(dropout)
        self.init_weights()

        # newly added
        # load title and image
        self.title = nn.Parameter(torch.empty(num_items, 100))
        self.title.data = title_data
        self.image = nn.Parameter(torch.empty(num_items, 2048))
        self.image.data = image_data

        # model for title
        self.word_dilated = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=1, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=(1, 5), stride=1)
        )

        # model for image
        self.img_layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 5), stride=1)
        )
        self.img_layer2 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 5), stride=1)
        )

        self.linear = nn.Linear(dfac+100+2048-12, dfac)
        self.drop_title = nn.Dropout(dropout)

    def forward(self, input):
        # clustering
        cores = F.normalize(self.cores)
        # items = F.normalize(self.items)

        # process title and image
        title = self.word_dilated(self.title.unsqueeze(1).unsqueeze(1))
        title = title.squeeze()
        image = self.img_layer1(self.image.unsqueeze(1).unsqueeze(1))
        image = self.img_layer2(image)
        image = image.squeeze()

        items_concat = torch.cat((self.items, title, image), 1)
        items_final = self.linear(items_concat)
        items_final = F.normalize(items_final)
        cates_logits = torch.mm(items_final, cores.t()) / self.tau

        if self.nogb:
            cates = F.softmax(cates_logits, dim=1)
        else:
            cates_dist = F.gumbel_softmax(cates_logits, tau=1)
            cates_mode = F.softmax(cates_logits, dim=1)
            if self.training:
                cates = cates_dist
            else:
                cates = cates_mode

        z_list = []
        probs = None
        std_list = []
        for k in range(self.kfac):
            # cates_k = torch.flatten(cates[:, k])
            cates_k = cates[:, k].unsqueeze(0)

            # q-network
            x_k = input * cates_k
            mu_k, std_k, lnvarq_sub_lnvar0_k = self.encode(x_k)
            z_k = self.reparameterize(mu_k, std_k)
            std_list.append(lnvarq_sub_lnvar0_k)

            if self.save_emb:
                z_list.append(z_k)

            # p-network
            z_k = F.normalize(z_k)
            # logits_k = torch.mm(z_k, self.items.t()) / self.tau
            logits_k = torch.mm(z_k, items_final.t()) / self.tau
            probs_k = torch.exp(logits_k)
            probs_k = probs_k * cates_k
            probs = (probs_k if (probs is None) else (probs + probs_k))

        logits = torch.log(probs)
        # logits = F.log_softmax(logits, dim=-1)

        return std_list, logits

    def encode(self, input):
        h = F.normalize(input)
        h = self.drop(h)

        mu, std_q, lnvarq_sub_lnvar0 = None, None, None
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

        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


def loss_function_mul(x, std_list, std_list_t, std_list_i, recon_x, anneal=1.0):
    # BCE = F.binary_cross_entropy(recon_x, x)
    # BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    # KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    recon_loss = torch.mean(torch.sum(-F.log_softmax(recon_x, 1) * x, -1))
    kl = None
    kl_t, kl_i = None, None
    for i in range(len(std_list)):
        lnvarq_sub_lnvar0 = std_list[i]
        kl_k = torch.mean(torch.sum(0.5 * (-lnvarq_sub_lnvar0 + torch.exp(lnvarq_sub_lnvar0) - 1.), dim=1))
        kl = (kl_k if (kl is None) else (kl + kl_k))

        lnvarq_sub_lnvar0_t = std_list_t[i]
        kl_k_t = torch.mean(torch.sum(0.5 * (-lnvarq_sub_lnvar0_t + torch.exp(lnvarq_sub_lnvar0_t) - 1.), dim=1))
        kl_t = (kl_k_t if (kl_t is None) else (kl_t + kl_k_t))

        lnvarq_sub_lnvar0_i = std_list_i[i]
        kl_k_i = torch.mean(torch.sum(0.5 * (-lnvarq_sub_lnvar0_i + torch.exp(lnvarq_sub_lnvar0_i) - 1.), dim=1))
        kl_i = (kl_k_i if (kl_i is None) else (kl_i + kl_k_i))

    # neg_elbo = recon_loss + anneal * kl
    return recon_loss + anneal * (kl+kl_i+kl_t)


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


class MultiVAE_Mul(nn.Module):
    """
    Container module for Multi-VAE.

    Multi-VAE : Variational Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, title_data, image_data,
                 q_dims=None, dropout=0.5, tau=0.1, std=0.075, kfac=7, nogb=False):
        super(MultiVAE_Mul, self).__init__()
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
        # self.cores = nn.Parameter(torch.empty(self.kfac, dfac))
        self.items = nn.Parameter(torch.empty(num_items, dfac))
        self.cores = nn.Parameter(torch.empty(self.kfac, dfac))
        nn.init.xavier_normal_(self.cores.data)
        nn.init.xavier_normal_(self.items.data)
        self.tau = tau
        self.std = std  # Standard deviation of the Gaussian prior
        self.save_emb = False
        self.nogb = nogb

        # load title and image
        self.title = nn.Parameter(torch.empty(num_items, 100))
        self.title.data = title_data
        self.image = nn.Parameter(torch.empty(num_items, 2048))
        self.image.data = image_data
        self.cores_t = nn.Parameter(torch.empty(self.kfac, 100))
        nn.init.xavier_normal_(self.cores_t.data)
        self.cores_i = nn.Parameter(torch.empty(self.kfac, 2048))
        nn.init.xavier_normal_(self.cores_i.data)

        self.q_layers_t = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        self.q_dims_i = [num_items, dfac, 2048]
        temp_q_dims_i = self.q_dims_i[:-1] + [self.q_dims_i[-1] * 2]
        self.q_layers_i = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(temp_q_dims_i[:-1], temp_q_dims_i[1:])])


        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, input):
        # clustering
        cates, items = self.clustering(self.cores, self.items)
        cates_t, title = self.clustering(self.cores_t, self.title)
        cates_i, image = self.clustering(self.cores_i, self.image)

        z_list = []
        probs = None
        std_list, std_list_t,  std_list_i= [], [], []
        for k in range(self.kfac):
            # cates_k = torch.flatten(cates[:, k])
            cates_k = cates[:, k].unsqueeze(0)
            cates_k_t = cates_t[:, k].unsqueeze(0)
            cates_k_i = cates_i[:, k].unsqueeze(0)

            # q-network
            x_k = input * cates_k
            mu_k, std_k, lnvarq_sub_lnvar0_k = self.encode(x_k, self.q_layers)
            z_k = self.reparameterize(mu_k, std_k)
            std_list.append(lnvarq_sub_lnvar0_k)

            # q-network for title
            x_k_t = input * cates_k_t
            mu_k_t, std_k_t, lnvarq_sub_lnvar0_k_t = self.encode(x_k_t, self.q_layers_t)
            z_k_t = self.reparameterize(mu_k_t, std_k_t)
            std_list_t.append(lnvarq_sub_lnvar0_k_t)

            # q-network for image
            x_k_i = input * cates_k_i
            mu_k_i, std_k_i, lnvarq_sub_lnvar0_k_i = self.encode_img(x_k_i, self.q_layers_i)
            z_k_i = self.reparameterize(mu_k_i, std_k_i)
            std_list_i.append(lnvarq_sub_lnvar0_k_i)

            if self.save_emb:
                z_list.append(z_k)

            # p-network
            z_k = F.normalize(z_k)
            logits_k = torch.mm(z_k, items.t()) / self.tau
            probs_k = torch.exp(logits_k)
            probs_k = probs_k * cates_k

            # p-network for title
            z_k_t = F.normalize(z_k_t)
            logits_k_t = torch.mm(z_k_t, title.t()) / self.tau
            probs_k_t = torch.exp(logits_k_t)
            probs_k_t = probs_k_t * cates_k_t

            # p-network for image
            z_k_i = F.normalize(z_k_i)
            logits_k_i = torch.mm(z_k_i, image.t()) / self.tau
            probs_k_i = torch.exp(logits_k_i)
            probs_k_i = probs_k_i * cates_k_i

            probs_k_avg = (probs_k + probs_k_t + probs_k_i)/3
            probs = (probs_k_avg if (probs is None) else (probs + probs_k_avg))

        logits = torch.log(probs)
        # logits = F.log_softmax(logits, dim=-1)

        return logits, std_list, std_list_t, std_list_i

    def clustering(self, cores, items):
        cores = F.normalize(cores)
        items = F.normalize(items)
        cates_logits = torch.mm(items, cores.t()) / self.tau
        if self.nogb:
            cates = F.softmax(cates_logits, dim=1)
        else:
            cates_dist = F.gumbel_softmax(cates_logits, tau=1)
            cates_mode = F.softmax(cates_logits, dim=1)
            if self.training:
                cates = cates_dist
            else:
                cates = cates_mode
        return cates, items

    def encode(self, input, layers):
        h = F.normalize(input)
        h = self.drop(h)

        for i, layer in enumerate(layers):
            h = layer(h)
            if i != len(layers) - 1:
                h = torch.tanh(h)
            else:  # for last layer
                mu = h[:, :self.q_dims[-1]]
                mu = F.normalize(mu)
                # logvar = h[:, self.q_dims[-1]:]
                lnvarq_sub_lnvar0 = -h[:, self.q_dims[-1]:]
                std0 = self.std
                std_q = torch.exp(0.5 * lnvarq_sub_lnvar0) * std0
        return mu, std_q, lnvarq_sub_lnvar0

    def encode_img(self, input, layers):
        h = F.normalize(input)
        h = self.drop(h)

        for i, layer in enumerate(layers):
            h = layer(h)
            if i != len(layers) - 1:
                h = torch.tanh(h)
            else:  # for last layer
                mu = h[:, :self.q_dims_i[-1]]
                mu = F.normalize(mu)
                # logvar = h[:, self.q_dims[-1]:]
                lnvarq_sub_lnvar0 = -h[:, self.q_dims_i[-1]:]
                std0 = self.std
                std_q = torch.exp(0.5 * lnvarq_sub_lnvar0) * std0
        return mu, std_q, lnvarq_sub_lnvar0

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

        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


class MultiVAE_user(nn.Module):
    """
    Container module for Multi-VAE.

    Multi-VAE : Variational Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, title_data, image_data,
                 q_dims=None, dropout=0.5, tau=0.1, std=0.075, kfac=7, nogb=False):
        super(MultiVAE_user, self).__init__()
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
        self.q_cnn = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 12), dilation=9, stride=1, padding=0)

        dfac = self.q_dims[-1]
        self.kfac = kfac
        num_items = self.q_dims[0]
        # self.cores = nn.Parameter(torch.empty(self.kfac, dfac))
        self.items = nn.Parameter(torch.empty(num_items, dfac))
        self.cores = nn.Parameter(torch.empty(self.kfac, dfac))
        nn.init.xavier_normal_(self.cores.data)
        nn.init.xavier_normal_(self.items.data)
        self.tau = tau
        self.std = std  # Standard deviation of the Gaussian prior
        self.save_emb = False
        self.nogb = nogb

        # load title and image
        self.title = nn.Parameter(torch.empty(num_items, 100))
        self.title.data = title_data
        self.image = nn.Parameter(torch.empty(num_items, 2048))
        self.image.data = image_data
        # self.cores_t = nn.Parameter(torch.empty(self.kfac, 100))
        # nn.init.xavier_normal_(self.cores_t.data)
        # self.cores_i = nn.Parameter(torch.empty(self.kfac, 2048))
        # nn.init.xavier_normal_(self.cores_i.data)
        #
        # self.q_layers_t = nn.ModuleList([nn.Linear(d_in, d_out) for
        #                                d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        # self.q_dims_i = [num_items, dfac, 2048]
        # temp_q_dims_i = self.q_dims_i[:-1] + [self.q_dims_i[-1] * 2]
        # self.q_layers_i = nn.ModuleList([nn.Linear(d_in, d_out) for
        #                                d_in, d_out in zip(temp_q_dims_i[:-1], temp_q_dims_i[1:])])
        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, input):
        # clustering
        cates, items = self.clustering(self.cores, self.items)

        z_list = []
        probs = None
        std_list, std_list_t,  std_list_i= [], [], []
        for k in range(self.kfac):
            # cates_k = torch.flatten(cates[:, k])
            cates_k = cates[:, k].unsqueeze(0)

            # q-network
            title = self.title.repeat(input.shape[0], 1, 1)
            tmp = input.unsqueeze(2) * title
            x_k = tmp * cates_k.unsqueeze(2)
            x_k = x_k.unsqueeze(1)
            x_k = self.q_cnn(x_k).squeeze()
            mu_k, std_k, lnvarq_sub_lnvar0_k = self.encode(x_k, self.q_layers)
            z_k = self.reparameterize(mu_k, std_k)
            std_list.append(lnvarq_sub_lnvar0_k)

            if self.save_emb:
                z_list.append(z_k)

            # p-network
            z_k = F.normalize(z_k)
            logits_k = torch.mm(z_k, items.t()) / self.tau
            probs_k = torch.exp(logits_k)
            probs_k = probs_k * cates_k

            probs = (probs_k if (probs is None) else (probs + probs_k))

        logits = torch.log(probs)
        # logits = F.log_softmax(logits, dim=-1)

        return logits, std_list

    def clustering(self, cores, items):
        cores = F.normalize(cores)
        items = F.normalize(items)
        cates_logits = torch.mm(items, cores.t()) / self.tau
        if self.nogb:
            cates = F.softmax(cates_logits, dim=1)
        else:
            cates_dist = F.gumbel_softmax(cates_logits, tau=1)
            cates_mode = F.softmax(cates_logits, dim=1)
            if self.training:
                cates = cates_dist
            else:
                cates = cates_mode
        return cates, items

    def encode(self, input, layers):
        h = F.normalize(input)
        h = self.drop(h)

        for i, layer in enumerate(layers):
            h = layer(h)
            if i != len(layers) - 1:
                h = torch.tanh(h)
            else:  # for last layer
                mu = h[:, :self.q_dims[-1]]
                mu = F.normalize(mu)
                # logvar = h[:, self.q_dims[-1]:]
                lnvarq_sub_lnvar0 = -h[:, self.q_dims[-1]:]
                std0 = self.std
                std_q = torch.exp(0.5 * lnvarq_sub_lnvar0) * std0
        return mu, std_q, lnvarq_sub_lnvar0

    def encode_img(self, input, layers):
        h = F.normalize(input)
        h = self.drop(h)

        for i, layer in enumerate(layers):
            h = layer(h)
            if i != len(layers) - 1:
                h = torch.tanh(h)
            else:  # for last layer
                mu = h[:, :self.q_dims_i[-1]]
                mu = F.normalize(mu)
                # logvar = h[:, self.q_dims[-1]:]
                lnvarq_sub_lnvar0 = -h[:, self.q_dims_i[-1]:]
                std0 = self.std
                std_q = torch.exp(0.5 * lnvarq_sub_lnvar0) * std0
        return mu, std_q, lnvarq_sub_lnvar0

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
