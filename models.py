import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MultiDAE(nn.Module):
    """
    Container module for Multi-DAE.

    Multi-DAE : Denoising Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, q_dims=None, dropout=0.5):
        super(MultiDAE, self).__init__()
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        self.dims = self.q_dims + self.p_dims[1:]
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(self.dims[:-1], self.dims[1:])])
        self.drop = nn.Dropout(dropout)
        
        self.init_weights()
    
    def forward(self, input):
        h = F.normalize(input)
        h = self.drop(h)

        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != len(self.layers) - 1:
                h = F.tanh(h)
        return h

    def init_weights(self):
        for layer in self.layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


class MultiVAE(nn.Module):
    """
    Container module for Multi-VAE.

    Multi-VAE : Variational Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, q_dims=None, dropout=0.5, vocab_size=None, tot_items=None):
        super(MultiVAE, self).__init__()
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1] # make the copy of the list in reverse order

        # Last dimension of q- network is for mean and variance
        # :-1 everything except last one. -1 last one
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2] # same as q_dims but last element *2
        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])
        
        # encode title
        self.hidden_dim = 100
        self.t_dims = [102*300] + self.q_dims[1:]
        temp_t_dims = [self.t_dims[0]] + [self.q_dims[1]] + [self.q_dims[-1] * 2]
        self.t_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_t_dims[:-1], temp_t_dims[1:])])

        # lstm for title words
        self.char_embeds = nn.Embedding(vocab_size, self.hidden_dim)
        self.init_embedding(self.char_embeds.weight)
        self.char_lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, bidirectional=True)
        self.init_lstm(self.char_lstm)

        # lstm encoding for item
        self.word_embeds = nn.Embedding(tot_items, self.hidden_dim)

        # direct concatenate
        self.hidden_dim = 100
        self.a_dims = [102*300+tot_items] + self.q_dims[1:]
        temp_a_dims = [self.a_dims[0]] + [self.q_dims[1]] + [self.q_dims[-1] * 2]
        self.a_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_a_dims[:-1], temp_a_dims[1:])])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

        self.experts = ProductOfExperts()
    
    def forward(self, input, purchased_items, title_mask, title_length, d):
        # process title words
        chars_embeds = self.char_embeds(title_mask).transpose(0, 1)  # title_mask: itemxword; wordxitemx100
        packed = torch.nn.utils.rnn.pack_padded_sequence(chars_embeds, title_length)
        lstm_out, _ = self.char_lstm(packed)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)  # outputs: wordxitemx200
        outputs = outputs.transpose(0, 1)  # itemxwordx200
        chars_embeds_temp = torch.FloatTensor(torch.zeros((outputs.size(0), outputs.size(2)))).to(device)
        for i, index in enumerate(output_lengths):
            chars_embeds_temp[i] = torch.cat((outputs[i, index - 1, :self.hidden_dim],
                                              outputs[i, 0, self.hidden_dim:]))
        chars_embeds = chars_embeds_temp.clone()  # itemx200
        for i in range(chars_embeds.size(0)):
            chars_embeds[d[i]] = chars_embeds_temp[i]

        embeds = self.word_embeds(purchased_items)  # itemx100
        concat_embeds = torch.cat((embeds, chars_embeds), 1)  # itemx300
        concat_embeds = self.drop(concat_embeds)  # itemx300

        # mu_t, logvar_t = self.encode_title(concat_embeds.view(1, -1))
        #
        # # seq input
        # mu_s, logvar_s = self.encode(input)

        # # POE
        # mu = torch.cat((mu_t, mu_s), dim=0)  # 2x100
        # logvar = torch.cat((logvar_t, logvar_s), dim=0)  # 2x100
        # mu, logvar = self.experts(mu, logvar)
        # mu = mu.unsqueeze(0)
        # logvar = logvar.unsqueeze(0)

        # direct concatenate
        combined = torch.cat((input, concat_embeds.view(1, -1)), dim=1)
        mu, logvar = self.encode_all(combined)

        # hidden emb
        z = self.reparameterize(mu, logvar)

        return self.decode(z), mu, logvar
    
    def encode(self, input):
        h = F.normalize(input)
        h = self.drop(h)
        
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = torch.tanh(h)
            else: # for last layer
                mu = h[:, :self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1]:]
        return mu, logvar

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

    def encode_all(self, input):
        h = F.normalize(input)
        h = self.drop(h)

        for i, layer in enumerate(self.a_layers):
            h = layer(h)
            if i != len(self.a_layers) - 1:
                h = torch.tanh(h)
            else:  # for last layer
                mu = h[:, :self.a_dims[-1]]
                logvar = h[:, self.a_dims[-1]:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = torch.tanh(h)
        return h

    def init_weights(self):
        for layer in self.q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.t_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.a_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

    def init_embedding(self, input_embedding):
        """
        Initialize embedding
        """
        bias = np.sqrt(3.0 / input_embedding.size(1))
        nn.init.uniform_(input_embedding, -bias, bias)

    def init_lstm(self, input_lstm):
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


def loss_function(recon_x, x, mu, logvar, anneal=1.0):
    # BCE = F.binary_cross_entropy(recon_x, x)
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    return BCE + anneal * KLD


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