import torch as th
from torch import nn
import numpy as np


class MLPExtractor(nn.Module):
    def __init__(self, observation_space, policy_kwargs):
        super(MLPExtractor,self).__init__()

        activations = {"rrelu" : nn.RReLU(0.1, 0.3), "relu" : nn.ReLU(), "elu" : nn.ELU(), "leaky" : nn.LeakyReLU(), "tanh" : nn.Tanh()}
        print(policy_kwargs)
        
        # Activation params
        bit_board_activ = activations[policy_kwargs["bit_board"]["activ"]]
        cnn_activ = activations[policy_kwargs["cnn"]["activ"]]
        value_activ = activations[policy_kwargs["value_net"]["activ"]]
        policy_activ = activations[policy_kwargs["policy_net"]["activ"]]

        # linear params
        bit_board_input = self.bit_board_size = np.prod(observation_space["bitboard"].shape)

        policy_hidden1 = policy_kwargs["policy_net"]["hidden1"]
        policy_hidden2 = policy_kwargs["policy_net"]["hidden2"]
        value_hidden1 = policy_kwargs["value_net"]["hidden1"]
        value_hidden2 = policy_kwargs["value_net"]["hidden2"]
        cnn_hidden = policy_kwargs["cnn"]["hidden"]
        bit_board_hidden = policy_kwargs["bit_board"]["hidden"]

        bit_board_output = policy_kwargs["bit_board"]["output"]
        cnn_output = policy_kwargs["cnn"]["output"]

        # Bias
        value_bias = policy_kwargs["value_net"]["bias"]
        
        # CNN params
        input_channels = observation_space["virtual_board"].shape[0]
        hidden_channels = policy_kwargs["cnn"]["hidden_channels"]
        output_channels = policy_kwargs["cnn"]["output_channels"]
        kernel_size = policy_kwargs["cnn"]["conv_kernel"]
        pool_kernel = policy_kwargs["cnn"]["pool_kernel"]
        pool_stride = policy_kwargs["cnn"]["pool_stride"]

        # Embedded network params
        input_dim = bit_board_output + cnn_output #length of feature vector for MLP
        latent_pi_dim = 4 * 8 # Output action dimensions
        latent_vf_dim = 1     # Output value dimensions

        self.bit_board_network = nn.Sequential(
                nn.Linear(bit_board_input, bit_board_hidden),
                bit_board_activ,
                nn.Linear(bit_board_hidden, bit_board_output),
                bit_board_activ,
        )

        self.value_network = nn.Sequential(
                nn.Linear(input_dim, value_hidden1, bias=value_bias),
                value_activ,
                nn.Linear(value_hidden1, value_hidden2, bias=value_bias),
                value_activ,
                nn.Linear(value_hidden2, latent_vf_dim, bias=value_bias),
        )

        self.cnn_network = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size),
                cnn_activ,
                nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride),
                cnn_activ,
                nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn_network(
                th.as_tensor(observation_space["virtual_board"].sample()[None]).float()
            ).shape[1]

        self.cnn_linear = nn.Sequential(nn.Linear(n_flatten, cnn_output), cnn_activ)


        self.output = nn.Sequential(
            nn.Linear(input_dim, policy_hidden1), 
            policy_activ,
            nn.Linear(policy_hidden1, policy_hidden2),
            policy_activ,
            nn.Linear(policy_hidden2, latent_pi_dim),
        )        
        
    def forward(self, data):
        # print('data from PPO',data)
        bit_board = data["bitboard"].float()
        virtual_board = data["virtual_board"].float()
        mask = data["mask"]

        bit_board_embedding = self.bit_board_network(bit_board.reshape(-1, self.bit_board_size))
        
        virtual_board_embedding = self.cnn_linear(self.cnn_network(virtual_board))

        final_embedding = th.concat((bit_board_embedding, virtual_board_embedding), dim=1)

        value_embedding = self.value_network(final_embedding)

        # output = self.mask_log_softmax(self.output(final_embedding), mask)
        # print(mask, output)

        logits = self.mask_log_softmax(self.output(final_embedding), mask)

        return final_embedding, logits, value_embedding

    def mask_log_softmax(self, vector, mask):
        """
        from https://github.com/allenai/allennlp/blob/b6cc9d39651273e8ec2a7e334908ffa9de5c2026/allennlp/nn/util.py#L272-L303:

        ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
        masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
        ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.

        ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
        broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
        unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
        do it yourself before passing the mask into this function.

        In the case that the input vector is completely masked, the return value of this function is
        arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
        of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
        that we deal with this case relies on having single-precision floats; mixing half-precision
        floats with fully-masked vectors will likely give you ``nans``.

        If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
        lower), the way we handle masking here could mess you up.  But if you've got logit values that
        extreme, you've got bigger problems than this.
        """
        if mask is not None:
            mask = mask.float()
            while mask.dim() < vector.dim():
                mask = mask.unsqueeze(1)
            # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
            # results in nans when the whole vector is masked.  We need a very small value instead of a
            # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
            # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
            # becomes 0 - this is just the smallest value we can actually use.
            vector = vector + (mask + 1e-45).log()
        return th.nn.functional.log_softmax(vector, dim=-1)