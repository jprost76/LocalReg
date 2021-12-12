import torch.nn as nn
import torch

class CNN(nn.Module):
    def __init__(self, conv_params, act, r):
        """
        conv_params : [(cin, cout, kernel_size, stride), ...]
        act : non linear activation between conv layers (eg torch.nn.ReLU())
        r : receptive field (int)
        """
        super(CNN, self).__init__()
        modules = []
        for cin, cout, kernel_size, stride in conv_params[:-1]:
            modules += [nn.Conv2d(cin, cout, kernel_size, stride), act]
        # no activation in the last layer
        cin, cout, kernel_size, stride = conv_params[-1]
        modules += [nn.Conv2d(cin, cout, kernel_size, stride)]
        self.receptive_field = r
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return torch.mean(self.net(x), dim=1)
        #return self.net(x)


def load_model(params, checkpoint=None):
    """
    load the model according to user inputs.
     If checkpoint is specified, restore model parameters stored in checkpoint
    :param params: dict containing parameters of the model
    :param checkpoint : dictionnary. Must contain key 'model', containing model.state_dict()
    :return: discriminator model
    """
    discriminator = None
    if params.model == "CNN":
        act = nn.LeakyReLU() if params.act.lower() in ('leakyrelu', 'lrelu') else nn.ReLU()
        discriminator = CNN(params.conv_params, act, r=params.patch_size)

    assert discriminator is not None, "wrong model name. check \"model\" argument in params.json file"

    if checkpoint is not None:
        print('loading model state dict')
        discriminator.load_state_dict(checkpoint['model'])
    return discriminator


