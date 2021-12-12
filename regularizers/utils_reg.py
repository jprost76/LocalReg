import torch
import json
import os

class AddGaussianNoise(object):
    """
    add a gaussian noise to the patch and clamp the result to keep it between 0 and 1
    eg : process = transforms.Compose([transforms.ToTensor(), AddGaussianNoise(0,1)
    """
    def __init__(self, mean=0., std=1.):
        self.std = std # must be a float, or a range
        self.mean = mean

    def __call__(self, tensor):
        if type(self.std) == float:
            s = self.std
        else:
            s = torch.empty(1).uniform_(*self.std).item()
        noisy = tensor + torch.empty_like(tensor).normal_(self.mean, s)
        return noisy

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


def load_optim(params, model, checkpoint=None):
    """
    load optimizer according to specified parameters.
    If checkpoint is specified, restore optim parameters stored checkpoint
    :param params: dict containing parameters of the model
    :param checkpoint : dictionnary. Must contain key 'optim', containing optim.state_dict()
    :param model_dir: if specified, load checkpoint_last.pt state dictionnary
    :return: optimizer
    """
    if params.optim in ('adam', 'Adam', 'ADAM'):
        optim = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    else:
        # use same params as in DnCNN
        optim = torch.optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9)
    if checkpoint is not None:
        optim.load_state_dict(checkpoint['optim'])
    return optim


def save_checkpoint(dir, model, optim, epoch, is_best=False):
    state = {
        'model': model.state_dict(),
        'optim': optim.state_dict(),
        'current_epoch': epoch
    }
    torch.save(state, os.path.join(dir, 'checkpoint_last.pt'))
    if is_best:
        torch.save(state, os.path.join(dir, 'checkpoint_best.pt'))



