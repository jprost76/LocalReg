import torch
import argparse
from torchvision import transforms
import os
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from PIL import Image

from models import load_model
from utils_reg import AddGaussianNoise, Params, load_optim, save_checkpoint

torch.manual_seed(123)


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=torchvision.transforms.ToTensor()):
        """
        :param root_dir: path of the image directory
        :param transform: transform to apply to the patches (must include ToTensor())
        """
        # path of the directory containing the patches
        self.root_dir = root_dir
        self.transform = transform
        # list of the patches path
        self.patches = [os.path.join(self.root_dir, fname) for fname in os.listdir(self.root_dir)]

    def __getitem__(self, index):
        path = self.patches[index]
        img = Image.open(path)
        if self.transform is not None:
            t = self.transform(img)
        return t

    def __len__(self):
        return len(self.patches)


def gradient_penalty(real_batch, fake_batch, discriminator, device):
    """
    source https://github.com/EmilienDupont/wgan-gp/blob/master/training.py
    :param real_batch:
    :param fake_batch:
    :return:
    """
    batch_size = real_batch.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(real_batch).to(device)
    interpolated = alpha * real_batch.data + (1 - alpha) * fake_batch.data
    interpolated = Variable(interpolated, requires_grad=True)
    interpolated = interpolated.to(device)
    # Calculate probability of interpolated examples
    prob_interpolated = discriminator(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                    grad_outputs=torch.ones(prob_interpolated.size(), device=device),
                                    create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gp = ((gradients_norm - 1) ** 2).mean()
    return gp


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',
                    help='images directory for training on synthetic noise(must contain subdirectory train/ and val/).'
                         'AWGN is added to create noisy images')
parser.add_argument("--clean",
                    help="directory of clean reference images for training and validation (must contain subdirectory train/ and val/)")
parser.add_argument("--noisy", help="directory of noisy images for training and validation")
parser.add_argument('--model_dir', default='models_zoo/small',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None, help="load checkpoint (best or last)")  # 'best' or 'train'
parser.add_argument('--use_cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--workers', type=int, default=1, help="number of workers to load data")


def init_data_loaders(argt, paramt):
    """
    initiate loaders for synthetic and real data
    """
    assert argt.data_dir or (
                argt.clean and argt.noisy), "must specified data directory, in --data_dir, or --clean and --noisy"
    tf_clean = transforms.Compose([
        transforms.RandomCrop(size=paramt.patch_size),
        transforms.RandomVerticalFlip(),
        # ToNumpy(),
        # transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    if argt.data_dir:
        # use synthetic noise
        path_train_clean = os.path.join(argt.data_dir, 'train')
        path_train_noisy = os.path.join(argt.data_dir, 'train')
        path_val_clean = os.path.join(argt.data_dir, 'val')
        path_val_noisy = os.path.join(argt.data_dir, 'val')
        # add AWGN to generate noisy data
        tf_noisy = transforms.Compose([
            tf_clean,
            AddGaussianNoise(0., paramt.noise_std)
        ])
    else:  # argt.clean and argt.noisy
        path_train_clean = os.path.join(argt.clean, 'train')
        path_train_noisy = os.path.join(argt.noisy, 'train')
        path_val_clean = os.path.join(argt.clean, 'val')
        path_val_noisy = os.path.join(argt.noisy, 'val')
        # noise is already here
        tf_noisy = tf_clean

    dataset_train_clean = ImageDataset(path_train_clean, transform=tf_clean)
    dataset_train_noisy = ImageDataset(path_train_noisy, transform=tf_noisy)
    dataset_val_clean = ImageDataset(path_val_clean, transform=tf_clean)
    dataset_val_noisy = ImageDataset(path_val_noisy, transform=tf_noisy)

    loader_train_clean = DataLoader(dataset_train_clean, batch_size=paramt.batch_size,
                                    shuffle=True,
                                    num_workers=argt.workers // 2 - 1,
                                    pin_memory=(device == torch.device('cuda')),
                                    drop_last=True)
    loader_train_noisy = DataLoader(dataset_train_noisy, batch_size=paramt.batch_size,
                                    shuffle=True,
                                    num_workers=argt.workers // 2 - 1,
                                    pin_memory=(device == torch.device('cuda')),
                                    drop_last=True)

    loader_val_clean = DataLoader(dataset_val_clean, batch_size=paramt.batch_size,
                                  shuffle=True,
                                  num_workers=argt.workers // 2 - 1,
                                  pin_memory=(device == torch.device('cuda')),
                                  drop_last=True)
    loader_val_noisy = DataLoader(dataset_val_noisy, batch_size=paramt.batch_size,
                                  shuffle=True,
                                  num_workers=argt.workers // 2 - 1,
                                  pin_memory=(device == torch.device('cuda')),
                                  drop_last=True)
    return loader_train_clean, loader_train_noisy, loader_val_clean, loader_val_noisy


if __name__ == "__main__":
    # ==================
    # load models, dataset, optim ...
    # ==================
    args = parser.parse_args()
    if args.use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(device)
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    loader_train_clean, loader_train_noisy, loader_val_clean, loader_val_noisy = init_data_loaders(args, params)

    if args.restore_file is not None:
        print('loading checkpoint')
        checkpoint = torch.load(os.path.join(args.model_dir, 'checkpoint_{}.pt'.format(args.restore_file)))
        current_epoch = checkpoint['current_epoch']
    else:
        checkpoint = None
        current_epoch = 0
    discriminator = load_model(params, checkpoint).to(device)
    optim = load_optim(params, discriminator, checkpoint)

    # use lr_scheduler if specified
    if 'lr_decay' in params.dict:
        # lr_decay = learning rate at the last epoch
        # parameter of exponential lr decay lr(t+1) = g*lr(t)
        g = (params.lr_decay / params.learning_rate) ** (1 / params.epochs)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=g)

    discriminator.eval()

    # ================
    # Training loop
    # ================
    # best total loss
    best = 1000000
    # iterator over noisy data
    noisy_iter = iter(loader_train_noisy)
    step = 0
    # loss sum
    training_log = {'lx': 0, 'ly': 0, 'gp': 0, 'loss': 0}
    for epoch in range(current_epoch, params.epochs):
        discriminator.train()
        # iterate over training data
        for x in loader_train_clean:
            step += 1
            # load noisy batch
            try:
                y = next(noisy_iter)
            except StopIteration:
                noisy_iter = iter(loader_train_noisy)
                y = next(noisy_iter)
            x = x.to(device)
            y = y.to(device)
            discriminator.zero_grad()
            lx = discriminator(x).mean()
            ly = discriminator(y).mean()
            gp = params.mu * gradient_penalty(x, y, discriminator, device)
            loss = lx - ly + gp
            # update average
            batch_size = x.shape[0]
            training_log['lx'] += lx.item() / batch_size
            training_log['ly'] += ly.item() / batch_size
            training_log['gp'] += gp.item() / batch_size
            training_log['loss'] += loss.item() / batch_size
            loss.backward()
            optim.step()
            # save checkpoint
            if step % params.checkpoint_step == 0:
                # save checkpoint
                is_best = training_log['loss'] < best
                if is_best:
                    best = training_log['loss']
                print('saving checkpoint, is_best = {}'.format(is_best))
                save_checkpoint(args.model_dir, discriminator, optim, epoch + 1, is_best=is_best)
            if step % params.log_step == 0:
                print(
                    'Epoch [{}/{}] ; step : {}; dx : {:.4f} ; dy : {:.4f} ; gp : {:.4f} ; loss : {:.4f} '.format(epoch,
                                                                                                                 params.epochs,
                                                                                                                 step,
                                                                                                                 training_log[
                                                                                                                     'lx'],
                                                                                                                 training_log[
                                                                                                                     'ly'],
                                                                                                                 training_log[
                                                                                                                     'gp'],
                                                                                                                 training_log[
                                                                                                                     'loss'])
                    )
                # reset log
                training_log = {'lx': 0, 'ly': 0, 'gp': 0, 'loss': 0}
        scheduler.step()
        # compute validation loss
        if epoch % params.val_epoch == 0:
            val_log = {'lx': 0, 'ly': 0, 'gp': 0, 'd': 0}
            discriminator.eval()
            iter_val = iter(loader_val_noisy)
            # iterate over validation dataset
            for x in loader_val_clean:
                try:
                    y = next(iter_val)
                except StopIteration:
                    iter_val = iter(loader_val_noisy)
                    y = next(iter_val)
                x = x.to(device)
                y = y.to(device)
                with torch.no_grad():
                    lx = discriminator(x).mean()
                    ly = discriminator(y).mean()
                gp = params.mu * gradient_penalty(x, y, discriminator, device)
                loss = lx - ly + gp
                # update average
                batch_size = x.shape[0]
                val_log['lx'] += lx.item() / batch_size
                val_log['ly'] += ly.item() / batch_size
                val_log['gp'] += gp.item() / batch_size
            print('Val      [{}/{}] ; dx : {:.4f} ; dy : {:.4f} ; gp : {:.4f} ; tot : {:.4f} '.format(epoch,
                                                                                                      params.epochs,
                                                                                                      val_log['lx'],
                                                                                                      val_log['ly'],
                                                                                                      val_log['gp'],
                                                                                                      training_log[
                                                                                                          'loss'])
                  )
