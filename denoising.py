import argparse
import torch
from torchvision.utils import save_image
from utils import load_image_tensor, load_regularizer, WeightedL2, psnr_tensor
import os

torch.manual_seed(123)

parser = argparse.ArgumentParser()
parser.add_argument("path", help="input image path")
parser.add_argument("--add_noise", action="store_true", help="add noise to the input image (else consider it is already noisy)")
parser.add_argument("--std", type=int, default=25, help="noise standard deviation (int) in range [0-255] (default : 25)")
parser.add_argument("--model_dir", default="cnn15", help="name of the trained regularizer directory in regularizers/models_zoo (default : cnn15)")
parser.add_argument("--out", help="path to save restored image (if not specified a default path is created in results/ folder)")
parser.add_argument("--name", help="name of the denoised image")
parser.add_argument("--save_intermediate", action="store_true", help="save intermediate iterations")
parser.add_argument("--l", type=float, default=1, help="regularization parameter (default=1)")
parser.add_argument("--oracle", action="store_true", help="initialize the optimization with the ground truth")
parser.add_argument("--log_loss", action="store_true")
parser.add_argument("--optim", default="adam", help="optim (adam (default) or gradient descent")
parser.add_argument("--it", type=int, default=70, help="number of iterations (default : 70)")
parser.add_argument("--lr0", type=float, default=0.01, help="learning rate at the first iteration (default 0.1)")
parser.add_argument("--lr1", type=float, default=0.001, help="learning rate at the last iteration (default 0.01)")
parser.add_argument("--sqrt", action="store_true", help="take the square root of the square L2 distance")
parser.add_argument("--r2", action="store_true", help="take the square of the local regularizer")
parser.add_argument("--rabs", action="store_true", help="take the absolute value of the local regularizer")
parser.add_argument("--rb", type=float, default=-0.07)
parser.add_argument("--init_random", action="store_true")
parser.add_argument("--init_smooth", action="store_true")
args = parser.parse_args()

# optimizer step size at the first iteration
LR0 = args.lr0
# optimizer step size at the last iteration
LR1 = args.lr1
# number of iterations
IT_MAX = args.it


if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # ======================
    # Define output folders and paths
    # ======================
    img_name, ext = os.path.splitext(os.path.basename(args.path))
    model_name = os.path.basename(args.model_dir)
    # output directory
    if args.out:
        out_dir = args.out
    else:
        # noise specific directory
        out_dir = os.path.join('oracle' if args.oracle else 'results',
                               'denoising',
                               'std{}'.format(args.std),
                               args.model_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    # directory to save interations
    if args.save_intermediate:
        inter_dir = os.path.join(out_dir, '{}_l{}'.format(img_name, args.l))
        if not os.path.exists(inter_dir):
            os.makedirs(inter_dir)

    out_file = '{}'.format(args.name) if args.name else '{}_l{}'.format(img_name, args.l)
    out_path = os.path.join(out_dir, '{}.png'.format(out_file))
    noisy_path = os.path.join(out_dir, '{}_noisy.png'.format(img_name))

    # file to save losses
    if args.log_loss:
        log_path = os.path.join(out_dir, '{}_loss.csv'.format(out_file))
        flog = open(log_path, "w")
        flog.write("it;F;lr;psnr\n")

    # ================================
    # load image, kernel, regularizer and optimizer
    # ================================
    I = load_image_tensor(args.path).to(device)
    # add noise if specified
    if args.add_noise:
        y = torch.clamp(I + torch.empty_like(I).normal_(0, args.std/255.), min=0, max=1)
    else:
        y = I

    # load regularizer
    rt = load_regularizer(args.model_dir).to(device)

    # init data fidelity loss
    wl2 = WeightedL2(rt.receptive_field, y.shape).to(device)

    if args.sqrt:
        F = lambda x : torch.sqrt(wl2(x, y) + 1e-9)
    else:
        F = lambda x : wl2(x, y)

    if args.r2:
        R = lambda x : (rt(x)-args.rb)**2
    elif args.rabs:
        R = lambda x: torch.abs(rt(x) - args.rb)
    else:
        R = lambda x : rt(x)

    # variable to be optimized
    if args.oracle:
        xk = I.clone().requires_grad_()
    elif args.init_random:
        xk = torch.randn_like(y) * 0.1 + 0.5
        xk = xk.requires_grad_()
    elif args.init_smooth:
        xk = (torch.ones_like(y) * 0.5).requires_grad_()
    else:
        xk = y.clone().requires_grad_()

    # init optimizer and scheduler
    assert args.optim.lower() in ("adam", "gd", "sgd", "gradient_descent"), '--optim flag not recognized'
    if args.optim.lower() == "adam":
        optim = torch.optim.Adam([xk], lr=LR0)
    if args.optim.lower() in ("gd", "sgd", "gradient_descent"):
        optim = torch.optim.SGD([xk], lr=LR0)
    g = (LR1 / LR0) ** (1 / IT_MAX)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=g)

    # ===========================
    # Run ALR
    # ===========================
    for i in range(IT_MAX):
        optim.zero_grad()
        if args.save_intermediate:
            # store previous value
            xc = xk.clone()
            # compute grad R
            rxc = R(xc)
            grx = torch.autograd.grad(outputs=rxc.mean(), inputs=xc)
            save_image(xc, os.path.join(inter_dir, 'it_{}.png'.format(i)))
            torch.save(rxc, os.path.join(inter_dir, 'local_reg_{}.pt'.format(i)))
            torch.save(grx, os.path.join(inter_dir, 'grad_{}.pt'.format(i)))

        optim.zero_grad()
        # regularization term
        rx = R(xk).mean()
        # data fidelity
        Fxy = F(xk)
        # total loss
        loss = Fxy + args.l * rx
        if args.log_loss:
            psnr = psnr_tensor(xk, I) if args.add_noise else None
            flog.write('{};{};{};{}\n'.format(i, Fxy.item(), rx.item(), psnr))
        loss.backward()
        optim.step()
        scheduler.step()

        if i % 10 == 0:
            print('[{: <3}/ {}] F : {:.8f}; l.R : {:.8f}; tot : {:.8f}'.format(i+1, IT_MAX, Fxy.item(), rx.item(), loss.item()))
    if args.save_intermediate:
        save_image(xk, os.path.join(inter_dir, 'it_{}.png'.format(i)))
        torch.save(R(xk), os.path.join(inter_dir, 'local_reg_{}.pt'.format(i)))
        optim.zero_grad()
        grx = torch.autograd.grad(outputs=R(xk).mean(), inputs=xk)
        torch.save(grx, os.path.join(inter_dir, 'grad_{}.pt'.format(i)))

    if args.add_noise:
        psnr = psnr_tensor(I, xk)
        print('psnr : {:.3f}'.format(psnr))

    # save results and noisy image
    save_image(xk, out_path)
    if args.add_noise:
        save_image(y, noisy_path)

    if args.log_loss:
        flog.close()

