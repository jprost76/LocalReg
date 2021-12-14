import argparse
import torch
from torchvision.utils import save_image
from utils import load_image_tensor, load_regularizer, WeightedL2, psnr_tensor, load_conv_kernel
import os

torch.manual_seed(123)

parser = argparse.ArgumentParser()
parser.add_argument("path", help="input image path")
parser.add_argument("--is_blurred", action="store_true",
                    help="use this flag if the input is already blurred. Else, input image will be blurred")
parser.add_argument("--std", type=int, default=15, help="noise standard deviation (int) in range [0-255] (default : 15)")
parser.add_argument("--model_dir", default="cnn15", help="name of the trained regularizer directory in regularizers/models_zoo (default : cnn15)")
parser.add_argument("--out", help="path to save restored image (if not specified a default path is created in results/ folder)")
parser.add_argument("--save_intermediate", action="store_true", help="path to save intermediate results (default=None)")
parser.add_argument("--kernel", type=int, default=1, help="blurring kernel index (between 1 and 12)")
parser.add_argument("--l", type=float, default=1, help="regularization parameter (default=0.1)")
parser.add_argument("--oracle", action="store_true", help="initialize the optimization with the ground truth")
args = parser.parse_args()

# optimizer step size at the first iteration
LR0 = 0.1
# optimizer step size at the last iteration
LR1 = 0.01
# number of iterations
IT_MAX = 50

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # ======================
    # Define output folders and paths
    # ======================
    img_name, ext = os.path.splitext(os.path.basename(args.path))
    model_name = args.model_dir
    # output directory
    if args.out:
        out_dir = args.out
    else:
        out_dir = os.path.join('oracle' if args.oracle else 'results',
                               'deblurring',
                               'k{}'.format(args.kernel),
                               'std{}'.format(args.std),
                               model_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    # directory to save iterations
    if args.save_intermediate:
        inter_dir = os.path.join(out_dir,  '{}_l{}'.format(img_name, args.l))
        if not os.path.exists(inter_dir):
            os.makedirs(inter_dir)

    # path of deblurred image
    out_path = os.path.join(out_dir, '{}_l{}.png'.format(img_name, args.l))
    # path of the degraded image
    blurred_path = os.path.join(out_dir, '{}_blurry.png'.format(img_name, args.l))

    # ================================
    # load image, kernel, regularizer and optimizer
    # ================================

    # k = get_gaussian_kernel(kernel_size=15, sigma=6).to(device)
    k = load_conv_kernel(args.kernel, pad=True).to(device)
    # input image
    I = load_image_tensor(args.path).to(device)
    if not args.is_blurred:
        kI = k(I)
        y = torch.clamp(kI + torch.empty_like(kI).normal_(0, args.std / 255.), min=0, max=1)
    else:
        y = I

    R = load_regularizer(args.model_dir).to(device)

    # variable to be optimized
    if args.oracle:
        xk = I.clone().requires_grad_()
    else:
        xk = y.clone().requires_grad_()

    # init optimizer and scheduler
    optim = torch.optim.Adam([xk], lr=LR0)
    g = (LR1 / LR0) ** (1 / IT_MAX)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=g)

    # init data fidelity loss
    F = WeightedL2(R.receptive_field, y.shape, k).to(device)

    # ===========================
    # Run ALR
    # ===========================
    if args.save_intermediate:
        save_image(xk, os.path.join(inter_dir, 'it_{}.png'.format(0)))
        torch.save(R(xk), os.path.join(inter_dir, 'local_reg_{}.pt'.format(0)))

    for i in range(IT_MAX):
        optim.zero_grad()
        # regularization term
        rx = R(xk).mean()
        # data fidelity
        Fxy = F(xk, y)
        # total loss
        loss = Fxy + args.l * rx
        loss.backward()
        optim.step()
        scheduler.step()
        if args.save_intermediate:
            save_image(xk, os.path.join(inter_dir, 'it_{}.png'.format(i+1)))
            torch.save(R(xk), os.path.join(inter_dir, 'local_reg_{}.pt'.format(i+1)))
        if i % 10 == 0:
            print('[{: <3}/ {}] F : {:.8f}; l.R : {:.8f}; tot : {:.8f}'.format(i + 1, IT_MAX, Fxy.item(), rx.item(),
                                                                               loss.item()))

    if not args.is_blurred:
        psnr = psnr_tensor(I, xk)
        print('psnr : {:.3f}'.format(psnr))

    # save results and noisy image
    save_image(xk, out_path)
    if not args.is_blurred:
        save_image(y, blurred_path)