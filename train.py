# coding=utf-8
from __future__ import print_function
import argparse
import os
from dataset import fusiondata
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.optim as optim
from net import FusionNetwork
from net import AuxiliaryDecoder
from loss import SimMaxLoss, SimMinLoss
import kornia



def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Training settings
parser = argparse.ArgumentParser(description='Image Fusion Network Implementation')
parser.add_argument('--dataset', type=str, default='data', help='dataset name')
parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--input_nc', type=int, default=1, help='input image channels')
parser.add_argument('--output_nc', type=int, default=1, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.0001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=150, help='weight on L1 term in objective')
parser.add_argument('--alpha', type=int, default=0.25, help='alpha parameter value')
parser.add_argument('--ema_decay', type=float, default=0.9, help='ema_decay parameter')
opt = parser.parse_args()

use_cuda = not opt.cuda and torch.cuda.is_available()

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

device = torch.device("cuda" if use_cuda else "cpu")

print('===> Loading datasets')
root_path = "data/"
dataset = fusiondata(os.path.join(root_path, opt.dataset))
training_data_loader = DataLoader(dataset=dataset, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

print('===> Building model')
model = FusionNetwork().cuda()
aux_model = AuxiliaryDecoder().cuda()


def update_ema_variables(model, ema_model, alpha, global_step):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


# Loss functions
SSIMLoss = kornia.losses.SSIMLoss(3, reduction='mean')
criterion = [
    SimMaxLoss(metric='cos', alpha=opt.alpha).cuda(),
    SimMinLoss(metric='cos').cuda(),
    SimMaxLoss(metric='cos', alpha=opt.alpha).cuda()
]

# Optimizers
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
optimizer1 = optim.Adam(aux_model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20, 40, 60,80], gamma=0.5)

print('---------- Networks initialized -------------')
print('-----------------------------------------------')

loss_plot1 = []


def M_v(IR, VIS):
    """
    Generate an adaptive mask based on IR and VIS inputs
    """
    # Calculate basic saliency features
    ir_mean = torch.mean(IR)
    ir_std = torch.std(IR)

    diff = IR - VIS
    diff_mean = torch.mean(diff)
    diff_std = torch.std(diff)

    # Adaptive weight calculation based on saliency
    diff_saliency = torch.sigmoid((diff - diff_mean) / diff_std)
    ir_saliency = torch.sigmoid((IR - ir_mean) / ir_std)

    # Enhanced interaction
    enhanced_saliency = (ir_saliency * (1 + diff_saliency) + diff_saliency * (1 + ir_saliency)) / 2

    # Adaptive threshold
    saliency_mean = torch.mean(enhanced_saliency)
    saliency_std = torch.std(enhanced_saliency)
    adaptive_thresh = saliency_mean + saliency_std

    # Final mask generation
    final_mask = (enhanced_saliency > adaptive_thresh).float()

    return final_mask


class IVIFLoss(nn.Module):
    """
    Infrared-Visible Image Fusion Loss
    """

    def __init__(self, lambda1=1.0, lambda2=1, lambda3=0.5, device='cuda'):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        self.sobel_x = nn.Parameter(torch.Tensor([[1, 0, -1],
                                                  [2, 0, -2],
                                                  [1, 0, -1]]).view(1, 1, 3, 3).to(device),
                                    requires_grad=False)
        self.sobel_y = nn.Parameter(torch.Tensor([[1, 2, 1],
                                                  [0, 0, 0],
                                                  [-1, -2, -1]]).view(1, 1, 3, 3).to(device),
                                    requires_grad=False)

    def gradient_loss(self, img1, img2):
        eps = 1e-6

        grad1_x = F.conv2d(img1, self.sobel_x, padding=1)
        grad2_x = F.conv2d(img2, self.sobel_x, padding=1)
        grad1_y = F.conv2d(img1, self.sobel_y, padding=1)
        grad2_y = F.conv2d(img2, self.sobel_y, padding=1)

        grad1 = torch.sqrt(grad1_x.pow(2) + grad1_y.pow(2) + eps)
        grad2 = torch.sqrt(grad2_x.pow(2) + grad2_y.pow(2) + eps)

        return F.l1_loss(grad1, grad2)

    def frequency_loss(self, img1, img2, block_size=8, alpha=1.0, beta=1.0):
        """
        FFT-based saliency texture loss
        """
        # FFT for both images
        fft1 = torch.fft.rfft2(img1, norm='ortho')
        fft2 = torch.fft.rfft2(img2, norm='ortho')

        # Calculate spectrum magnitude
        amp1 = torch.log(torch.abs(fft1) + 1e-8)
        amp2 = torch.log(torch.abs(fft2) + 1e-8)

        # Block the spectrum and calculate average energy
        blocks1 = F.avg_pool2d(amp1, block_size)
        blocks2 = F.avg_pool2d(amp2, block_size)

        # Calculate energy difference between blocks
        diff = blocks1 - blocks2

        # Calculate saliency weights for each block
        weights = (torch.abs(diff) - torch.mean(torch.abs(diff))) / torch.std(torch.abs(diff))
        weights = torch.sigmoid(weights)

        # Weighted energy difference
        weighted_diff = weights * diff
        prominent_loss = torch.mean(weighted_diff ** 2)

        # Calculate suppression loss
        suppress_loss = torch.mean((1 - weights) * (diff ** 2))

        # Final loss is weighted sum
        loss = alpha * prominent_loss + beta * suppress_loss
        return loss

    def forward(self, Fused, IR, VIS):
        # Texture enhancement loss (gradient alignment with VIS)
        texture_loss = SSIMLoss(Fused, VIS) + SSIMLoss(Fused, IR)

        # Total loss
        total_loss = self.lambda2 * texture_loss

        return total_loss


class SaliencySelfsupervisedLoss(nn.Module):
    """
    Self-supervised loss for saliency maps
    """

    def __init__(self, device='cuda'):
        super().__init__()
        # Sobel operators
        self.sobel_x = nn.Parameter(torch.Tensor([[1, 0, -1],
                                                  [2, 0, -2],
                                                  [1, 0, -1]]).view(1, 1, 3, 3).to(device),
                                    requires_grad=False)
        self.sobel_y = nn.Parameter(torch.Tensor([[1, 2, 1],
                                                  [0, 0, 0],
                                                  [-1, -2, -1]]).view(1, 1, 3, 3).to(device),
                                    requires_grad=False)

    def structure_consistency(self, M, IR):
        """Structure consistency loss: M gradient should correlate with IR gradient"""
        # Calculate M gradient
        m_grad_x = F.conv2d(M, self.sobel_x, padding=1)
        m_grad_y = F.conv2d(M, self.sobel_y, padding=1)
        m_grad = torch.sqrt(m_grad_x.pow(2) + m_grad_y.pow(2) + 1e-6)

        # Calculate IR gradient
        ir_grad_x = F.conv2d(IR, self.sobel_x, padding=1)
        ir_grad_y = F.conv2d(IR, self.sobel_y, padding=1)
        ir_grad = torch.sqrt(ir_grad_x.pow(2) + ir_grad_y.pow(2) + 1e-6)

        # Calculate structure consistency
        norm_m = m_grad / (torch.max(m_grad) + 1e-6)
        norm_ir = ir_grad / (torch.max(ir_grad) + 1e-6)
        struct_loss = F.mse_loss(norm_m, norm_ir)

        return struct_loss

    def region_saliency(self, M, IR):
        """Region saliency loss: high temperature regions should have higher M values"""
        # Get mask for high temperature regions in IR
        temp_thresh = torch.mean(IR) + torch.std(IR)
        temp_mask = (IR > temp_thresh).float()

        # Calculate M activation level in high temperature regions
        region_loss = -torch.mean(M * temp_mask)  # Negative sign to encourage higher M values
        return region_loss

    def texture_guidance(self, M, VIS):
        """Texture guidance loss: texture-rich regions should get more attention"""
        # Calculate VIS texture complexity
        vis_grad_x = F.conv2d(VIS, self.sobel_x, padding=1)
        vis_grad_y = F.conv2d(VIS, self.sobel_y, padding=1)
        texture_map = torch.sqrt(vis_grad_x.pow(2) + vis_grad_y.pow(2) + 1e-6)
        texture_mask = (texture_map > torch.mean(texture_map)).float()

        # M should have stronger response in texture-rich regions
        texture_loss = -torch.mean(M * texture_mask)
        return texture_loss

    def forward(self, M, IR, VIS):
        # 1. Structure consistency constraint
        struct_loss = self.structure_consistency(M, IR)

        # 2. Region saliency constraint
        region_loss = self.region_saliency(M, IR)

        # 3. Texture guidance constraint
        texture_loss = self.texture_guidance(M, VIS)

        # Weighted sum with adjustable weights
        total_loss = struct_loss + 1 * region_loss + 0.3 * texture_loss

        return total_loss


class FCR(nn.Module):
    """
    Frequency Contrastive Regularization
    """

    def __init__(self, ablation=False):
        super(FCR, self).__init__()
        self.l1 = nn.L1Loss()
        self.multi_n_num = 2

    def forward(self, mask, a, IR, VIS):
        a_fft = torch.fft.fft2(a * mask)
        a1_fft = torch.fft.fft2(a * (1 - mask))
        p1_fft = torch.fft.fft2(IR * mask)
        p2_fft = torch.fft.fft2(VIS * (1 - mask))
        n1_fft = torch.fft.fft2(IR * (1 - mask))
        n2_fft = torch.fft.fft2(VIS * mask)

        loss = (F.l1_loss(a_fft, p1_fft) + F.l1_loss(a1_fft, p2_fft)) / \
               (F.l1_loss(a_fft, n1_fft) + F.l1_loss(a_fft, n2_fft) +
                F.l1_loss(a1_fft, n1_fft) + F.l1_loss(a1_fft, n2_fft) + 1e-10)
        return loss

    def mask_loss(self, mask, a, IR, VIS):
        a_fft = torch.fft.fft2(a * mask)
        a1_fft = torch.fft.fft2(a * (1 - mask))
        p1_fft = torch.fft.fft2(IR * mask)
        p2_fft = torch.fft.fft2(VIS * (1 - mask))
        loss = F.l1_loss(a_fft, p1_fft) + F.l1_loss(a1_fft, p2_fft)
        return loss


# Initialize loss functions
total_loss = IVIFLoss()
M_loss = SaliencySelfsupervisedLoss()
ctr_loss = FCR()


def train(e, k):
    """
    Training function

    Args:
        e: Current epoch
        k: Weight parameter for loss balance
    """
    o = 0

    for iteration, batch in enumerate(training_data_loader, 1):
        imgA, imgB = batch[0], batch[1]
        IR = imgA.to(device)
        VIS = imgB.to(device)

        Fused, fea_M,_ = model(VIS, IR)
        mask = M_v(IR, VIS)
        M = aux_model(fea_M)

        # Combined loss with weighted components
        loss = 1 * ctr_loss(mask, Fused, IR, VIS) + k * total_loss(Fused, IR, VIS) + M_loss(M, IR, VIS)

        optimizer.zero_grad()
        optimizer1.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer1.step()

        o += 1

        print('Epoch: {} Iteration: {}, Loss: {:.6f}'.format(e, o, loss.item()))

        if e % 10 == 0:
            net_g_model_out_path = "./Trained/{}stop_{}.pth".format(k, e)
            torch.save(model, net_g_model_out_path)


if __name__ == '__main__':
    k = [5]
    for j in k:
        for epoch in range(100):
            train(epoch + 1, j)
            scheduler.step()
            print('Completed epoch: {}'.format(epoch + 1))