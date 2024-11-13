import torch
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_gaussian_kernel(sd2):
    cen = np.ceil(np.sqrt(sd2) * 4)
    x = np.arange(0, 2 * cen + 1)
    temp = np.exp(-(x - cen)**2 / (2 * sd2))
    return int(cen), np.outer(temp, temp)

class OptimizerUtils:
    """ Utility functions for the optimizer """
    def __init__(self, sigma2_g, p_centers, g_centers, all_pixels, image_nrow, image_ncol):
        self.padding, self.kernel = generate_gaussian_kernel(sigma2_g)
        self.torch_C_a = torch.from_numpy(p_centers).float().cuda()
        self.torch_all_pixel = torch.from_numpy(all_pixels).float().cuda()
        self.n_pixels = self.torch_all_pixel.size()[0]
        self.n_centers = self.torch_C_a.size()[0]
        self.image_nrow = image_nrow
        self.image_ncol = image_ncol
        self.g_centers = g_centers

    def beta_tensors(self,betas):
        batch_size = betas.size()[0]
        empty_tensor = torch.zeros((batch_size, 2,  self.image_nrow, self.image_ncol), device = device)
        rows, cols = self.g_centers.T
        empty_tensor[:, :, rows, cols] = torch.transpose(betas, 1, 2)

        return empty_tensor

    def convolution(self, beta_images, kernel):
        kernel = torch.from_numpy(kernel).float().to(device)
        kernel_row, kernel_col = kernel.size()
        kernel = kernel.expand(2, 1, kernel_row, kernel_col)
        out = F.conv2d(input = beta_images, weight = kernel, padding = self.padding, groups = 2)
        return out

    def get_deformation(self, res, batch_size):
        flat_res = res.reshape((batch_size, 2, -1))
        deformation = torch.transpose(flat_res, 1, 2)
        return deformation

    def get_deformation_from_conv(self, betas):
        batch_size = betas.size()[0]
        beta_img = self.beta_tensors(betas)
        res = self.convolution(beta_img, kernel=self.kernel)
        deformation = self.get_deformation(res = res, batch_size = batch_size)
        return deformation

class Optimizer:
    """ Optimizer for the beta values """
    def __init__(self, alphas, curr_beta, g_inv, sigma2_p, sigma2_g, images, optimizer_utils):
        self.optimizer_utils = optimizer_utils
        self.alphas = torch.from_numpy(alphas).float().to(device)
        self.curr_betas = torch.tensor(np.array(curr_beta), dtype=torch.float32,  device=device)
        self.image = torch.tensor(np.array(images), dtype = torch.float32, device=device)
        self.g_inv = torch.from_numpy(g_inv).float().to(device)
        self.sigma2_p = sigma2_p
        self.sigma2_g = sigma2_g

    def optimize_betas(self, iter):
        betas = self.curr_betas.clone().detach().requires_grad_(True)

        criterion = torch.nn.MSELoss(reduction = 'sum').to(device)
        optimizer = torch.optim.AdamW([betas], lr = 0.1)

        for _ in range(iter):
            deformation = self.optimizer_utils.get_deformation_from_conv(betas)
            deformed = self.optimizer_utils.torch_all_pixel - deformation
            temp = ((deformed[:, :, None, :] - self.optimizer_utils.torch_C_a[None, None, :, :])**2).sum(-1)
            pred = torch.exp(-temp / (2 * self.sigma2_p)) @ self.alphas

            loss = criterion(self.image, pred) / (2 * self.sigma2_g)

            temp = torch.einsum('nik,ij,njl->nkl', betas, self.g_inv, betas)
            loss = loss + 0.5 * torch.diagonal(temp, 0, -2, -1).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        to_numpy = betas.detach().cpu().numpy()
        list_of_numpy = list(to_numpy)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return list_of_numpy