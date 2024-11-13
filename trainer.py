import numpy as np
import pathlib
import torch
import matplotlib.pyplot as plt

import optimize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_sigma(centers, sigma2):
    """ Create the sigma matrix for the Gaussian kernel """
    diff = np.linalg.norm(centers[:, None] - centers[None, :], axis = 2)
    Sigma = np.exp(-diff**2 / (2 * sigma2))
    return Sigma

def save_plot(img, path, name):
    """ Save the image to the path """
    save_path = path / (name + ".png")
    plt.imshow(img, cmap = 'gray')
    plt.title(name)
    plt.savefig(save_path)
    plt.clf()

class Trainer:
    def __init__(self, images, ag, ap, sigma2_p, sigma2_g, sigma2_0, epochs = 1000, iterations = 5,
                 template_name = 'template_0', output_path = pathlib.Path()):
        self.template_name = template_name
        self.template = None

        self.ag, self.ap = ag, ap
        self.sigma2_p, self.sigma2_g = sigma2_p, sigma2_g
        self.sigma2_0 = sigma2_0

        self.images = images
        self.num_images = len(images)
        self.image_nrow, self.image_ncol = self.images[0].shape
        self.image_total = self.image_nrow * self.image_ncol
        self.images = [image.reshape(-1, 1) for image in self.images]

        IY, IX = np.meshgrid(np.arange(self.image_ncol), np.arange(self.image_nrow))
        self.all_pixels = np.c_[IX.ravel(), IY.ravel()]

        self.p_centers = self.all_pixels[np.sum(self.all_pixels, axis = 1) % 2 == 0]
        self.g_centers = self.all_pixels[np.sum(self.all_pixels, axis = 1) % 2 == 0]
        self.kp = self.p_centers.shape[0]
        self.kg = self.g_centers.shape[0]

        self.alphas = np.zeros((self.kp, 1))
        self.betas = np.zeros((self.num_images, self.kg, 2))
        self.mup = np.zeros((self.kp, 1))

        self.sigma2 = sigma2_0

        self.kBps = None

        self.sigma_p_inv = create_sigma(self.p_centers, self.sigma2_p)
        self.Sigma_g_inv = create_sigma(self.g_centers, self.sigma2_g)

        self.yy = sum([np.sum(image**2) for image in self.images]) / self.num_images
        
        self.predictions = None
        
        self.epochs = epochs
        self.iterations = iterations
        self.output_path = output_path

        diff_vector = self.all_pixels[:, None, :] - self.g_centers[None, :, :]
        self.pixels_centers = np.exp(-np.sum(diff_vector**2, axis = -1) / (2 * self.sigma2_g))
    
    def update_betas(self):
        """ Update the betas (approximation with modes, 4.2) """
        optimizer_utils = optimize.OptimizerUtils(sigma2_g = self.sigma2_g, p_centers = self.p_centers,
                                                  g_centers = self.g_centers, all_pixels = self.all_pixels,
                                                  image_nrow = self.image_nrow, image_ncol = self.image_ncol)
        optimizer = optimize.Optimizer(alphas = self.alphas, curr_beta = self.betas.copy(),
                                       g_inv = self.Sigma_g_inv,
                                       sigma2_p = self.sigma2_p, sigma2_g = self.sigma2,
                                       images = self.images.copy(),
                                       optimizer_utils = optimizer_utils)
        self.betas = optimizer.optimize_betas(self.epochs)

    def beta_beta_l(self):
        """ Compute [beta beta^T]_l (formula 12) """
        self.update_betas()
        return sum([beta @ beta.T for beta in self.betas]) / self.num_images

    def update_Gamma(self):
        """ Update Gamma (formula 13) """
        new_gamma = (self.num_images * self.beta_beta_l() + self.ag * np.linalg.inv(self.Sigma_g_inv))
        new_gamma = new_gamma / (self.num_images + self.ag)
        
        self.Gamma_Inv = np.linalg.inv(new_gamma)

    def calculate_k_p_beta(self, betas):
        """ Compute K_p^beta (4.1.2) """
        deformation = self.pixels_centers.dot(betas)
        deformed_pixel = self.all_pixels - deformation

        diff_vector = deformed_pixel[:, None, :] - self.p_centers[None, :, :]
        gaussian = np.exp(-np.sum(diff_vector**2, axis = -1) / (2 * self.sigma2_p))
        return gaussian
    def update_k_p_beta(self):
        """ Update K_p^beta for all images """
        self.kBps = [self.calculate_k_p_beta(beta) for beta in self.betas]

    def update_predictions(self):
        """ Update the predictions """
        self.predictions = [kBp.dot(self.alphas) for kBp in self.kBps]

    def ky_kk(self):
        """ Compute [K_p^beta^T Y]_l and [K_p^beta K_p^beta^T]_l (formula 14) """
        self.update_k_p_beta()

        ky = [kBp.transpose().dot(image) for kBp, image in zip(self.kBps, self.images)]
        kk = [kBp.transpose().dot(kBp) for kBp in self.kBps]

        kyl = sum(ky) / self.num_images
        kkl = sum(kk) / self.num_images

        return kyl, kkl

    def update_alpha_and_sigma2(self):
        """ Update alpha and sigma^2 (formula 16) """
        kyl, kkl = self.ky_kk()
        p_inverse = self.sigma_p_inv
        for _ in range(10):
            new_alpha = np.linalg.inv(self.num_images * kkl + self.sigma2 * p_inverse)
            new_alpha = new_alpha @ (self.num_images * kyl + self.sigma2 * (p_inverse @ self.mup))

            new_sigma2 = self.yy + self.alphas.T @ kkl @ self.alphas - 2 * self.alphas.T @ kyl
            new_sigma2 = (self.num_images * new_sigma2 + self.ap * self.sigma2_0)
            new_sigma2 = new_sigma2 / (self.num_images * self.image_total + self.ap)
            
            self.alphas = new_alpha
            self.sigma2 = new_sigma2.item()

    def train(self, iterations):
        """ Train the model """
        for i in range(iterations):
            print(f"Iteration {i + 1}/{iterations}.")

            print("Updating geometric parameters.")
            self.update_Gamma()

            print("Updating photometric parameters.\n")
            self.update_alpha_and_sigma2()

        print("Computing and saving the images.")
        self.calculate_template()
        self.update_predictions()

    def calculate_template(self):
        diff_vector = self.all_pixels[:, None, :] - self.p_centers[None, :, :]
        gaussian = np.exp(-np.sum(diff_vector**2, axis = -1) / (2 * self.sigma2_p))
        self.template = gaussian @ self.alphas

    def save_images(self):
        path = self.output_path / self.template_name
        path.mkdir(parents = True, exist_ok = True)
        
        for i in range(self.num_images):
            prediction = self.predictions[i].reshape(-1, self.image_ncol)
            image_name = "prediction_" + str(i)
            save_plot(prediction, path, image_name)

            original = self.images[i].reshape(-1, self.image_ncol)
            image_name = "original_image_" + str(i)
            save_plot(original, path, image_name)

        template = self.template.reshape(-1, self.image_ncol)
        image_name = "template"
        save_plot(template, path, image_name)