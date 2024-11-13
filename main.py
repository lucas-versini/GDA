import trainer

import pathlib
import matplotlib.pyplot as plt

sigma2_p = 3
sigma2_g = 3
ag = 0.5
ap = 100
sigma2_0 = 1 # Initial value of sigma^2 (sigma_0^2)
epochs = 1000 # Steps for optimization on beta
iterations = 5 # Iterations for the EM algorithm

def load_train_images(template_path = pathlib.Path()):
    """ Load the training images """
    image_folder = template_path / "train"
    images_list = [plt.imread(image_path) for image_path in image_folder.glob("*.png")]
    return images_list

def train():
    # Create a new directory for the training output
    counter = 0
    output_path = pathlib.Path(__file__).parent / "output"
    while output_path.is_dir():
        output_path = pathlib.Path(__file__).parent / ("output_" + str(counter))
        counter += 1

    # Train for each template
    data_path = pathlib.Path(__file__).resolve().parent / 'data'
    for template_path in data_path.glob('template*'):
        template_name = template_path.stem
        print(f"\nTraining for {template_name}")
        trainer_ = trainer.Trainer(images = load_train_images(template_path = template_path),
                                   ag = ag, ap = ap,
                                   sigma2_p = sigma2_p, sigma2_g = sigma2_g, sigma2_0 = sigma2_0,
                                   epochs = epochs, iterations = iterations,
                                   template_name = template_name,
                                   output_path = output_path)
        trainer_.train(iterations)
        trainer_.save_images()

if __name__ == '__main__':
    train()