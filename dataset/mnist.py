from sklearn import datasets
import numpy as np
from .ssl_dataset import BaseSSLDataset


class MNIST(BaseSSLDataset):
    def __init__(self) -> None:
        super().__init__()
        mnist = datasets.load_digits()

        n_samples = len(mnist.images)
        self.images = mnist.images
        self.X = mnist.images.reshape((n_samples, -1))
        print(self.X[0].shape)

        self.y = mnist.target
        self.name = "MNIST"

    def make_noise(self, image):
        length = len(image)
        mean = 0
        var = 0.5
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,length)
        # gauss = gauss.reshape
        return gauss



if __name__ == "__main__":
    dataset = MNIST()

    print(dataset.get_data(40))
