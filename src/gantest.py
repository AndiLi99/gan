from gan import GAN
import mnist_loader
import numpy as np

gan = GAN((1, 28,28), (1,10,10), 2)
gan.add_layer_to_generator("deconv", (15,15), (3,5,5))
gan.add_layer_to_generator("deconv", (28,28), (3,3,3))
gan.add_layer_to_discriminator("conv", None, (3,5,5))
gan.add_layer_to_discriminator("conv", None, (3,3,3))
gan.add_layer_to_discriminator("dense", 10)
gan.add_layer_to_discriminator("soft", 2)

noise = [np.random.randn(1,10,10) for i in range(100)]
noise_set = [(n, np.array([1,0])) for n in noise]
real_images = mnist_loader.load_data_wrapper()
generated_images = [(gan.generate_image(n), np.array([0,1])) for n in noise]

training_set = []
training_set.extend(real_images[0:100])
training_set.extend(generated_images)

print np.shape(training_set)
gan.train_discriminator(1, 0.001, 50, training_set)
gan.train_generator(1, 0.001, 50, noise_set)
