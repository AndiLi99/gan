from generator import Generator
from discriminator import Discriminator

class GAN:
    def __init__(self, image_shape, generator_input_shape, discriminator_output_shape):
        self.image_shape = image_shape
        self.generator_input_shape = generator_input_shape
        self.discriminator_output_shape = discriminator_output_shape

        self.generator = Generator(generator_input_shape)
        self.discriminator = Discriminator(image_shape)

    def train_generator(self, epochs, step_size, mini_batch_size, training_inputs, expected_outputs):
        self.generator.stochastic_gradient_descent(epochs,
                                                   step_size,
                                                   mini_batch_size,
                                                   training_inputs,
                                                   expected_outputs,
                                                   self.discriminator)

    def train_discriminator(self, epochs, step_size, mini_batch_size, training_inputs, expected_outputs):
        self.discriminator.stochastic_gradient_descent(epochs,
                                                       step_size,
                                                       mini_batch_size,
                                                       training_inputs,
                                                       expected_outputs)

    def generate_images(self, input_list):
        imgs = []
        for i in input_list:
            imgs.append(self.generator.feed_forward(i))
        return imgs

    def add_layer_to_generator(self, layer_type, output_size, kernel_size):
        self.generator.add(layer_type, output_size, kernel_size)

    def add_layer_to_discriminator(self, layer_type, output_size, kernel_size):
        self.discriminator.add(layer_type, output_size, kernel_size)

    def generator_feed_forward (self, network_input):
        return self.generator.feed_forward(network_input)

    def discriminator_feed_forward(self, network_input):
        return self.discriminator.feed_forward(network_input)