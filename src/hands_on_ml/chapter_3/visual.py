import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_mnist_sample(some_digit):
    some_digit_image = some_digit.reshape(28, 28)

    plt.imshow(some_digit_image, cmap=mpl.cm.binary, interpolation='nearest')
    plt.axis('off')
    plt.show()
