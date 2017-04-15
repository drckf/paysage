import example_mnist_grbm as grbm
import example_mnist_hopfield as hopfield
import example_mnist_rbm as rbm
import example_mnist_tap_machine as tap

def all_examples():
    print("GRBM")
    grbm.example_mnist_grbm()
    print("Hopfield")
    hopfield.example_mnist_hopfield()
    print("RBM")
    rbm.example_mnist_rbm()
    print("TAP machine rbm")
    tap.example_mnist_tap_machine()
    print("Finished")

if __name__ == "__main__":
    all_examples()

