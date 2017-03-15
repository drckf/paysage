import example_mnist_grbm as grbm
import example_mnist_hopfield as hopfield
import example_mnist_rbm as rbm
import example_mnist_rbm_sklearn as rbm_sklearn

def all_examples():
    print("grbm")
    grbm.example_mnist_grbm()
    print("hopfield")
    hopfield.example_mnist_hopfield()
    print("rbm")
    rbm.example_mnist_rbm()
    print("rbm sklearn")
    rbm_sklearn.example_mnist_rbm_sklearn()
    print("Finished all examples")

if __name__ == "__main__":
    all_examples()

