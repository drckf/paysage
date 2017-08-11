import mnist_grbm as grbm
import mnist_hopfield as hopfield
import mnist_rbm as rbm
import mnist_tap as tap

def all_examples():
    print("GRBM")
    grbm.run()
    print("Hopfield")
    hopfield.run()
    print("RBM")
    rbm.run()
    print("TAP machine rbm")
    tap.run()
    print("Finished")

if __name__ == "__main__":
    all_examples()

