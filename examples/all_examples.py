import mnist_grbm as grbm
import mnist_hopfield as hopfield
import mnist_rbm as rbm
import mnist_dbm as deep
import mnist_tap as tap
import mnist_relu_rbm as relu
import mnist_student_rbm as student

def all_examples():
    print("GRBM")
    grbm.run()
    print("Hopfield")
    hopfield.run()
    print("RBM")
    rbm.run()
    print("Deep RBM")
    deep.run()
    print("TAP machine rbm")
    tap.run()
    print("ReLU rbm")
    relu.run()
    print("Student rbm")
    student.run()
    print("Finished")

if __name__ == "__main__":
    all_examples()

