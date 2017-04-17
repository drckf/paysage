import cProfile
import pstats

import example_mnist_grbm as grbm
import example_mnist_hopfield as hopfield
import example_mnist_rbm as rbm
import example_mnist_tap_machine as tap

def run_profiling(mode="short") -> None:
    """
    Execute a shortened run of example models
    """
    if mode == "short":
        grbm.example_mnist_grbm(num_epochs=3)
        hopfield.example_mnist_hopfield(num_epochs=3)
        rbm.example_mnist_rbm(num_epochs=3)
    else:
        grbm.example_mnist_grbm()
        hopfield.example_mnist_hopfield()
        rbm.example_mnist_rbm()
        tap.example_mnist_tap_machine()

def pstats() -> None:
    """
    Example showing some pstats analysis of the run
    """
    cProfile.run("run_profiling()", 'profstats')
    p = pstats.Stats('profstats')
    p.sort_stats('cumulative').print_stats(20)
    p.sort_stats('time').print_stats(20)

if __name__ == "__main__":
    run_profiling("short")
