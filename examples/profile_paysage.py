import cProfile
import pstats

import example_mnist_grbm as grbm
import example_mnist_hopfield as hopfield
import example_mnist_rbm as rbm


def run_profiling(mode="short") -> None:
    """
    Execute a shortened run of example models
    """
    if mode == "short":
        rbm.example_mnist_rbm(num_epochs=3)
    else:
        grbm.example_mnist_grbm()
        hopfield.example_mnist_hopfield()
        rbm.example_mnist_rbm()

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
