import cProfile
import pstats

import example_mnist_grbm as grbm
import example_mnist_hopfield as hopfield
import example_mnist_rbm as rbm

def run_profiling() -> None:
    """
    Execute a shortened run of example models
    """
    # grbm.example_mnist_grbm(num_epochs=1)
    # hopfield.example_mnist_hopfield(num_epochs=1)
    rbm.example_mnist_rbm(num_epochs=2)

def pstats() -> None:
    """
    Example showing some pstats analysis of the run
    """
    cProfile.run("run_profiling()", 'profstats')
    p = pstats.Stats('profstats')
    p.sort_stats('cumulative').print_stats(20)
    p.sort_stats('time').print_stats(20)
    p.sort_stats('cumulative').print_callers(5)
    p.sort_stats('time').print_callers(5)

if __name__ == "__main__":
    run_profiling()
