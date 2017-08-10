import cProfile
import pstats

import mnist_grbm as grbm
import mnist_rbm as rbm
import mnist_rbm_inmemory as inmemory


def run_profiling(mode="short") -> None:
    """
    Execute a shortened run of example models
    """
    if mode == "short":
        inmemory.run(num_epochs=3)
#        rbm.run(num_epochs=3)
#        grbm.run(num_epochs=3)
    else:
        grbm.run()
        rbm.run()

def custom_pstats() -> None:
    """
    Example showing some pstats analysis of the run
    """
    cProfile.run("run_profiling()", 'profstats')
    p = pstats.Stats('profstats')
    p.sort_stats('cumulative').print_stats(20)
    p.sort_stats('time').print_stats(20)

if __name__ == "__main__":
    run_profiling("short")
