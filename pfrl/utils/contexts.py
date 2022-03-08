from contextlib import contextmanager


@contextmanager
def evaluating(net):
    """Temporarily switch to evaluation mode."""
    istrain = net.train
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()
