import subprocess
import tempfile
import os


TRAIN_CMD = './train'


def shell(command, cwd=None, shell=False):
    """Execute a command (via a shell or directly). Capture the stdout
    and stderr streams as a string.
    """
    return subprocess.check_output(
        command,
        cwd=cwd,
        stderr=subprocess.STDOUT,
        shell=shell,
    )


def train(datafile, topology, epochs=100, learning_rate=0.7):
    topostr = '-'.join(str(n) for n in topology)
    fd, fn = tempfile.mkstemp()
    os.close(fd)
    shell([TRAIN_CMD, datafile, topostr, str(epochs), str(learning_rate), fn])
    return fn


def get_shape(data_fn):
    """Get the number of inputs and number of outputs in the data file.
    """
    # Get at least the first three numbers in the file.
    parts = []
    with open(data_fn) as f:
        for line in f:
            parts += line.strip().split()
            if len(parts) >= 3:
                break

    return int(parts[1]), int(parts[2])


def nntune(fn):
    ninputs, noutputs = get_shape(fn)
    nnfn = train(fn, [ninputs, 64, 2, noutputs])
    try:
        with open(nnfn) as f:
            print(f.read())
    finally:
        os.remove(nnfn)


if __name__ == '__main__':
    nntune('test/jmeint.data')
