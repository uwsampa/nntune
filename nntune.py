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


def nntune():
    fn = train('test/jmeint.data', [18, 64, 2, 2])
    try:
        with open(fn) as f:
            print(f.read())
    finally:
        os.remove(fn)


if __name__ == '__main__':
    nntune()
