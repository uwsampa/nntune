from __future__ import print_function
import subprocess
import tempfile
import os
import random


TRAIN_CMD = './train'
RECALL_CMD = './recall'


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


def recall(nnfn, datafn):
    rmse = shell([RECALL_CMD, nnfn, datafn])
    return float(rmse)


def read_data(fn):
    """Read a data file as a list of (input, output) pairs.
    """
    values = []
    with open(fn) as f:
        for line in f:
            values += line.strip().split()
    nsamples = int(values.pop(0))
    inputdim = int(values.pop(0))
    outputdim = int(values.pop(0))

    pairs = []
    pos = 0
    for i in range(nsamples):
        inputs = values[pos:pos + inputdim]
        pos += inputdim
        outputs = values[pos:pos + outputdim]
        pos += outputdim
        pairs.append(([float(n) for n in inputs], [float(n) for n in outputs]))

    return pairs


def dump_data(data, f):
    """Dump a series of (input, output) vector pairs to a file.
    """
    print(len(data), file=f)
    print(len(data[0][0]), file=f)
    print(len(data[0][1]), file=f)
    for x, y in data:
        print(' '.join(str(n) for n in x), file=f)
        print(' '.join(str(n) for n in y), file=f)


def dump_data_to_temp(data):
    """Dump the data to a temporary file. Return the filename.
    """
    fd, fn = tempfile.mkstemp()
    f = os.fdopen(fd, 'w')
    dump_data(data, f)
    f.close()
    return fn


def divide_data(pairs, proportion=0.7):
    """Given a data set (sequence of pairs), divide it into two parts.
    Return two filenames.
    """
    split_point = int(len(pairs) * proportion)
    random.shuffle(pairs)
    data1, data2 = pairs[:split_point], pairs[split_point:]

    fn1 = dump_data_to_temp(data1)
    fn2 = dump_data_to_temp(data2)

    return fn1, fn2


def evaluate(datafn, hidden_topology):
    # Read data.
    pairs = read_data(datafn)
    ninputs, noutputs = len(pairs[0][0]), len(pairs[0][1])
    topology = [ninputs] + hidden_topology + [noutputs]

    # Split into training and testing files.
    trainfn, testfn = divide_data(pairs)
    try:
        # Train.
        nnfn = train(trainfn, topology)
        try:
            # Test.
            return recall(nnfn, testfn)

        finally:
            os.remove(nnfn)
    finally:
        os.remove(trainfn)
        os.remove(testfn)


def nntune(datafn):
    print(evaluate(datafn, [64, 2]))


if __name__ == '__main__':
    nntune('test/jmeint.data')
