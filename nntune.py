#!/usr/bin/env python

from __future__ import print_function
import subprocess
import tempfile
import os
import random
import logging
import collections
import sys
import argparse
import csv
import datetime
import math

# This needs to be modified to point to the fann shared resource dir
# e.g. should be pointing to /usr/local/lib if installed locally
FANN_LIB_DIR = '../fann-snnap/src'

TRAIN_CMD = './train'
RECALL_CMD = './recall'

# Output file paths
LOG_FILE = 'nntune.log'
CSV_FILE = 'output.csv'

# Define Defaults Here
DEFAULT_REPS                = 4     # Number of times we are training the same NN
DEFAULT_EPOCHS              = 1000
DEFAULT_LEARNING_RATE       = 0.2
DEFAULT_TEST_RATIO          = 0.7
DEFAULT_TOPO_EXPONENTIAL    = True  # Set to true if number of neurons should increase exponentially
DEFAULT_TOPO_LIN_INCR       = 5     # If above is set to False, defines the step size
DEFAULT_TOPO_MAX_LAYERS     = 1
DEFAULT_TOPO_MAX_NEURONS    = 8
DEFAULT_ERROR_MODE          = 0     # 0 for MSE, 1 for classification
DEFAULT_PRECISION           = 0     # 0 for float, anything else: fixed

def get_params():
    params = []
    params.append(["timestamp", datetime.datetime.now()])
    params.append(["reps", DEFAULT_REPS])
    params.append(["epochs", DEFAULT_EPOCHS])
    params.append(["learning rate", DEFAULT_LEARNING_RATE])
    params.append(["test ratio", DEFAULT_TEST_RATIO])
    params.append(["topo exponential", DEFAULT_TOPO_EXPONENTIAL])
    params.append(["topo lin incr", DEFAULT_TOPO_LIN_INCR])
    params.append(["topo max layers", DEFAULT_TOPO_MAX_LAYERS])
    params.append(["topo max neurons", DEFAULT_TOPO_MAX_NEURONS])
    params.append(["topo error mode", DEFAULT_ERROR_MODE])
    return params

def shell(command, cwd=None, shell=False):
    """Execute a command (via a shell or directly). Capture the stdout
    and stderr streams as a string.
    """
    my_env = os.environ
    if (my_env.get("LD_LIBRARY_PATH")):
        my_env["LD_LIBRARY_PATH"] = FANN_LIB_DIR + ":" + my_env["LD_LIBRARY_PATH"]
    else:
        my_env["LD_LIBRARY_PATH"] = FANN_LIB_DIR

    # Setup environment correctly
    outstr = subprocess.check_output(
        command,
        cwd=cwd,
        stderr=subprocess.STDOUT,
        shell=shell,
        env=my_env
    )
    logging.info( str(outstr) )
    return outstr


def train(datafile, topology, epochs=DEFAULT_EPOCHS, learning_rate=DEFAULT_LEARNING_RATE):
    topostr = '-'.join(str(n) for n in topology)
    fd, fn = tempfile.mkstemp()
    os.close(fd)
    shell([TRAIN_CMD, datafile, topostr, str(epochs), str(learning_rate), str(DEFAULT_PRECISION), fn])
    return fn


def recall(nnfn, datafn, error_mode=DEFAULT_ERROR_MODE):
    rmse = shell([RECALL_CMD, nnfn, datafn, str(error_mode)])
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


def divide_data(pairs, proportion=DEFAULT_TEST_RATIO):
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
    topology = [ninputs] + list(hidden_topology) + [noutputs]

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


def increment_topo(topo, index, max_neurons, logSearch=DEFAULT_TOPO_EXPONENTIAL, incr=DEFAULT_TOPO_LIN_INCR):
    if (logSearch):
        topo[index] /= 2
    else:
        topo[index] -= incr
    if topo[index] < 1:
        if index == 0:
            return True
        else:
            topo[index] = max_neurons
            return increment_topo(topo, index - 1, max_neurons)
    else:
        return False


def exhaustive_topos(max_layers=DEFAULT_TOPO_MAX_LAYERS, max_neurons=DEFAULT_TOPO_MAX_NEURONS):
    for layers in range(max_layers, 0, -1):
        topo = [max_neurons] * layers
        while True:
            yield tuple(topo)
            if increment_topo(topo, layers - 1, max_neurons):
                break


def nntune_sequential(datafn, csvpath):
    min_error = None
    min_topo = None
    experiments = [] # experiments results

    for topo in exhaustive_topos():
        errors = []
        for i in range(DEFAULT_REPS):
            logging.info('testing {}, rep {}'.format('-'.join(map(str, topo)),
                                                     i + 1))
            error = evaluate(datafn, topo)
            logging.debug('error: {}'.format(error))
            errors.append(error)
        average_error = sum(errors) / DEFAULT_REPS
        logging.info('average error: {}'.format(average_error))

        experiments.append({"topo":topo, "error":average_error})

        if min_error is None or average_error < min_error:
            logging.debug('new best')
            min_error = average_error
            min_topo = topo

    logging.info('best topo: {}'.format('-'.join(map(str, min_topo))))
    logging.info('error: {}'.format(min_error))

    # Read training data params
    input_neurons = 0
    output_neurons = 0
    with open(datafn, 'r') as f:
        params = f.readline().rstrip().split(" ")
        input_neurons = params[1]
        output_neurons = params[2]
    # Prepare CSV data
    csv_data = get_params()
    for t in experiments:
        topo = t["topo"]
        topo_str = '-'.join(map(str, topo))
        madds = 0
        for i, hidden_neurons in enumerate(topo):
            if (i==0):
                madds += int(input_neurons)*int(hidden_neurons)
            else:
                madds += int(topo[i-1])*int(topo[i])
        madds += int(topo[len(topo)-1])*int(output_neurons)
        csv_data.append([topo_str, madds, t["error"]])
    # Dump to CSV
    with open(csvpath, 'wb') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        for line in csv_data:
            wr.writerow(line)


def nntune_cw(datafn, clusterworkers, csvpath):
    import cw.client
    import threading

    # Useful maps
    topo_2_idx_map = {}
    idx_2_topo_map = {}
    for topo_idx, topo in enumerate(exhaustive_topos()):
        topo_2_idx_map[topo]=topo_idx
        idx_2_topo_map[topo_idx]=topo

    # Map job IDs to topologies.
    jobs = {}
    jobs_lock = threading.Lock()

    # Map topologies to errors.
    topo_errors = collections.defaultdict(list)

    def completion(jobid, output):
        with jobs_lock:
            topo = jobs.pop(jobid)
        logging.info(u'got result for {}: error = {}'.format('-'.join(map(str, topo)), output))
        topo_errors[topo].append(output)

    # Kill the master/workers in case previous run failed
    cw.slurm.stop()

    # Run jobs.
    cw.slurm.start(nworkers=clusterworkers)
    client = cw.client.ClientThread(completion, cw.slurm.master_host())
    client.start()
    for topo in exhaustive_topos():
        for i in range(DEFAULT_REPS):
            jobid = cw.randid()
            with jobs_lock:
                jobs[jobid] = topo
            client.submit(jobid, evaluate, datafn, topo)
    logging.info('all jobs submitted')
    client.wait()
    cw.slurm.stop()
    logging.info('all jobs finished')

    # Find best.
    min_error = None
    min_topo = None
    for topo, errors in topo_errors.items():
        error = sum(errors) / len(errors)
        if min_error is None or error < min_error:
            min_error = error
            min_topo = topo
    print('best:', '-'.join(map(str, min_topo)))
    print('error:', min_error)


    # Read training data params
    input_neurons = 0
    output_neurons = 0
    with open(datafn, 'r') as f:
        params = f.readline().rstrip().split(" ")
        input_neurons = params[1]
        output_neurons = params[2]
    # Prepare CSV data
    csv_data = get_params()
    for topo, errors in topo_errors.items():
        topo_str = '-'.join(map(str, topo))
        error = sum(errors) / len(errors)
        madds = 0
        for i, hidden_neurons in enumerate(topo):
            if (i==0):
                madds += int(input_neurons)*int(hidden_neurons)
            else:
                madds += int(topo[i-1])*int(topo[i])
        madds += int(topo[len(topo)-1])*int(output_neurons)
        csv_data.append([topo_str, madds, t["error"]])
    # Dump to CSV
    with open(csvpath, 'wb') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        for line in csv_data:
            wr.writerow(line)

def cli():
    parser = argparse.ArgumentParser(
        description='Exhaustive neural network training'
    )
    parser.add_argument(
        '-train', dest='trainfn', action='store', type=str, required=True,
        default=None, help='training data file'
    )
    parser.add_argument(
        '-c', dest='clusterworkers', action='store', type=int, required=False,
        default=0, help='parallelize on cluster'
    )
    parser.add_argument(
        '-d', dest='debug', action='store_true', required=False,
        default=False, help='print out debug messages'
    )
    parser.add_argument(
        '-csv', dest='csvpath', action='store', type=str, required=False,
        default=CSV_FILE, help='path to csv results file'
    )
    parser.add_argument(
        '-log', dest='logpath', action='store', type=str, required=False,
        default=LOG_FILE, help='path to log file'
    )
    args = parser.parse_args()

    # Take care of log formatting
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s", datefmt='%m/%d/%Y %I:%M:%S %p')
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler(args.logpath)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    if(args.debug):
        rootLogger.setLevel(logging.DEBUG)
    else:
        rootLogger.setLevel(logging.INFO)

    if args.clusterworkers>0:
        nntune_cw(args.trainfn, args.clusterworkers, args.csvpath)
    else:
        nntune_sequential(args.trainfn, args.csvpath)

if __name__ == '__main__':

    cli()
