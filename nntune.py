#!/usr/bin/env python

####################################################################################
## File Name: nntune.py
## Authors: Adrian Sampson, Thierry Moreau
## Email: asampson@uw.edu, moreau@uw.edu
##
## Copyright (c) 2012-2016 University of Washington
## All rights reserved.
##
## Redistribution and use in source and binary forms, with or without modification,
## are permitted provided that the following conditions are met:
## -       Redistributions of source code must retain the above copyright notice,
##         this list of conditions and the following disclaimer.
## -       Redistributions in binary form must reproduce the above copyright notice,
##         this list of conditions and the following disclaimer in the documentation
##         and/or other materials provided with the distribution.
## -       Neither the name of the University of Washington nor the names of its
##         contributors may be used to endorse or promote products derived from this
##         software without specific prior written permission.
##
## THIS SOFTWARE IS PROVIDED BY THE UNIVERSITY OF WASHINGTON AND CONTRIBUTORS "AS IS"
## AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
## WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
## IN NO EVENT SHALL THE UNIVERSITY OF WASHINGTON OR CONTRIBUTORS BE LIABLE FOR ANY
## DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
## (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
## OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
## THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
## NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
## IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
####################################################################################

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
import shlex
import shutil

# This needs to be modified to point to the fann shared resource dir
# e.g. should be pointing to /usr/local/lib if installed locally
FANN_LIB_DIR = '../fann-snnap/src'

# Path where nntune.py resides
NNTUNEDIR = os.path.dirname(os.path.realpath(__file__))

TRAIN_CMD = './train'
RECALL_CMD = './recall'
RECALL_FIX_CMD = './recall_fix'

# Output file paths
LOG_FILE = 'nntune.log'
NN_FILE = 'output.nn'

# Define Defaults Here
DEFAULT_REPS                = 2     # Number of times we are training the same NN
DEFAULT_EPOCHS              = 500   # Number of epochs
DEFAULT_LEARNING_RATE       = 0.2   # Default learning rate
DEFAULT_TRAIN_RATIO         = 0.7   # Proportion of training data to test data
DEFAULT_TOPO_EXPONENTIAL    = True  # Set to true if number of neurons should increase exponentially
DEFAULT_TOPO_INCR           = 2     # Topology exploration step size
DEFAULT_TOPO_MAX_LAYERS     = 1     # Maximum number of hidden layers
DEFAULT_TOPO_MAX_NEURONS    = 64    # Maximum number of neurons
DEFAULT_ERROR_MODE          = 0     # 0 for MSE, 1 for classification
DEFAULT_ERROR_TARGET        = 0.01  # Error target
DEFAULT_INTPREC             = 0     # Number of bits in integer portion (limits magnitude of weight)
DEFAULT_DECPREC             = 0     # 0 for float, anything else: fixed

def get_params(intbits, decbits, epochs, error_mode, error_target):

    params = []
    # Output timestamp info
    params.append(["timestamp", datetime.datetime.now()])
    # Training parameters
    params.append(["reps", DEFAULT_REPS])
    params.append(["learning rate", DEFAULT_LEARNING_RATE])
    params.append(["training ratio", DEFAULT_TRAIN_RATIO])
    params.append(["epochs", epochs])
    # Exploration parameters
    params.append(["topo exponential", DEFAULT_TOPO_EXPONENTIAL])
    params.append(["topo lin incr", DEFAULT_TOPO_INCR])
    params.append(["topo max layers", DEFAULT_TOPO_MAX_LAYERS])
    params.append(["topo max neurons", DEFAULT_TOPO_MAX_NEURONS])
    # Error evaluation mode
    params.append(["error mode", error_mode])
    params.append(["error target", error_target])
    # Precision parameters
    params.append(["integer precision", intbits])
    params.append(["decimal precision", decbits])

    return params

def shell(command, cwd=NNTUNEDIR, shell=False):
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
    return outstr


def train(datafile, topology, prec, epochs=DEFAULT_EPOCHS, testfile=None, learning_rate=DEFAULT_LEARNING_RATE):
    topostr = '-'.join(str(n) for n in topology)
    fd, fn = tempfile.mkstemp()
    os.close(fd)
    # Testfile is passed in for fixed point operation
    if testfile:
        shell([TRAIN_CMD, datafile, topostr, str(epochs), str(learning_rate), str(prec), fn, testfile])
    else:
        shell([TRAIN_CMD, datafile, topostr, str(epochs), str(learning_rate), str(prec), fn])
    return fn


def recall(nnfn, datafn, prec, error_mode=DEFAULT_ERROR_MODE):
    if prec==0:
        rmse = shell([RECALL_CMD, nnfn, datafn, str(error_mode)])
    else:
        rmse = shell([RECALL_FIX_CMD, nnfn, datafn, str(error_mode)])
    return float(rmse)

def isInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def read_data(fn):
    """Read a data file as a list of (input, output) pairs.
    """
    # Files can come in different formats
    # FANN formatting requires the first line to contain (1) the number of samples
    # (2) the number of inputs and (3) the number of outputs
    # If those are not explicitly provided, we need to infer this from the file
    values = []
    missingHeader = False
    numLines = 0
    numInput = None
    numOutput = None
    with open(fn) as f:
        for idx, line in enumerate(f):
            lineElems = line.strip().split()
            values += line.strip().split()
            # Check wether we are missing the header or not
            if idx==0 and (len(lineElems) or not isInt(lineElems[0])):
                missingHeader = True
            # If the header is not provided, derive the FANN header fields
            if missingHeader:
                # Determine if the current line is an input or output line
                if idx%2==0:
                    numLines += 1
                    if numInput:
                        assert(numInput == len(lineElems))
                    else:
                        numInput = len(lineElems)
                else:
                    if numOutput:
                        assert(numOutput == len(lineElems))
                    else:
                        numOutput = len(lineElems)
    if missingHeader:
        nsamples = numLines
        inputdim = numInput
        outputdim = numOutput
    else:
        nsamples = int(values.pop(0))
        inputdim = int(values.pop(0))
        outputdim = int(values.pop(0))

    logging.debug("FANN training file with {} samples, {} inputs and {} outputs".format(nsamples, inputdim, outputdim))

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
    if (len(data)>0):
        fd, fn = tempfile.mkstemp()
        f = os.fdopen(fd, 'w')
        dump_data(data, f)
        f.close()
    else:
        fn = None
    return fn

def divide_data(pairs, proportion=DEFAULT_TRAIN_RATIO):
    """Given a data set (sequence of pairs), divide it into two parts.
    Return two filenames.
    """
    split_point = int(len(pairs) * proportion)
    random.shuffle(pairs)
    data1, data2 = pairs[:split_point], pairs[split_point:]

    fn1 = dump_data_to_temp(data1)
    fn2 = dump_data_to_temp(data2)

    return fn1, fn2

def splitPosNeg(pairs):
    """Given a data set (sequence of pairs), divide it into positive and negative datasets.
    Return two filenames.
    """
    pos, neg = [], []
    for pair in pairs:
        if pair[1][0] >= 0.5:
            pos.append(pair)
        else:
            neg.append(pair)

    fn1 = dump_data_to_temp(pos)
    fn2 = dump_data_to_temp(neg)

    return fn1, fn2

def evaluate(datafn, hidden_topology, prec, errormode, epochs, nntarget, rep):
    # Read data.
    pairs = read_data(datafn)
    ninputs, noutputs = len(pairs[0][0]), len(pairs[0][1])
    topology = [ninputs] + list(hidden_topology) + [noutputs]

    # Split into training and testing files.
    trainfn, testfn = divide_data(pairs)
    if (errormode == 1): # binary classifier
        testPairs = read_data(testfn)
        postestfn, negtestfn = splitPosNeg(testPairs)

    try:
        # Train.
        nnfn = train(trainfn, topology, prec, epochs, testfn)
        try:
            # Test.
            misclassification = recall(nnfn, testfn, prec, errormode)
            # If binary classification, separate out Pos and Neg
            if (errormode):
                false_pos = recall(nnfn, negtestfn, prec, errormode) if (negtestfn) else -1
                false_neg = recall(nnfn, postestfn, prec, errormode) if (postestfn) else -1
                return [rep, misclassification, false_pos, false_neg]
            else:
                return [rep, misclassification]
        finally:
            if (nntarget):
                shutil.copyfile(nnfn, nntarget)
            os.remove(nnfn)
    finally:
        os.remove(trainfn)
        os.remove(testfn)


def increment_topo(topo, index, max_neurons, logSearch=DEFAULT_TOPO_EXPONENTIAL, incr=DEFAULT_TOPO_INCR):
    if (logSearch):
        topo[index] *= incr
    else:
        topo[index] += incr
    if topo[index] > max_neurons:
        if index == 0:
            return True
        else:
            topo[index] = 1
            return increment_topo(topo, index - 1, max_neurons)
    else:
        return False

def exhaustive_topos(max_layers=DEFAULT_TOPO_MAX_LAYERS, max_neurons=DEFAULT_TOPO_MAX_NEURONS):
    for layers in range(1, max_layers + 1):
        topo = [1] * layers
        while True:
            yield tuple(topo)
            if increment_topo(topo, layers - 1, max_neurons):
                break

def nntune_sequential(datafn, prec, errormode, errortarget, epochs, nnpath):
    min_error = None
    min_topo = None
    experiments = [] # experiments results

    for topo in exhaustive_topos():
        for i in range(DEFAULT_REPS):
            topoStr = '-'.join(map(str, topo))
            logging.info('testing {}, rep {}'.format(topoStr, i + 1))
            error = evaluate(datafn, topo, prec, errormode, epochs, nnpath, i)
            MSE = error[1]
            logging.info('\tMSE: {}'.format(MSE))
            if MSE<errortarget:
                logging.info("ANN topology {} with {} MSE meets target of {}".format(topoStr,  MSE, errortarget))
                return True


def exploreTopologies(trainfn, intprec, decprec, errormode, errortarget, epochs, nnpath):

    # Exponentiate the wlim
    intprec = pow(2, intprec)

    # Recompile the executables
    shell(shlex.split('make WEIGHTLIM='+str(intprec)))

    found = nntune_sequential(
        datafn=trainfn,
        prec=decprec,
        errormode=errormode,
        errortarget=errortarget,
        epochs=epochs,
        nnpath=nnpath
    )

    if not found:
        logging.info("No topology was found that meets error requirements.")

def cli():
    parser = argparse.ArgumentParser(
        description='Exhaustive neural network training'
    )
    parser.add_argument(
        '-train', dest='trainfn', action='store', type=str, required=True,
        default=None, help='training data file'
    )
    parser.add_argument(
        '-intbits', dest='intbits', action='store', type=int, required=False,
        default=DEFAULT_INTPREC, help='integer precision of trained weights'
    )
    parser.add_argument(
        '-decbits', dest='decbits', action='store', type=int, required=False,
        default=DEFAULT_DECPREC, help='decimal precision of trained weights'
    )
    parser.add_argument(
        '-epochs', dest='epochs', action='store', type=int, required=False,
        default=DEFAULT_EPOCHS, help='number of epochs required for training'
    )
    parser.add_argument(
        '-error_mode', dest='error_mode', action='store', type=int, required=False,
        default=DEFAULT_ERROR_MODE, help='error mode: 0: MSE, 1: Classification'
    )
    parser.add_argument(
        '-error_target', dest='error_target', action='store', type=float, required=False,
        default=DEFAULT_ERROR_TARGET, help='error target'
    )
    parser.add_argument(
        '-d', dest='debug', action='store_true', required=False,
        default=False, help='print out debug messages'
    )
    parser.add_argument(
        '-log', dest='logpath', action='store', type=str, required=False,
        default=LOG_FILE, help='path to log file'
    )
    parser.add_argument(
        '-nnpath', dest='nnpath', action='store', type=str, required=False,
        default=NN_FILE, help='path to nn file'
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

    exploreTopologies(
        trainfn=args.trainfn,
        intprec=args.intbits,
        decprec=args.decbits,
        errormode=args.error_mode,
        errortarget=args.error_target,
        epochs=args.epochs,
        nnpath=args.nnpath
    )

if __name__ == '__main__':

    cli()
