#!/usr/bin/env python
import random
import argparse
import math
import numpy as np
from matplotlib import cm, pyplot as plt

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

def visualize(datafn, samples):
    faces = read_data(datafn)
    ninputs, noutputs = len(faces[0][0]), len(faces[0][1])
    dim = int(math.sqrt(ninputs))
    for face in faces[0:samples]:
        pixels = []
        for i in range(dim):
            pixels.append(face[0][i*dim:(i+1)*dim])
        img_array = np.array(pixels)
        plt.imshow(img_array, interpolation='nearest', cmap = cm.Greys_r)
        plt.show()


def cli():
    parser = argparse.ArgumentParser(
        description='Vizualize input data'
    )
    parser.add_argument(
        '-path', dest='datafn', action='store', type=str, required=True,
        default=None, help='data file'
    )
    parser.add_argument(
        '-samples', dest='samples', action='store', type=int, required=False,
        default=1, help='data file'
    )
    args = parser.parse_args()

    visualize(args.datafn, args.samples)

if __name__ == '__main__':

    cli()