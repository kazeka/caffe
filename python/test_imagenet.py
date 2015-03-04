#!/usr/bin/env python


import caffe
import itertools
import numpy
import os
import pickle
import sys


DATA_PREFIX = '/mnt/data'
TEST_FILE = 'test.cont'
MODEL_DEF = 'models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED_MODEL = 'models/bvlc_reference_caffenet/caffenet_train_iter_450000.caffemodel'
MEANS_FILE = 'python/caffe/imagenet/ilsvrc_2012_mean.npy' # 'data/ilsvrc12/imagenet_mean.binaryproto'


def grouper(n, iterable):
    it = iter(iterable)
    while True:
       chunk = tuple(itertools.islice(it, n))
       if not chunk:
           return
       yield chunk


def catch(func, handle=lambda e: e, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return handle(e)


def main(ofname):
    with open(os.path.join(DATA_PREFIX, TEST_FILE), 'r') as fp:
        test_data = fp.readlines()
    files = [line.split(' ')[0] for line in test_data]
    classes = [line.split(' ')[1] for line in test_data]

    mean = numpy.load(os.path.join(MEANS_FILE))

    classifier = caffe.Classifier(MODEL_DEF, PRETRAINED_MODEL, image_dims=(256,256), mean=mean, raw_scale=255., channel_swap=(2,1,0))


    # classify in chunks of 100 images at a time not to OOM
    res = []
    for fchunk, catchunk in zip(grouper(100, files), grouper(100, classes)):
        inputs = [caffe.io.load_image(os.path.join(DATA_PREFIX, filename)) for filename in fchunk]
        predictions = classifier.predict(inputs)
        res.append((fchunk, catchunk, predictions))
        with open(ofname, 'w') as fp:
            pickle.dump(res, fp)
        print ".",


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usage: ./test_imagenet.py <output file name>"
        sys.exit(1)

    main(sys.argv[1])
