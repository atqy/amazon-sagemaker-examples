from mxnet.gluon.data.vision import datasets, transforms
from mxnet import ndarray as nd, image, recordio, base
from mxnet.gluon.utils import download, check_sha1, _get_repo_file_url
import gzip
import os
import struct
import numpy as np
from mxnet.util import is_np_array

class FashionMNIST(datasets.MNIST):
    """A dataset of Zalando's article images consisting of fashion products,
    a drop-in replacement of the original MNIST dataset from
    https://github.com/zalandoresearch/fashion-mnist
    Each sample is an image (in 3D NDArray) with shape (28, 28, 1).
    Parameters
    ----------
    root : str, default $MXNET_HOME/datasets/fashion-mnist'
        Path to temp folder for storing data.
    train : bool, default True
        Whether to load the training or testing set.
    transform : function, default None
        DEPRECATED FUNCTION ARGUMENTS.
        A user defined callback that transforms each sample. For example::
            transform=lambda data, label: (data.astype(np.float32)/255, label)
    """
    def __init__(self, root=os.path.join(base.data_dir(), 'datasets', 'fashion-mnist'),
                 train=True, transform=None):
        self._train = train
        self._train_data = 'train-images-idx3-ubyte.gz'
        self._train_label = 'train-labels-idx1-ubyte.gz'
        self._test_data = 't10k-images-idx3-ubyte.gz'
        self._test_label = 't10k-labels-idx1-ubyte.gz'
        self._namespace = 'fashion-mnist'
        super(datasets.MNIST, self).__init__(root, transform) # pylint: disable=bad-super-call
    
    def _get_data(self):
        if self._train:
            data, label = self._train_data, self._train_label
        else:
            data, label = self._test_data, self._test_label
        data_file = download("https://sagemaker-sample-files.s3.amazonaws.com/datasets/image/fashion-MNIST/" + data,
                             path=self._root)
        label_file = download("https://sagemaker-sample-files.s3.amazonaws.com/datasets/image/fashion-MNIST/" + label,
                              path=self._root)

        with gzip.open(label_file, 'rb') as fin:
            struct.unpack(">II", fin.read(8))
            label = np.frombuffer(fin.read(), dtype=np.uint8).astype(np.int32)
            if is_np_array():
                label = _mx_np.array(label, dtype=label.dtype)

        with gzip.open(data_file, 'rb') as fin:
            struct.unpack(">IIII", fin.read(16))
            data = np.frombuffer(fin.read(), dtype=np.uint8)
            data = data.reshape(len(label), 28, 28, 1)

        array_fn = _mx_np.array if is_np_array() else nd.array
        self._data = array_fn(data, dtype=data.dtype)
        self._label = label
