
import numpy as np
from numba import jit
from pytest import mark
from sklearn.decomposition import PCA

from fastats.linear_algebra import pca
from tests.data.datasets import SKLearnDataSets


@mark.parametrize('A', SKLearnDataSets)
def test_pca_sklearn(A):
    data = A.value.data

    for n in range(1, data.shape[1] + 1):
        sk_pca = PCA(n_components=n)
        sk_pca.fit(data)
        expected = sk_pca.transform(data)
        output = pca(data, components=n)
        assert np.allclose(np.abs(expected), np.abs(output))  # vector could legitimately be in 'opposite' direction


@mark.parametrize('A', SKLearnDataSets)
def test_pca_jit_sklearn(A):
    data = A.value.data
    pca_jit = jit(pca)

    for n in range(1, data.shape[1] + 1):
        sk_pca = PCA(n_components=n)
        sk_pca.fit(data)
        expected = sk_pca.transform(data)
        output = pca_jit(data, components=n)
        assert np.allclose(np.abs(expected), np.abs(output))  # vector could legitimately be in 'opposite' direction


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
