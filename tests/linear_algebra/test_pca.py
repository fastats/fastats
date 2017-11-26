
from numba import jit
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import (
    load_iris, load_diabetes, load_wine
)

from fastats.linear_algebra import pca


def test_pca_sklearn_iris():
    iris = load_iris()

    sk_pca = PCA(n_components=2)
    sk_pca.fit(iris.data)
    sk2 = sk_pca.transform(iris.data)

    data2 = pca(iris.data, components=2)

    assert np.allclose(np.abs(sk2), np.abs(data2))

    sk_pca = PCA(n_components=4)
    sk_pca.fit(iris.data)
    sk4 = sk_pca.transform(iris.data)

    data4 = pca(iris.data, components=4)

    assert np.allclose(np.abs(sk4), np.abs(data4))


def test_pca_sklearn_diabetes():
    diab = load_diabetes()

    sk_pca = PCA(n_components=2)
    sk_pca.fit(diab.data)
    sk2 = sk_pca.transform(diab.data)

    data2 = pca(diab.data, components=2)

    assert np.allclose(np.abs(sk2), np.abs(data2))

    sk_pca = PCA(n_components=4)
    sk_pca.fit(diab.data)
    sk4 = sk_pca.transform(diab.data)

    data4 = pca(diab.data, components=4)

    assert np.allclose(np.abs(sk4), np.abs(data4))


def test_pca_jit_sklearn_wine():
    wine = load_wine()

    pca_jit = jit(pca)

    sk_pca = PCA(n_components=2)
    sk_pca.fit(wine.data)
    sk2 = sk_pca.transform(wine.data)

    data2 = pca_jit(wine.data, components=2)

    assert np.allclose(np.abs(sk2), np.abs(data2))


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
