from enum import Enum

from sklearn import datasets


class SKLearnDataSets(Enum):
    BOSTON = datasets.load_boston()
    CANCER = datasets.load_breast_cancer()
    DIABETES = datasets.load_diabetes()
    LINNERUD = datasets.load_linnerud()
    IRIS = datasets.load_iris()
    WINE = datasets.load_wine()
