from enum import Enum

from sklearn import datasets


class SKLearnDataSets(Enum):
    CANCER = datasets.load_breast_cancer().data
    DIABETES = datasets.load_diabetes().data
    LINNERUD = datasets.load_linnerud().data
    IRIS = datasets.load_iris().data
    WINE = datasets.load_wine().data
