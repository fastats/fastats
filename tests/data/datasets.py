from sklearn import datasets


class SKLeanDataSets:

    def __iter__(self):
        iris = datasets.load_iris()
        diabetes = datasets.load_diabetes()
        boston = datasets.load_boston()
        cancer = datasets.load_breast_cancer()
        linnerud = datasets.load_linnerud()
        return iter((iris, diabetes, boston, cancer, linnerud))

    @staticmethod
    def describe(data_set):
        return data_set.DESCR.split()[0]
