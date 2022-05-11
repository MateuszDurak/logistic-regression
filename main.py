from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

        # konfiguruje generator znaczników i mapę kolorów
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        # rysuje wykres powierzchni decyzyjnej
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        # rysuje wykres wszystkich próbek
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl, edgecolor='black')


class LogisticRegressionGD(object):
        def __init__(self, eta=0.05, n_iter=100, random_state=1):
                self.eta = eta
                self.n_iter = n_iter
                self.random_state = random_state

        def fit(self, X, y):
                rgen = np.random.RandomState(self.random_state)
                self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
                self.cost_ = []

                for i in range(self.n_iter):
                        net_input = self.net_input(X)
                        output = self.activation(net_input)
                        errors = (y - output)
                        self.w_[1:] += self.eta * X.T.dot(errors)
                        self.w_[0] += self.eta * errors.sum()
                        cost = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))))
                        self.cost_.append(cost)
                return self

        def net_input(self, X):
                return np.dot(X, self.w_[1:]) + self.w_[0]

        def activation(self, z):
                return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

        def predict(self, X):
                return np.where(self.net_input(X) >= 0.0, 1, 0)
class Classifier:
    def __init__(self, lrgd1, lrgd2):
        self.lrgd1 = lrgd1
        self.lrgd2 = lrgd2

    def predict(self, x):
        return np.where(self.lrgd1.predict(x) == 1, 0, np.where(self.lrgd2.predict(x) == 1, 0, 1))
def main():
        iris = datasets.load_iris()
        X = iris.data[:, [2, 3]]
        y = iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
        y_train_01_subset = y_train.copy()
        y_train_03_subset = y_train.copy()
        X_train_01_subset = X_train.copy()
        # w regresji logarytmicznej wyjście przyjmuje wartości 0 lub 1 (prawdopodobieństwa)

        y_train_01_subset[(y_train == 1) | (y_train == 2)] = 0
        y_train_01_subset[(y_train_01_subset == 0)] = 1

        y_train_03_subset[(y_train == 1) | (y_train == 0)] = 0
        y_train_03_subset[(y_train_03_subset == 2)] = 1
        print('y_train_01_subset: ', y_train_01_subset)
        print('y_train_03_subset: ', y_train_03_subset)
        # uczenie modelu

        lrgd1 = LogisticRegressionGD(eta=0.03, n_iter=1000 ,random_state=1)
        lrgd1.fit(X_train_01_subset, y_train_01_subset)


        lrgd2 = LogisticRegressionGD(eta=0.03, n_iter=1000, random_state=1)
        lrgd2.fit(X_train_01_subset, y_train_03_subset)

        y1_predict = lrgd1.predict(X_train)
        y3_predict = lrgd2.predict(X_train)

        acc1 = acc(lrgd1.predict(X_train), y_train_01_subset)
        acc2 = acc(lrgd1.predict(X_train), y_train_03_subset)
        print("acc1: ", acc1)
        print("acc2: ", acc2)

        _classifier = Classifier(lrgd1, lrgd2)

        #lrgd = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
        #lrgd.fit(X_train_01_subset, y_train_01_subset)
        plot_decision_regions(X=X_train, y=y_train, classifier=_classifier)
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.legend(loc='upper left')
        plt.show()


def acc(y_results, y_train):
        return (1 - np.mean(y_results != y_train)) * 100

if __name__ == '__main__':
        main()