"""
========================================
Linear Classifier
Author: Thomas W. Rauber mailto:trauber@gmail.com
========================================
"""

import os
import sys
here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(here + '/../lib')  # ; print('sys.path=\n', sys.path)
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score, f1_score


from defines import defines
rootdir = defines['rootdir']
from util import cm2inch, path_setup, tex_setup, print_array, second_largest
graphics_dim_two = defines['graphics_dim_two']
wid = cm2inch(graphics_dim_two[0])
hei = cm2inch(graphics_dim_two[1])
from Cohen_Sutherland_Clipping import cohenSutherlandClip

path_setup()
tex_setup(usetex=True)

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from Iris import Iris


def _init_w_b(X, y, randominit=None):
    """Initializes the weights and the bias."""
    n, d = X.shape
    if randominit is not None:
        np.random.seed(randominit)
        weights = np.random.randn(d)
        b = 0
        return weights, b

    idxpos = np.where(y == +1)[0]
    # print('idxpos=', idxpos)
    Xpos = X[idxpos]
    idxneg = np.where(y == -1)[0]
    # print('idxneg=', idxneg)
    Xneg = X[idxneg]
    # print('X=\n', X, 'shape=', X.shape, '\ny=\n', y)
    # print('Xpos=\n', Xpos, 'shape=', Xpos.shape)
    # print('Xneg=\n', Xneg, 'shape=', Xneg.shape)
    # This is the initial difference vector between the two means
    mupos = np.mean(Xpos, axis=0)
    muneg = np.mean(Xneg, axis=0)
    mu = mupos - muneg
    # print('mu=', mu)
    weights = mu
    on_hyperplane = (mupos + muneg) / 2
    # The distance of this point to the hyperplane is zero
    b = -np.dot(weights, on_hyperplane)
    # Verify
    # print('Verify: ', np.dot(weights, on_hyperplane) + b,
    # np.dot(weights, mupos) + b, np.dot(weights, muneg) + b)
    print('INIT weights and bias: weights=', weights, 'b=', b)
    return weights, b


def _perceptron_learning_batch(X, y, epochs, eta, randominit, verbose):
    """Rosenblatt perceptron learning, batch version."""
    n, d = X.shape
    # Set the initial weights
    weights, bias = _init_w_b(X, y, randominit)

    # verbose = False

    print('P E R C E P T R O N  B A T C H  T R A I N I N G')
    loss_values = []

    for i in range(epochs):
        numerr = 0
        gradsumW = 0.0
        gradsumb = 0.0
        for j in range(n):
            yj = y[j]
            activation = np.dot(X[j], weights) + bias
            # print('X[', j, ']=', X[j], 'activation=%.3f' % activation, end=' ')
            y_hat = np.sign(activation)
            err = yj != y_hat
            numerr += err[0]
            if verbose:
                print('Epoch=%4d' % (i+1), 'pattern no=%4d' % (j+1), 'y=', yj,
                      ' y_hat=', y_hat, ' err=', err, ' nerr=', numerr,
                      'weights=', weights, ' bias=', bias,
                      'yj.shape=', yj.shape, 'yhat.shape=', y_hat.shape)

            aux = -np.sign(yj-y_hat)
            if err:
                gradsumW += aux*X[j]
                gradsumb += aux  # * 1.0

        weights = weights - eta * gradsumW / n
        bias = bias - eta * gradsumb / n
        loss_values.append(numerr)

    W = np.concatenate((bias, weights)).reshape((d+1, 1))
    print('History number of errors:', loss_values)
    print('weights=', weights, 'bias=', bias, 'concat=', W)
    return W, loss_values


def _perceptron_learning(X, y, batchsize, epochs, eta, randominit, verbose):
    """Rosenblatt perceptron learning."""
    n, d = X.shape
    print('\n=== Perceptron learning ===\nNumber of samples=', n,
          'batchsize=', batchsize, 'eta=', eta, 'epochs=', epochs,
          'random init=', randominit)
    if n == batchsize:
        # eta = 0.05
        # epochs = 1000
        return _perceptron_learning_batch(X, y, epochs, eta, randominit,
                                          verbose)

    # Set the initial weights
    weights, bias = _init_w_b(X, y, randominit)

    loss_values = []

    shuffled = list(range(n))
    for i in range(epochs):
        numerr = 0
        np.random.shuffle(shuffled)
        for j in shuffled:
            yj = y[j]
            activation = np.dot(X[j], weights) + bias  # w x + b
            y_hat = np.sign(activation)
            err = yj != y_hat
            numerr += err[0]
            if verbose:
                print('Epoch=%4d' % (i+1), 'pattern no=%4d' % (j+1), 'y=', yj,
                      ' y_hat=', y_hat, ' err=', err, ' nerr=', numerr,
                      'weights=', weights, ' bias=', bias,
                      'yj.shape=', yj.shape, 'yhat.shape=', y_hat.shape)
            '''
            if err:
                # print('weights=', weights, 'shape=', weights.shape)
                # print('X[', j, ']=', weights, 'shape=', X[j].shape)
                weights = weights + eta * X[j]*yj
                bias = bias + eta * yj
            '''
            # A L T E R N A T I V E
            # Stochastic gradient
            aux = -np.sign(yj-y_hat)
            weights = weights - eta * aux*X[j]
            bias = bias - eta * aux

            # if numerr == 0:
            #     break
        loss_values.append(numerr)

    W = np.concatenate((bias, weights)).reshape((d+1, 1))
    print('weights=', weights, 'bias=', bias, 'concat=', W)
    return W, loss_values


class LinearClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    """ Linear Classifier for two classes
    Deterministic supervised learning of parameters w, b
    Parameters
    ----------

    Attributes
    ----------
    """

    def __init__(self, W=None, bias=None, weights=None, wnorm=None,
                 lb=LabelBinarizer()):
        # print('Executing __init__() ....')
        self.W = W
        self.bias = bias
        self.weights = weights
        self.wnorm = wnorm
        self.lb = lb
        self.method = None
        self.batchsize = None  # For iterative methods
        self.epochs = None
        self.randominit = None  # Can contain the seed of random generator
        self.eta = None
        self.learning_history = None

    def fit(self, X, y, method='pseudoinverse', batchsize=1, epochs=1000,
            eta=1.0, randominit=None, verbose=False):
        """Learn weight vector and bias."""
        n, m = X.shape
        self.method = method
        self.randominit = randominit
        self.lb.fit(y)
        Y = self.lb.transform(y)
        Y = 2*Y - 1
        if method == 'pseudoinverse':
            # Deterministic learning by Pseudoinverse of data, times targer.
            Xaug = np.concatenate((np.ones((n, 1)), X), axis=1)
            Xpinv = np.linalg.pinv(Xaug)
            self.W = np.dot(Xpinv, Y)
            '''
            print('Xaug=\n', Xaug, 'shape=', Xaug.shape)
            print('Xpinv.shape=', Xpinv.shape)
            print('Xpinv=\n', Xpinv)
            '''

        if method == 'perceptron':
            self.W, self.learning_history = _perceptron_learning(X, Y,
                                                                 batchsize,
                                                                 epochs,
                                                                 eta,
                                                                 randominit,
                                                                 verbose)
            self.batchsize = batchsize
            self.epochs = epochs
            self.eta = eta

            import pickle
            picklefile = './_tmp_iris_learned_perceptron' + '.pkl'
            print('Saving temporarily learned parameters to', picklefile)
            with open(picklefile, 'wb') as f:
                pickle.dump(dict(W=self.W, X=X, Y=Y), f)


        self.bias = self.W[0]
        self.weights = self.W[1:]
        self.wnorm = np.linalg.norm(self.weights)
        '''
        print('\nLinearClassifier.fit>\nmethod=', self.method,
              '\nlabels=y=', y, '\nY=\n', Y, 'Y.shape=', Y.shape)
        print('X.shape=', X.shape)
        print('W.shape=', self.W.shape)
        print('W=\n', self.W)
        print('bias=\n', self.bias)
        print('weights=\n', self.weights)
        print('norm of weights=\n', self.wnorm) ;  # quit()
        '''

    def predict(self, X):
        Y = np.dot(X, self.weights) + self.bias
        # y_pred = self.lb.inverse_transform(Y)
        y_pred = np.sign(Y)
        # print('\npredict: Y=\n', Y, 'shape=', Y.shape,
        # '\ny_pred=\n', y_pred, 'shape=', y_pred.shape) ; quit()
        return y_pred

    def score(self, X):
        Y = np.dot(X, self.weights) + self.bias
        # The real distances of each pattern from the hyperplane
        dists = Y / self.wnorm
        # print('\nY=\n', Y, 'shape=', Y.shape)
        return Y, dists

    # definition of hyperplane c0 = +- intercept
    # c0 + c1 x + c2 y = 0
    # Three cases: vertical, horizontal, general

    def plot_hyperplane(self, ax, c, color, coef, intercept,
                        cmap=plt.cm.Paired, bb=None):
        def y(x, coef, intercept):
            # does not work for coef[c, 1]==0 which is a vertical line
            return (-(x * coef[c, 0]) - intercept[c]) / coef[c, 1]

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        c0 = coef[c, 0]
        c1 = coef[c, 1]
        # print('plot_hyperplane> c0=', c0, 'c1=', c1)
        if c0 == 0:
            y0 = y1 = intercept[c] / c1
            x0 = xmin
            x1 = xmax
        elif c1 == 0:
            x0 = x1 = intercept[c] / c0
            y0 = ymin
            y1 = ymax
        else:
            x0 = xmin
            x1 = xmax
            y0 = y(xmin, coef, intercept)
            y1 = y(xmax, coef, intercept)

        if bb is not None:
            print('x0=%.3f, x1=%.3f, y0=%.3f, y1=%.3f: ' %
                  (x0, x1, y0, y1), 'bb=', bb)

            x0, x1, y0, y1 = cohenSutherlandClip(x0, x1, y0, y1,
                                                 bb=bb, verbose=False)
            xm = x0 + (x1-x0)/3.0
        else:
            xm = 3.75  # x-Position of weight vector in figure

        print('Plotting: xmin=%.3f, xmax=%.3f, ymin=%.3f, ymax=%.3f,\
              x0=%.3f, x1=%.3f, y0=%.3f, y1=%.3f: ' %
              (xmin, xmax, ymin, ymax, x0, x1, y0, y1))
        ax.plot([x0, x1], [y0, y1], ls="--", color=color)

        # xm = (x0+x1)/2
        ym = (y0+y1)/2

        ym = y(xm, coef, intercept)

        # ax.plot([xm, xm+c0], [ym, ym+c1], ls="-", color='red', linewidth=2)
        ax.arrow(xm, ym, c0, c1, width=0.01, head_width=0.1,
                 facecolor='r', edgecolor='r', linewidth=1)
        wstr = r'$\mathbf{\hat{w}}$'
        wstr = r'$\mathbf{w}$'
        ax.annotate(wstr, xy=(xm, 2.8), c='r', ha='right')
        ax.annotate('hiperplano', xy=(x0+0.1, y0), c='k', ha='left')

    def plot_linclassifier(self, fig, ax, iris, lc, xx, yy, Z,
                           cmap, colors, bb):

        cs = ax.contourf(xx, yy, Z, 20, cmap=cmap)
        # cbar = fig.colorbar(cs, ax=ax)
        # cbar.ax.set_ylabel('scores')

        X = iris.X
        y = iris.y
        feat1 = iris.features[0]
        feat2 = iris.features[1]

        cn = 0
        for i, color in zip(iris.classset, colors):
            idx = np.where(y == i)
            Xi = X[idx]
            ax.scatter(Xi[:, 0], Xi[:, 1], label=iris.classname[cn],
                       edgecolor=color, facecolors='none', s=20, lw=0.5)
            cn = cn + 1

        params = lc.get_params(deep=True)
        W, bias, weights, wnorm = params['W'], params['bias'],\
            params['weights'], params['wnorm']
        W_orig = np.copy(W)
        coef = weights.T
        intercept = bias
        print('Hyperplane defined by normal vector w=(%.3f, %.3f)\
              with norm ||w||=%.3f and bias w0=%.3f' %
              (coef[0][0], coef[0][1], wnorm, bias[0]))
        coef[0][0] /= wnorm  # This command implicitly changes W
        coef[0][1] /= wnorm  # This command implicitly changes W
        intercept[0] /= wnorm  # This command implicitly changes W
        self.plot_hyperplane(ax, 0, 'black', coef, intercept, cmap=cmap, bb=bb)

        handles, labels = ax.get_legend_handles_labels()
        # print('handles=', handles, 'labels=', labels)
        ax.set(xlabel=r'$x_{' + str(feat1+1) +
               '}=\\mathrm{' + iris.featname_all_mathmode[feat1] + '}$')
        ax.set(ylabel=r'$x_{' + str(feat2+1) +
               '}=\\mathrm{' + iris.featname_all_mathmode[feat2] + '}$')

        ax.legend(handles, labels)
        methodstr2 = lc.method
        if iris.language_code == 'pt_BR':
            methodstr1 = 'Algoritmo de aprendizagem'
            if methodstr2 == 'pseudoinverse':
                methodstr2 = 'pseudoinversa'
        else:
            methodstr1 = 'Learning method'
        # print('W=', W, '\nweights=', weights)
        tit = (methodstr1 + ': ' + methodstr2 + '\n' +
               '$w=[%.2f, %.2f], b=%.2f, ||w||=%.2f$'
               % (W_orig[1], W_orig[2], W_orig[0], wnorm))
        ax.title.set_text(tit)
        # plt.legend([hp,]+handles, ['Hyperplane normal w',]+labels)
        ax.axis('equal')


def test_iris_resubstitution(iris):
    X = iris.X
    y = iris.y

    feat1 = iris.features[0]
    feat2 = iris.features[1]

    print('Iris data: Using only two features: feature 1 = ',
          1+feat1, '=', iris.featname[0], ' and feature 2 = ',
          1+feat2, '=', iris.featname[1])

    lc = LinearClassifier()
    lc.fit(X, y)
    lc.lb.fit(y)
    Y = lc.lb.transform(y)
    Y = 2*Y - 1
    Y_pred = lc.predict(X)

    params = lc.get_params(deep=True)
    W, bias, weights = params['W'], params['bias'], params['weights']

    print('W=\n', W, 'shape=', W.shape)
    print('bias=\n', bias)
    print('weights=\n', weights)

    print('\n==========> Resubstitution of training data:\n')
    print('Classification Report for all features and all classes: ')
    print(classification_report(Y, Y_pred, target_names=iris.classlabel,
                                digits=3))
    print('Accuracy=', '%.2f %%' % (100*accuracy_score(Y, Y_pred)))
    print('Confusion Matrix: ')
    print(confusion_matrix(Y, Y_pred))

    scores, dists = lc.score(X)
    numpat = X.shape[0]
    for i in range(numpat):
        print('Pattern %3d of %3d Distance from hyperplane = %+10.3f true class=%2d classification=%2d'
              % (i+1, numpat, dists[i], Y[i], np.sign(scores[i])))


def test_iris_k_fold(iris, K=10):
    X = iris.X
    y = iris.y

    lc = LinearClassifier()

    skf = StratifiedKFold(n_splits=K, shuffle=True)

    y_pred_overall = np.empty((0), dtype=int)
    y_test_overall = np.empty((0), dtype=int)

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        lc.fit(X_train, y_train)
        y_pred = lc.predict(X_test)[:, 0]

        y_pred_overall = np.concatenate([y_pred_overall, y_pred])
        y_test_overall = np.concatenate([y_test_overall, y_test])

    Y = lc.lb.transform(y_test_overall)
    Y = 2*Y - 1
    y_test_overall = Y

    accuracy = 100*accuracy_score(y_test_overall, y_pred_overall)
    print('\n==========> K-fold cross validation:\n')
    print('Linear Classifier ', K, '- Fold Classification Report: ')
    print(classification_report(y_test_overall, y_pred_overall,
                                target_names=iris.classname, digits=3))
    print('Accuracy=', '%.2f %%' % accuracy)
    print('Macro-averaged F1=', '%.3f' % (f1_score(y_test_overall,
                                                   y_pred_overall,
                                                   average='macro')))
    print('Micro-averaged F1=', '%.3f' % (f1_score(y_test_overall,
                                                   y_pred_overall,
                                                   average='micro')))
    print('Linear Classifier Confusion Matrix: ')
    print(confusion_matrix(y_test_overall, y_pred_overall))


def learn_and_visualize_linear_classifier(iris, feat1=2, feat2=3,
                                          cmap=plt.cm.Paired):

    fig, ax = plt.gcf(), plt.gca()
    fig.set_size_inches(cm2inch(14), cm2inch(7))

    X = iris.X
    y = iris.y
    feat1 = iris.features[0]
    feat2 = iris.features[1]

    print('\n\n===> V I S U A L I Z E  L I N E A R  C L A S S I F I E R  D E C I S I O N  R E G I O N S <===\n\n')
    print('Iris data: Using only two features: feature 1 = ', 1+feat1, '=',
          iris.featname[0], ' and feature 2 = ',
          1+feat2, '=', iris.featname[1])

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    h = .01  # step size in the mesh
    # h = .02  # step size in the mesh
    # x_min, x_max = 0, 8.001  # Best values 0, 8.001, eps-0.003
    # y_min, y_max = -2.5, 4.001  # Best values -1, 4.001 or -2.5, 4

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    xyplane = np.c_[xx.ravel(), yy.ravel()]
    lc = LinearClassifier()
    lc.fit(X, y)

    # y_pred = lc.predict(xyplane)
    scores, dists = lc.score(xyplane)
    Z = scores
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    # fig, ax = plt.gcf(), plt.gca()
    # fig.set_size_inches(cm2inch(8), cm2inch(8))

    ax.axis('off')
    # fig.set_size_inches(cm2inch(17), cm2inch(8))
    ax1 = fig.add_subplot(1, 2, 1)  # , projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)
    ax = ax1

    colors = "rgby"
    cmap = plt.cm.gnuplot
    cmap = plt.cm.bone
    cmap = 'RdBu'

    # --------------------------------
    lc.plot_linclassifier(fig, ax, iris, lc, xx, yy, Z, cmap, colors, bb=None)

    ax = ax2
    batchsize = 1
    batchsize = X.shape[0]
    lc.fit(X, y, method='perceptron', batchsize=batchsize)
    # y_pred = lc.predict(xyplane)
    scores, dists = lc.score(xyplane)
    Z = scores
    Z = Z.reshape(xx.shape)
    # --------------------------------
    lc.plot_linclassifier(fig, ax, iris, lc, xx, yy, Z, cmap, colors, bb=None)
    plt.tight_layout()

    plt.show()

'''
def main():
    print('Executing main() ....')
    iris = Iris(classset=(1, 2), features=(2, 3))

    # test_iris_resubstitution(iris)
    # test_iris_k_fold(iris)
    # return
    learn_and_visualize_linear_classifier(iris)


if __name__ == "__main__":
    main()
'''