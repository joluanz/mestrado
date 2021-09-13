"""
========================================
Linear Machine
Author: Thomas W. Rauber mailto:trauber@gmail.com
========================================
"""
#import sys
#sys.path.append('..')
#from defines import defines
#rootdir = defines['rootdir']
#sys.path.append(rootdir+'soft/lib/')

from util import cm2inch, tex_setup, myexit


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import sklearn.datasets as datasets
from LabelBinarizer2 import LabelBinarizer2

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import r2_score, accuracy_score, f1_score

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, TransformerMixin

from scipy.special import softmax


class LinearMachine(BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin):
    """ Linear Machine
    https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/base.py#L360
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).

    Parameters
    ----------

    Attributes
    ----------
    """
    def __init__(self, W=None, bias=None, weights=None, C=None, mode='classifier'):
        #print('Executing __init__() ....')
        self.W = W
        self.bias = bias
        self.weights = weights
        self.C = C  # Regularization hyperparameter
        self.mode = mode  # Act as regressor (continuous output) or classifier (class output: label or one-hot-encoded)
        self.lb = None
        if mode == 'classifier':
            self.lb = LabelBinarizer2()
            # https://github.com/scikit-learn/scikit-learn/blob/2beed55847ee70d363bdbfe14ee4401438fba057/sklearn/base.py
            self._estimator_type = 'classifier'
        else:
            self._estimator_type = 'regressor'


    def _pinv_regularized(self, X, C):
        ''' Regularized pseudoinverse
        '''
        return np.dot(np.linalg.inv(np.dot(X.T, X) + np.eye(X.shape[1])/C), X.T)

    def fit(self, X, y, labels_already_binarized=False):
        n, m = X.shape
        Xaug = np.concatenate((np.ones((n, 1)), X), axis=1)
        if self.mode == 'classifier' and not labels_already_binarized:
            self.lb.fit(y)
            Y = self.lb.transform(y)
        else:
            Y = y

        if self.C is None:
            Xpinv = np.linalg.pinv(Xaug)
        else:
            Xpinv = self._pinv_regularized(Xaug, self.C)
        self.W = np.dot(Xpinv, Y)
        self.bias = self.W[0]
        self.weights = self.W[1:]

        '''
        print('\nlabels=', y)
        print('X.shape=', X.shape, 'Xaug.shape=', Xaug.shape, 'Y.shape=', Y.shape)
        print('Xaug=\n', Xaug, '\nY=\n', Y)
        print('Xpinv.shape=', Xpinv.shape)
        print('Xpinv=\n', Xpinv)
        print('W.shape=', self.W.shape)
        print('W=\n', self.W)
        print('bias=\n', self.bias)
        print('weights=\n', self.weights)
        '''

    def predict(self, X, labels_already_binarized=False, use_softmax=False):
        # print('LinearMachine.predict> X=\n', X, '\nself.weights=', self.weights); # return
        Y = np.dot(X, self.weights) + self.bias
        # print('predict> Y=', Y)
        if use_softmax:
            # print('Y antes de softmax=', Y)
            Y = softmax(Y, axis=1)
            # print('Y depois de softmax=', Y)
        if self.mode == 'classifier' and not labels_already_binarized:
            y_pred = self.lb.inverse_transform(Y)
        else:
            y_pred = Y
        #print('\nY=\n', Y, 'shape=', Y.shape, '\ny_pred=\n', y_pred, 'shape=', y_pred.shape)
        #print('predict> y_pred=', y_pred)
        return y_pred

    def decision_function(self, X, use_softmax=False):
        Y = np.dot(X, self.weights)  + self.bias
        #print('\nY=\n', Y, 'shape=', Y.shape)
        if use_softmax:
            #print('Y antes de softmax=', Y)
            Y = softmax(Y, axis=1)
            #print('Y depois de softmax=', Y)
        return Y

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
    # def score(self, X, Y, use_softmax=False):
    #     Yhat = np.dot(X, self.weights)  + self.bias
    #     #print('\nY=\n', Y, 'shape=', Y.shape)
    #     if use_softmax:
    #         #print('Y antes de softmax=', Yhat)
    #         Yhat = softmax(Yhat, axis=1)
    #         #print('Y depois de softmax=', Y)
    #     return np.linalg.norm(Y - Yhat)

    def predict_proba(self, X):
        """Return posterior probabilities of classification.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Array of samples/test vectors.
        Returns
        -------
        C : ndarray of shape (n_samples, n_classes)
            Posterior probabilities of classification per class.
        """
        values = self.decision_function(X)
        # compute the likelihood of the underlying gaussian models
        # up to a multiplicative constant.
        likelihood = np.exp(values - values.max(axis=1)[:, np.newaxis])
        # compute posterior probabilities
        return likelihood / likelihood.sum(axis=1)[:, np.newaxis]


    def loss(self, X, Y):
        return np.linalg.norm(Y-(np.dot(X, self.weights)+self.bias))	# ||Y-Yhat||_F^2

def print_array(x, formatstr='%.2f'):
    print(np.array2string(x, formatter={'float_kind':lambda x: formatstr % x}))

def test_iris_resubstitution():
    # Get iris data
    iris = datasets.load_iris()
    X = iris.data
    labels = iris.target
    y = labels
    classset = np.unique(y)
    classes = iris.target_names
    numclasses = len(classes)
    featname = iris.feature_names

    lm = LinearMachine()
    lm.fit(X, y)
    params = lm.get_params(deep=True)
    W, bias, weights = params['W'], params['bias'], params['weights']
    print('W=\n', W)
    print_array(W, formatstr='%.3f')#; quit()
    y_pred = lm.predict(X)

    print('\n==========> Resubstitution of training data:\n')
    print('Classification Report for all features and all classes: ')
    print(classification_report(y, y_pred, target_names=classes, digits=3))
    print('Accuracy=', '%.2f %%' % (100*accuracy_score(y, y_pred)))
    print('Confusion Matrix: ')
    print(confusion_matrix(y, y_pred))


def test_iris_k_fold(K=10):
    # Get iris data
    iris = datasets.load_iris()
    X = iris.data
    labels = iris.target
    y = labels
    classset = np.unique(y)
    classes = iris.target_names
    numclasses = len(classes)
    featname = iris.feature_names

    skf = StratifiedKFold(n_splits=K, shuffle=True)
    #skf.get_n_splits(X, y)
    #print skf
    y_pred_overall = []
    y_test_overall = []

    lm = LinearMachine()

    for train_index, test_index in skf.split(X, y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        lm.fit(X_train, y_train)
        y_pred = lm.predict(X_test)
        
        y_pred_overall = np.concatenate([y_pred_overall, y_pred])
        y_test_overall = np.concatenate([y_test_overall, y_test])

    accuracy = 100*accuracy_score(y_test_overall, y_pred_overall)
    print('\n==========> K-fold cross validation:\n')
    print ('Linear Machine ', K, '- Fold Classification Report: ')
    print (classification_report(y_test_overall, y_pred_overall, target_names=classes, digits=3))
    print ('Accuracy=', '%.2f %%' % accuracy)
    print ('Macro-averaged F1=', '%.3f' % (f1_score(y_test_overall, y_pred_overall, average='macro')))
    print ('Micro-averaged F1=', '%.3f' % (f1_score(y_test_overall, y_pred_overall, average='micro')))
    print ('Linear Machine Confusion Matrix: ')
    print (confusion_matrix(y_test_overall, y_pred_overall))


def test_iris_visualize(feat1=2, feat2=3):
    # definition of hyperplane c0 = +- intercept
    # c0 + c1 x + c2 y = 0
    # Three cases: vertical, horizontal, general
    def plot_hyperplane(c, color, coef, bias):
        def y(x):
            # does not work for coef[c, 1]==0 which is a vertical line
            return (-(x * coef[c, 0]) - bias[c]) / coef[c, 1]
        
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()

        c0 = coef[c, 0]
        c1 = coef[c, 1]
        #print('plot_hyperplane> c0=', c0, 'c1=', c1)
        if c0 == 0:
            y0 = y1 = bias[c] / c1
            x0 = xmin
            x1 = xmax
        elif c1 == 0:
            x0 = x1 = bias[c] / c0
            y0 = ymin
            y1 = ymax
        else:
            x0 = xmin
            x1 = xmax
            y0 = y(xmin)
            y1 = y(xmax)
                
        #print('Plotting x0,x1,y0,y1: ', x0, x1, y0, y1)
        plt.plot([x0, x1], [y0, y1], ls="--", color=color, label=classes[c] +' specific')
        '''
        print('Plotting: ', xmin, xmax, line(xmin), line(xmax))
        plt.plot([xmin, xmax], [line(xmin), line(xmax)],
                ls="--", color=color)
        '''

    # Get iris data
    iris = datasets.load_iris()
    X = iris.data
    labels = iris.target
    y = labels
    classes = allclasses = iris.target_names

    '''
    # Only last two classes
    X = X[-100:]
    y = y[-100:]
    classes = classes[-2:]
    '''

    classset = np.unique(y)
    numclasses = len(classes)
    featname = iris.feature_names

    print('\n\n===> V I S U A L I Z E  L I N E A R  M A C H I N E  D E C I S I O N  R E G I O N S <===\n\n')
    print('Iris data: Using only two features: feature 1 = ', 1+feat1, '=',
          featname[feat1], ' and feature 2 = ', 1+feat2, '=', featname[feat2])

    X = X[:, [feat1,feat2]] # filter

    # Use regularization
    C = 1e-1    
    # Use no regularization
    #C = None
    lm = LinearMachine(C=C)
    lm.fit(X, y)
    
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    h = .02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max] x [y_min, y_max].

    xyplane = np.c_[xx.ravel(), yy.ravel()]
    #print('xyplane.shape=', xyplane.shape, '\nxyplane=\n', xyplane)

    y_pred = lm.predict(xyplane)
    #print('y_pred.shape=', y_pred.shape, '\ny_pred=\n', y_pred)
    Z = y_pred
    print('Z=\n', Z, 'shape=', Z.shape)
    numpixels = Z.shape[0]
    Zscores = lm.decision_function(xyplane)
    print('Zscores=\n', Zscores, 'shape=', Zscores.shape)
    scores = np.zeros(numpixels)
    for i in range(numpixels):
        ymax = np.argmax(Zscores[i])
        scores[i] = Zscores[i, ymax] # + ymax
    print('scores=\n', scores, 'shape=', scores.shape)
    S = scores.reshape(xx.shape)
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    fig, ax = plt.gcf(), plt.gca()
    mesh = plt.pcolormesh(xx, yy, S, cmap='RdBu', shading='auto',
                   norm=colors.Normalize(min(scores), max(scores)), zorder=0)
    #plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes')#, norm=colors.Normalize(0., 1.), zorder=0)
    fig.colorbar(mesh, ax=ax)

    plt.contour(xx, yy, Z, [0.5], linewidths=1., colors='white')
    plt.contour(xx, yy, Z, [1], linewidths=1., colors='white')

    # #######################################################
    # New Colormap
    cmap = colors.LinearSegmentedColormap(
        'red_blue_classes',
        {'red': [(0, 1, 1), (1, 0.7, 0.7)],
         'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
         'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
    plt.cm.register_cmap(cmap=cmap)

    cmap = plt.cm.gnuplot
    cmap = 'RdBu'
    color = ['r', 'g', 'b']

    # https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/contourf_demo.html
    #cs = plt.contourf(xx, yy, Z, cmap=cmap)
    #cs.cmap.set_under('yellow')
    plt.tight_layout()

    # Plot also the training points
    for i in range(numclasses):
    #for i, c in zip(classset, color):
        idx = np.where(y == i)
        #print('i=', i, 'allclasses[i]=', allclasses[i], 'idx=', idx)
        Xi = X[idx]
        plt.scatter(Xi[:, 0], Xi[:, 1], c=color[i], label=allclasses[i], edgecolor='black', s=20)
        yi_pred = lm.predict(Xi)
        #print('Class #', i+1, '=', classes[i],'Xi=\n', Xi, 'shape=', Xi.shape, 'yi_pred=\n',yi_pred, 'shape=', yi_pred.shape)

    y_pred = lm.predict(X)
    #print('y_pred=', y_pred)

    print('\n==========> Resubstitution of training data with only two features:\n')
    print('Classification Report for all features and all classes: ')
    print(classification_report(y, y_pred, target_names=classes, digits=3))
    print('Accuracy=', '%.2f %%' % (100*accuracy_score(y, y_pred)))
    print('Confusion Matrix: ')
    print(confusion_matrix(y, y_pred))

    params = lm.get_params(deep=True)
    W, bias, weights = params['W'], params['bias'], params['weights']
    print('W=', W)
    print('bias=\n', bias)
    print('weights=\n', weights)

    # plot class specific weight vectors (similarity vectors?)
    plot_class_specific = True
    plot_class_specific = False
    if plot_class_specific:
        coef = weights.T
        #intercept = bias
        for i in range(numclasses):
            #print('class=', i, 'color=', colors[i])
            plot_hyperplane(i, color[i], coef, bias)

    Cstr = ' --- No Regularization'
    if not lm.C is None:
        Cstr = ' --- Regularization C=' + str(lm.C)
    plt.title('Linear Machine' + Cstr, fontsize=10)

    plt.xlabel(featname[feat1])
    plt.ylabel(featname[feat2])

    plt.legend()
    print('The class specific hyperplanes cannot be interpreted as one-against-all hyperplanes.')

    plt.show()


def test_basic():
    testX = np.array([[3, 1.1], [4, 0.5], [3.75, 1.21], [1, 0], [3.5, 1]])

    print('---R E G R E S S I O N ---')
    testY = np.array([[1.0, 1.1, 0.1], [4, 0.5, 0.3], [3.75, 1.21, 9.3], [1, 0, 2], [3.5, 1, 2]])

    lm = LinearMachine(mode='regressor')
    lm.fit(testX, testY)

    print('---R E G R E S S I O N ---')
    print('TEST: INPUT=\n', testX)
    print('TEST DECISION FUNCTION:\n', lm.decision_function(testX))
    print('TEST SCORE:\n', lm.score(testX, testY))
    print('TEST PREDICT:\n', lm.predict(testX))

    print('---R E G R E S S I O N ---')
    testY = np.array([3, 2, 2, 1, 1])

    lm = LinearMachine(mode='classifier')
    lm.fit(testX, testY)
    print('TEST DECISION FUNCTION:\n', lm.decision_function(testX))
    print('TEST SCORE:\n', lm.score(testX, testY))
    print('TEST PREDICT:\n', lm.predict(testX))


def main():
    print('Executing main() ....')
    test_basic() #; myexit()
    test_iris_resubstitution()
    test_iris_k_fold()
    test_iris_visualize()

if __name__ == "__main__":
    main()
