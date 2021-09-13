"""========================================
Extreme Learning Machine
Author: Thomas W. Rauber mailto:trauber@gmail.com
========================================"""

import locale
import copy
import numpy as np
from scipy.special import expit, softmax
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.manifold import TSNE

from Iris import Iris
from LabelBinarizer2 import LabelBinarizer2
from util import cm2inch, tex_setup, myexit
from datetime import datetime
datetime = datetime.now().strftime('%Y_%m_%d__%H.%M.%S')  # avoid ":"


_actfunc = {'logistic': expit, 'tanh': np.tanh, 'softmax': softmax,
            'relu': lambda x: np.maximum(x, 0),
            'identity': lambda x: x, 'sign': np.sign}

class ExtremeLearningMachine(BaseEstimator, RegressorMixin, ClassifierMixin):
    """ Extreme Learning Machine
    Parameters
    ----------

    Attributes
    ----------
    """

    def __init__(self, L=100, activation='logistic', C=0.0,
                 seed=66649, mode='classifier'):
        # print('Executing __init__() ....', 'L=', L, 'C=', C)
        self.language_code, self.encoding = locale.getlocale()
        self.W = None
        self.Whidden = None
        self.biashidden = None
        self.L = L    # default number of hidden nodes
        self.C = C    # Regularization hyperparameter: 0 = no regularization
        self.seed = seed
        self.mode = mode  # Act as regressor (continuous output) or classifier (class output: label or one-hot-encoded)
        if L==0:
            print('WARNING: Number of hidden neuron zero') #; quit()
        self.activation = activation
        self.afun = _actfunc[activation]
        self.lb = None
        if mode == 'classifier':
            self.lb = LabelBinarizer2()

    def _pinv_regularized(self, Phi, C):
        ''' Regularized pseudoinverse
        '''
        return np.dot(np.linalg.inv(np.dot(Phi.T, Phi) + np.eye(Phi.shape[1]) / C), Phi.T)

    def fit(self, X, y, Whidden=None, biashidden=None):
        n, m = X.shape
        
        
        # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html
        # if output is multidimensional, only matrix with 0 and 1 is allowed
        if self.mode == 'classifier':
            self.lb.fit(y)
            Y = self.lb.transform(y)
        else:
            Y = y

        np.random.seed(self.seed)

        if not Whidden is None:
            assert Whidden.shape[1]==self.L, 'W shape =' + str(Whidden.shape)+' not ok. Number of hidden unnits= '+str(self.L)
            print('Use precalculated hidden weights and biases')
            self.Whidden = Whidden
            self.biashidden = biashidden
        else:
            self.Whidden = np.random.randn(m, self.L)
            self.biashidden = np.random.randn(1, self.L)
        activation = np.dot(X, self.Whidden) + self.biashidden
        #print('activation=', activation, 'shape=', activation.shape)#; quit()
        Phi = self.afun(activation)
        if self.C == 0.0:
            PHIpinv = np.linalg.pinv(Phi)
        else:
            PHIpinv = self._pinv_regularized(Phi, self.C)
        #print('PHIpinv=\n', PHIpinv, 'shape=', PHIpinv.shape)
        self.W = np.dot(PHIpinv, Y)

    def _input2hidden(self, X):
        activation = np.dot(X, self.Whidden) + self.biashidden
        Phi = self.afun(activation)
        return Phi

    def decision_function(self, X, y=None):   # y is dummy to avoid error in pipeline
        activation = np.dot(X, self.Whidden) + self.biashidden
        Phi = self.afun(activation)
        Y = np.dot(Phi, self.W)
        return Y

    def predict(self, X):
        if self.mode == 'classifier':
            Y = self.decision_function(X)
            y_pred = self.lb.inverse_transform(Y)
        else:
            y_pred = self.decision_function(X)
        #print('\nY=\n', Y, 'shape=', Y.shape, '\ny_pred=\n', y_pred, 'shape=', y_pred.shape)
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
        # activation = np.dot(X, self.Whidden) + self.biashidden
        # Phi = self.afun(activation)
        # Yhat = np.dot(Phi, self.W)
        # return np.linalg.norm(Y - Yhat)

    def predict_old(self, X):
        activation = np.dot(X, self.Whidden) + self.biashidden
        Phi = self.afun(activation)
        Y = np.dot(Phi, self.W)
        y_pred = self.lb.inverse_transform(Y)
        #print('\nY=\n', Y, 'shape=', Y.shape, '\ny_pred=\n', y_pred, 'shape=', y_pred.shape)
        return y_pred

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
        #print('ExtremeLearningMachine: score values\n', values)
        # compute the likelihood of the underlying gaussian models
        # up to a multiplicative constant.
        likelihood = np.exp(values - values.max(axis=1)[:, np.newaxis])
        # compute posterior probabilities
        proba = likelihood / likelihood.sum(axis=1)[:, np.newaxis]
        #print('ExtremeLearningMachine: proba\n', proba)
        return proba




def iris_resubstitution():
    # Get iris data
    '''
    iris = datasets.load_iris()
    X = iris.data
    labels = iris.target
    classes = iris.target_names
    y = labels
    classset = np.unique(y)
    numclasses = len(classes)
    featname = iris.featname
    '''

    iris = Iris()
    X = iris.X
    y = iris.y
    classname = iris.classname

    elm = ExtremeLearningMachine(L=10, activation='identity')
    # print('elm.get_params().keys():', elm.get_params().keys()); quit()

    #elm = ExtremeLearningMachine(L=10, activation='tanh')
    elm.fit(X, y)
    y_pred = elm.predict(X)

    class_report = classification_report(y, y_pred, target_names=classname, digits=3)
    if iris.language_code == 'pt_BR':
        print('\n==========> Resubstituição de dados de treino:\n')
        print('Relatório de classificação para todas as características e todas as classes: ')
        print(class_report)
        print('Acurácia=', '%.2f %%' % (100*accuracy_score(y, y_pred)))
        print('Matriz de confusão: ')

    else:
        print('\n==========> Resubstitution of training data:\n')
        print('Classification Report for all features and all classes: ')
        print(class_report)
        print('Accuracy=', '%.2f %%' % (100*accuracy_score(y, y_pred)))
        print('Confusion Matrix: ')
    print(confusion_matrix(y, y_pred))


def iris_k_fold(K=10, L=100):
    # Get iris data
    iris = Iris()
    X = iris.X
    y = iris.y
    classname = iris.classname
    numclasses = len(classname)
    featname = iris.featname

    skf = StratifiedKFold(n_splits=K, shuffle=True)
    #skf.get_n_splits(X, y)
    #print skf
    y_pred_overall = []
    y_test_overall = []

    #elm = ExtremeLearningMachine(L=L, activation='identity')
    elm = ExtremeLearningMachine(L=L, activation='relu')
    # print('elm.get_params().keys():', elm.get_params().keys()); quit()

    for train_index, test_index in skf.split(X, y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        elm.fit(X_train, y_train)
        y_pred = elm.predict(X_test)

        y_pred_overall = np.concatenate([y_pred_overall, y_pred])
        y_test_overall = np.concatenate([y_test_overall, y_test])

    accuracy = accuracy_score(y_test_overall, y_pred_overall)
    f1macro = f1_score(y_test_overall, y_pred_overall, average='macro')
    f1micro = f1_score(y_test_overall, y_pred_overall, average='micro')
    class_report = classification_report(y_test_overall, y_pred_overall, target_names=classname, digits=3)
    if iris.language_code == 'pt_BR':
        print('\n==========> Validação cruzada K-fold:\n')
        print('Relatório de classificação para todas as características e todas as classes: ')
        print(class_report)
        print('Acurácia=', '%.2f %%' % (100*accuracy))
        print('F1 Média Macro=', '%.3f' % f1macro)
        print('F1 Média Micro=', '%.3f' % f1micro)
        print('Matriz de confusão: ')
    else:
        print('\n==========> K-fold cross validation:\n')
        print('Extreme Learning Machine ', K, '- Fold Classification Report: ')
        print(class_report)
        print('Accuracy=', '%.2f %%' % (100*accuracy))
        print('Macro-averaged F1=', '%.3f' % f1macro)
        print('Micro-averaged F1=', '%.3f' % f1micro)
        print('Extreme Learning Machine Confusion Matrix: ')
    print (confusion_matrix(y_test_overall, y_pred_overall))

    return accuracy, f1macro, f1micro


def plot_tSNE(X, y, classname, numclasses=1, n_components=2):
    print ('Generating tSNEplot...')
    #print ('plot_tSNE> classname=', classname, 'numclasses=', numclasses)
    X_embedded = TSNE(n_components=n_components).fit_transform(X)
    Xplot = X_embedded
    xlab = 'tSNE Embedded dim 1'
    ylab = 'tSNE Embedded dim 2'
    tit = 'tSNE Plot'
    dimX = X.shape[1]
    if dimX == n_components:
        tit += ' --- plotting original feature space'
    #print('Xplot.shape=', Xplot.shape, 'y.shape=', y.shape, 'type(y)=', type(y),
    #      'Xplot.size=', Xplot.size, 'y.size=', y.size)
    #print('y=', y)
    cmap = plt.get_cmap('gnuplot')
    color = [np.array([cmap(i)]) for i in np.linspace(0, 1, numclasses)]
    if n_components==2:

        for i in range(len(classname)):
            idx = np.where(y == i)
            plt.scatter(Xplot[idx, 0], Xplot[idx, 1], c=[color[i]], label=classname[i])

    if n_components==3:
        zlab = 'tSNE Embedded dim 3'
        fig = plt.figure(1, figsize=(8, 6))

        ax = Axes3D(fig, elev=-150, azim=110)
        ax.set_title(tit)
        ax.set_xlabel(xlab)
        #ax.w_xaxis.set_ticklabels([])
        ax.set_ylabel(ylab)
        #ax.w_yaxis.set_ticklabels([])
        ax.set_zlabel(zlab)
        #ax.w_zaxis.set_ticklabels([])
        for i in range(numclasses):
            idx = np.where(y == i)
            ax.scatter(Xplot[idx, 0], Xplot[idx, 1], Xplot[idx, 2],
                    c=[color[i]], label=classname[i]) 
            plt.title(tit)
    plt.legend(loc=2)
    plt.show()


def plot_hidden_tSNE(L=30, n_components=3):
    elm = ExtremeLearningMachine(L=L)
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    classes = iris.target_names
    numclasses = len(classes)

    elm.fit(X, y)
    Phi = elm._input2hidden(X)
    plot_tSNE(Phi, y, classes, numclasses=numclasses, n_components=n_components)


def plot_regions(ax, iris, xx, yy, Z, plot_reg, L, C, activation, seed):
    X = iris.X
    y = iris.y
    classes = allclasses = iris.classname
    numclasses = len(classes)
    features = iris.features
    featname = iris.featname

    cmap = 'RdBu'
    cmap = plt.cm.bone
    cmap = 'Blues'
    colors = ['r', 'g', 'b']
    cmap = plt.cm.gnuplot

    cmap = copy.copy(cmap)


    # https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/contourf_demo.html
    cs = ax.contourf(xx, yy, Z, cmap=cmap)
    cs.cmap.set_under('yellow')
    plt.tight_layout()

    # Plot also the training points
    for i in range(numclasses):
        #for i, color in zip(classset, colors):
        idx = np.where(y == i)
        #print('i=', i, 'allclasses[i]=', allclasses[i], 'idx=', idx)
        Xi = X[idx]
        ax.scatter(Xi[:, 0], Xi[:, 1], c=colors[i], label=allclasses[i], edgecolor='black', s=20)
        #yi_pred = elm.predict(Xi)
        #print('Class #', i+1, '=', classes[i],'Xi=\n', Xi, 'shape=', Xi.shape, 'yi_pred=\n',yi_pred, 'shape=', yi_pred.shape)

    if iris.language_code == 'pt_BR':
        Cstr = ' --- Sem Regularização'
        tit = 'Neurônios ocultos: ' + str(L) + ' --- ativação: ' + activation\
        + ' --- semente: ' + str(seed)
    else:
        Cstr = ' --- No Regularization'
        tit = 'Hidden neurons: ' + str(L) + ' --- activation: ' + activation\
        + ' --- seed: ' + str(seed)
    if plot_reg:
        if iris.language_code == 'pt_BR':
            Cstr = ' --- Regularização: C=' + str(C)
        else:
            Cstr = ' --- Regularization: C=' + str(C)
        ax.set(ylabel='$x_{' + str(features[1]+1) + '}$ = ' + featname[1])

    ax.set(xlabel='$x_{' + str(features[0]+1) + '}$ = ' + featname[0])
    tit += Cstr
    ax.set(title=tit)
    ax.legend(loc='lower right')



def iris_visualize(feat1=2, feat2=3, L=10, C=0.0, seed=66649,
                   rootdir = '/export/thomas/experimental_results/'):
    # Get iris data
    iris = Iris(features=(feat1,feat2))
    X = iris.X
    #X = X[:, [feat1,feat2]] # filter
    y = iris.y
    classes = allclasses = iris.classname
    numclasses = len(classes)
    featname = iris.featname
    actfun = 'sign'
    actfun = 'logistic'
    actfun = 'relu'
    actfun = 'identity'
    actfun = 'tanh'

    '''
    iris = datasets.load_iris()
    X = iris.data
    labels = iris.target
    y = labels
    classes = allclasses = iris.target_names
    '''

    '''
    # Only last two classes
    X = X[-100:]
    y = y[-100:]
    classes = classes[-2:]
    '''

    # classset = np.unique(y)
    # numclasses = len(classes)
    # featname = iris.featname

    print('\n\n===> V I S U A L I Z E  E X T R E M E  L E A R N I N G  M A C H I N E  D E C I S I O N  R E G I O N S <===\n\n')
    print('Iris data: Using only two features: feature 1 = ', 1+feat1, '=',
          featname[0], ' and feature 2 = ', 1+feat2, '=', featname[1])

    # plot_tSNE(X, y, classes, numclasses=numclasses, n_components=2) ; quit()

    elm = ExtremeLearningMachine(L=L, activation=actfun, C=C, seed=seed)
    elm.fit(X, y)

    plot_also_noreg = not elm.C == 0.0
    if plot_also_noreg:
        elm_noreg = ExtremeLearningMachine(L=L, activation=actfun)
        elm_noreg.fit(X, y, elm.Whidden, elm.biashidden)

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    h = .02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max] x [y_min, y_max].

    xyplane = np.c_[xx.ravel(), yy.ravel()]
    # print('xyplane.shape=', xyplane.shape, '\nxyplane=\n', xyplane)

    y_pred = elm.predict(xyplane)
    #print('y_pred.shape=', y_pred.shape, '\ny_pred=\n', y_pred)
    Z = y_pred
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    #print('Z=\n', Z)

    tex_setup(True)
    fig, ax = plt.gcf(), plt.gca()
    plt.axis('off')
    if plot_also_noreg:
        fig.set_size_inches(cm2inch(17), cm2inch(8))
        ax1 = fig.add_subplot(1,2,1)#, projection='3d')
        ax2 = fig.add_subplot(1,2,2)
        Z_noreg = (elm_noreg.predict(xyplane)).reshape(xx.shape)
        plot_regions(ax1, iris, xx, yy, Z, True, elm.L, elm.C,
                     elm.activation, elm.seed)
        plot_regions(ax2, iris, xx, yy, Z_noreg, False, elm.L, elm.C,
                     elm.activation, elm.seed)
    else:
        plot_regions(ax, iris, xx, yy, Z, False, elm.L, elm.C,
                     elm.activation, elm.seed)


    
    y_pred = elm.predict(X)
    #print('y_pred=', y_pred)

    print('\n==========> Resubstitution of training data with only two features:\n')
    print('Classification Report for all features and all classes: ')
    print(classification_report(y, y_pred, target_names=classes, digits=3))
    print('Accuracy=', '%.2f %%' % (100*accuracy_score(y, y_pred)))
    print('Confusion Matrix: ')
    print(confusion_matrix(y, y_pred))

    
    fileext = '.eps'
    fileext = '.pgf'
    outfigfilename = rootdir+'elm_L_'+str(elm.L)+'__C_'+str(elm.C)+datetime+fileext
    print('Saving figure to ', outfigfilename)
    #plt.savefig(outfigfilename)

    plt.show()


def test_basic():
    testX = np.array([[3, 1.1], [4, 0.5], [3.75, 1.21], [1, 0], [3.5, 1]])

    print('---R E G R E S S I O N ---')
    testY = np.array([[1.0, 1.1, 0.1], [4, 0.5, 0.3], [3.75, 1.21, 9.3], [1, 0, 2], [3.5, 1, 2]])

    elm = ExtremeLearningMachine(mode='regressor', activation='relu')
    elm.fit(testX, testY)

    print('---R E G R E S S I O N ---')
    print('TEST: INPUT=\n', testX)
    print('TEST DECISION FUNCTION:\n', elm.decision_function(testX))
    print('TEST SCORE:\n', elm.score(testX, testY))
    print('TEST PREDICT:\n', elm.predict(testX))

    print('---R E G R E S S I O N ---')
    testY = np.array([3, 2, 2, 1, 1])

    elm = ExtremeLearningMachine(mode='classifier')
    elm.fit(testX, testY)
    print('TEST DECISION FUNCTION:\n', elm.decision_function(testX))
    print('TEST SCORE:\n', elm.score(testX, testY))
    print('TEST PREDICT:\n', elm.predict(testX))


def main():
    print('Executing main() ....')
    test_basic() #; myexit()
    #plot_hidden_tSNE(L=100, n_components=2) ; quit()

    seed = 66649
    seed = None

    #iris_resubstitution()
    #iris_k_fold()
    
    Lmax = 50
    
    performance = np.zeros((Lmax+1,3))
    for L in range(Lmax+1):
        performance[L] = iris_k_fold(L=L)
        #print('performance[', L, ']=', performance[L])

    #print('performance=\n', performance)

    plt.xticks(np.arange(0, Lmax+1, step=1))
    plt.plot(range(Lmax+1), performance)
    language_code, encoding = locale.getlocale()
    if language_code == 'pt_BR':
        plt.xlabel('Quantidade de neurônios ocultos')
        plt.ylabel('Desempenho')
        plt.legend(('Acurácia [%]','F1 macro','F1 micro'))
    else:
        plt.xlabel('Number of hidden neurons')
        plt.ylabel('Performance')
        plt.legend(('Accuracy [%]','F1 macro','F1 micro'))
    plt.show()

    visualize_for = range(Lmax+1)
    visualize_for = (0,1,2,3,5,10,15,20,30,40)
    # visualize_for = (15, )
    
    seeds = (12345, 67890)
    seeds = (None,)
    C = 0.0
    C = 1/10.0

    for L in visualize_for:
        for s in seeds:
            iris_visualize(L=L, C=C, seed=s)

if __name__ == "__main__":
    main()
