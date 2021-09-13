# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import sklearn.datasets as datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import expit, softmax

_actfunc = {'logistic': expit, 'tanh': np.tanh, 'softmax': softmax,
            'relu': lambda x: np.maximum(x, 0), 'identity': lambda x: x}


def MLP_classifier(X, y, Y, classname):
    """Use MLP regressor as classifier."""
    print('Model: Multilayer perceptron as classifier')

    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=None)
    '''
    # Leave-One-Out:
    n_splits = len(y)
    skf = KFold(n_splits=n_splits, shuffle=True)
    skf = LeaveOneOut()
    #skf.get_n_splits(X, y)
    #print skf
    '''

    # instantiate learning model (k = 3)
    model = MLPRegressor(hidden_layer_sizes=(9,), activation='logistic',
                         solver='sgd', max_iter=5000, tol=1e-6)

    y_pred_overall = []
    y_test_overall = []
    i = 0
    for train_index, test_index in skf.split(X, y):
        print('Fold #', i+1, 'de', n_splits, '\nTRAIN:',
              train_index, '\nTEST:', test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        # model.fit(X_train, y_train)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        dimY = Y_pred.shape
        # maxpos = Y_pred.index(max(Y_pred))
        Y_pred_discrete = np.zeros(dimY)
        winner = np.argmax(Y_pred, axis=1)
        for j in range(len(winner)):
            Y_pred_discrete[j, winner[j]] = 1

        # print('Y_test=', Y_test, '\nY_pred=', Y_pred, '\nY_pred_discrete=', Y_pred_discrete)
        i += 1

        y_pred_overall = np.concatenate([y_pred_overall, winner])
        y_test_overall = np.concatenate([y_test_overall, y_test])

    '''
        print(classification_report(y_test, y_pred, target_names=classname))
        print(f1_score(y_test, y_pred, average='micro'))
        print(f1_score(y_test, y_pred, average='macro'))
        print(cv_cm)
    '''

    print('y_test_overall=', y_test_overall, '\ny_pred_overall=',
          y_pred_overall)
    print('MLP Classification Report: ')
    print('y_test_overall=', y_test_overall, 'y_pred_overall=',
          y_pred_overall, 'classname=', classname)
    print(classification_report(y_test_overall, y_pred_overall,
                                target_names=classname, digits=3))
    print('Accuracy=', '%.2f %%' % (100*accuracy_score(y_test_overall,
                                                       y_pred_overall)))
    print('MLP Confusion Matrix: ')
    print(confusion_matrix(y_test_overall, y_pred_overall))


def peep_MLPregressor(model):
    """Show parameters of MLP."""
    print('=== MLP REGRESSOR ===')
    params = model.get_params()
    print('Parameters=', params)

    print('=== Attributes ===')
    print('Loss =', model.loss_, 'after', model.n_iter_, 'iterations')
    W = model.coefs_
    b = model.intercepts_
    num_layers = len(b)

    for k in range(num_layers):
        W = model.coefs_[k]
        b = model.intercepts_[k]
        with np.printoptions(precision=3, suppress=True):
            print('W[', k, ']=\n', W, 'shape=', W.shape)
            print('b[', k, ']=\n', b, 'shape=', b.shape)

    print('Number of layers=', model.n_layers_)
    print('Number of outputs=', model.n_outputs_)
    print('Activation function=', model.out_activation_)

    # Plot the 'loss_curve_' property of the model
    fig, ax = plt.gcf(), plt.gca()
    ax.semilogy(model.loss_curve_)
    ax.set(xlabel='Epoch', ylabel='Loss',
           title='Learning curve of MLP')
    ax.grid()
    plt.show()


def MLP_autoencoder(X, H=2, activation='logistic'):
    """Create one hidden layer as encoder."""
    print('Model: Multilayer perceptron as autoencoder')
    n, d = X.shape
    # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor
    model = MLPRegressor(hidden_layer_sizes=(H,), activation=activation,
                         max_iter=10000, tol=1e-6)
    model.fit(X, X)

    WX = np.dot(X, model.coefs_[0]) + model.intercepts_[0]
    activation = _actfunc[model.activation]
    print('activation function = ', model.activation)

    PHI = activation(WX)
    # print('PHI = activation(W.x+b) =\n', PHI, 'shape=', PHI.shape)
    return model, PHI


def plot_encoded(PHI, y, classname, activation):
    """Plot the encoded features of two or three dimensions."""
    n, d = PHI.shape
    fig, ax = plt.gcf(), plt.gca()
    if d == 3:
        ax = Axes3D(fig, elev=-150, azim=110)
        ax.set(zlabel='Nonlinear PC 3')

    color = ['r', 'g', 'b']
    for i in range(len(classname)):
        idx = np.where(y == i)
        Xi = PHI[idx]
        ax.scatter(Xi[:, 0], Xi[:, 1], c=color[i], label=classname[i],
                   edgecolor='black', s=20)

    ax.set(title='Encoded features, activation='+activation)
    ax.set(xlabel='Nonlinear PC 1')
    ax.set(ylabel='Nonlinear PC 2')

    ax.legend()
    plt.show()


def main():
    iris = datasets.load_iris()
    featname = iris['feature_names']
    X = iris.data
    y = iris.target
    classname = iris.target_names
    lb = LabelBinarizer()
    lb.fit(y)
    Y = lb.transform(y)

    # MLP_classifier( X, y, Y, classname )
    # Augmented X is not necessary, since MLP includes bias automatically
    # X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    activation = 'tanh'
    activation = 'softmax'
    activation = 'identity'
    activation = 'relu'
    activation = 'logistic'
    '''
    '''

    model, PHI = MLP_autoencoder(X, H=2, activation=activation)
    reconstruction_error = model.score(X, X)
    print('reconstruction_error=', reconstruction_error)
    Xpredict = model.predict(X)
    diff = np.linalg.norm(X-Xpredict)
    print('X-X_predict=\n,', X-Xpredict, '\n||X-X_predict||=', diff)

    '''


    H = 2
    # activation = 'relu'
    model, PHI = MLP_autoencoder(X, H=H, activation=activation)
    peep_MLPregressor(model)
    plot_encoded(PHI, y, classname, activation)

    reconstruction_error = model.score(X, X)
    print('reconstruction_error=', reconstruction_error)

    H = 3
    # activation = 'tanh'
    model, PHI = MLP_autoencoder(X, H=H, activation=activation)
    peep_MLPregressor(model)
    plot_encoded(PHI, y, classname, activation)

    reconstruction_error = model.score(X, X)
    print('reconstruction_error=', reconstruction_error)


    '''


if __name__ == '__main__':
    main()
