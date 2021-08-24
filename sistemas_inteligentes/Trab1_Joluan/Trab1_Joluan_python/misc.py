# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 10:44:36 2021

@author: jolua
"""

from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from numpy import mean, std
from scipy import stats
import numpy as np

def ciclos(metodo):
    scalar = StandardScaler()
    pipeline = Pipeline([('transformer', scalar), ('estimator', metodo)])

    grade={'estimator__n_estimators' : [10, 25, 50, 100]}

    gs = GridSearchCV(estimator=pipeline, param_grid = grade, 
                      scoring='accuracy', cv = 4)

    rkf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=36851234)

    return gs,rkf

def ciclos_hg(metodo):
    scalar = StandardScaler()
    pipeline = Pipeline([('transformer', scalar), ('estimator', metodo)])

    grade={'estimator__n_samples' : [1, 3, 5, 7]}

    gs = GridSearchCV(estimator=pipeline, param_grid = grade, 
                      scoring='accuracy', cv = 4)

    rkf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=36851234)

    return gs,rkf


#função para calcular a média das acuracias, desvio padrão, intervalo a 95% inferior e superior
def scor(scores):
    means = mean(scores)
    stds = std(scores)
    inf, sup = stats.norm.interval(0.95, loc=means,
                               scale=stds/np.sqrt(len(scores)))
    return means,stds,inf,sup
