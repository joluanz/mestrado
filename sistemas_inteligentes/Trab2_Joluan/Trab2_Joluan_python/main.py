# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 07:37:28 2021

@author: jolua
"""

from sklearn import datasets
from classificadores import ExecutaClassificador

#wine = datasets.load_wine()
#ExecutaClassificador(wine, "wine")

breastcancer = datasets.load_breast_cancer()
ExecutaClassificador(breastcancer, "breast_cancer")

#digits = datasets.load_digits()
#ExecutaClassificador(digits, "digits")