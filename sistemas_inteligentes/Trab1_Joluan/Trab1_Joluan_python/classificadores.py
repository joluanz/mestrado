# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 10:41:37 2021

@author: jolua
"""
from sklearn.ensemble import BaggingClassifier,AdaBoostClassifier,RandomForestClassifier
from sklearn.model_selection import cross_val_score
from misc import ciclos, ciclos_hg
from heterogeneouspooling import HeterogenousPoolingClassifier
from prints import boxplot,testehipotese,tabelascores

class Bagging():
  def __init__(self):
    pass

  def execute(self, dataset):
      dataset_X = dataset.data
      dataset_y = dataset.target
      
      metodo = BaggingClassifier()
      
      gs,rkf = ciclos(metodo)

      scores = cross_val_score(gs, dataset_X, dataset_y, scoring='accuracy', 
                               cv = rkf)
      
      return scores

class AdaBoost():
  def __init__(self):
    pass

  def execute(self, dataset):
      dataset_X = dataset.data
      dataset_y = dataset.target
      
      metodo = AdaBoostClassifier()
      
      gs,rkf = ciclos(metodo)

      scores = cross_val_score(gs, dataset_X, dataset_y, scoring='accuracy', 
                               cv = rkf)
      
      return scores

class RandomForest():
  def __init__(self):
    pass

  def execute(self, dataset):
      dataset_X = dataset.data
      dataset_y = dataset.target
      
      metodo = RandomForestClassifier()
      
      gs,rkf = ciclos(metodo)
      
      scores = cross_val_score(gs, dataset_X, dataset_y, scoring='accuracy', 
                               cv = rkf)

      return scores

class HeterogeneousPooling():
  def __init__(self):
    pass

  def execute(self, dataset):
      dataset_X = dataset.data
      dataset_y = dataset.target
      
      metodo = HeterogenousPoolingClassifier()
      
      gs,rkf = ciclos_hg(metodo)

      scores = cross_val_score(gs, dataset_X, dataset_y, scoring='accuracy', 
                               cv = rkf)
      
      return scores
  
def ExecutaClassificador(dataset, dataset_name):
  bagging = Bagging()
  adaboost = AdaBoost()
  randomforest = RandomForest()
  hgpooling = HeterogeneousPooling()
  
  scores_bagging = bagging.execute(dataset)
  scores_adaboost = adaboost.execute(dataset)
  scores_randomforest = randomforest.execute(dataset)
  scores_hgpooling = hgpooling.execute(dataset)
  
  
  boxplot(scores_bagging, scores_adaboost, scores_randomforest, scores_hgpooling, dataset_name)
  tabelascores(scores_bagging, scores_adaboost, scores_randomforest, scores_hgpooling, dataset_name)
  testehipotese(scores_bagging, scores_adaboost, scores_randomforest, scores_hgpooling, dataset_name)



  
  
  