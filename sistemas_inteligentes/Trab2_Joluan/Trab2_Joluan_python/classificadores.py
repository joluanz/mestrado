# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 10:41:37 2021

@author: jolua
"""
from sklearn.model_selection import cross_val_score
from misc import ciclos_hg
from heterogeneouspooling import HeterogenousPoolingClassifier_Heuristico
from prints import boxplot,testehipotese,tabelascores

class HeterogeneousPooling():
  def __init__(self, heuristica=None):
    super().__init__()
    self.heuristica = heuristica

  def execute(self, dataset):
      dataset_X = dataset.data
      dataset_y = dataset.target
      
      metodo = HeterogenousPoolingClassifier_Heuristico(self.heuristica)
      
      gs,rkf = ciclos_hg(metodo)

      scores = cross_val_score(gs, dataset_X, dataset_y, scoring='accuracy', 
                               cv = rkf)
      
      return scores
  
def ExecutaClassificador(dataset, dataset_name):
  hgpooling_hillclimbing = HeterogeneousPooling('hillclimbing')
  hgpooling_simulatedannealing = HeterogeneousPooling('simulatedannealing')
  hgpooling_genetic = HeterogeneousPooling('genetic')
  
  scores_hgpooling_hillclimbing = hgpooling_hillclimbing.execute(dataset)
  scores_hgpooling_simulatedannealing = hgpooling_simulatedannealing.execute(dataset)
  scores_hgpooling_genetic = hgpooling_genetic.execute(dataset)
  
  
  boxplot(scores_hgpooling_hillclimbing, scores_hgpooling_simulatedannealing, scores_hgpooling_genetic, dataset_name)
  tabelascores(scores_hgpooling_hillclimbing, scores_hgpooling_simulatedannealing, scores_hgpooling_genetic, dataset_name)
  testehipotese(scores_hgpooling_hillclimbing, scores_hgpooling_simulatedannealing, scores_hgpooling_genetic, dataset_name)



  
  
  