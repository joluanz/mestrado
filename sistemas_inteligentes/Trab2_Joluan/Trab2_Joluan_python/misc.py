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
from collections import Counter
import numpy as np
import pandas as pd

def ciclos_hg(metodo):
    scalar = StandardScaler()
    pipeline = Pipeline([('transformer', scalar), ('estimator', metodo)])

    grade={'estimator__n_samples' : [3, 5, 7]}

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

def predicao(predicts,ordem_classes):
  dados_todos = pd.DataFrame(data=predicts)
        
  #predicao: array que será retornado pelo método predict
  #contendo cada uma das predição(para o hgpooling) para cada exemplo do conjunto de teste
  predicao = []
  
  for index in dados_todos: #loop para percorrer cada exemplo do conjunto de teste
      coluna = dados_todos[index] #coluna: todos os predicts dos classificadores base para um exemplo do teste
      
      #mostc: vai ser nx2 => na primeira coluna[0] vai ter a classe, na segunda[1] a quantidade
      #em ordem decrescente
      mostc = Counter(coluna).most_common() 

      #a condição a seguir cria um array empatados[]
      #para salvar os primeiros colocados entre 
      empatados = []
      if len(mostc) == 1: #caso só exista uma classe, não tem como ter empate, logo retorna ela mesma
          empatados.append(mostc[0][0])
      else: #caso tenha mais que uma classe, verifica se tem quantidades iguais
          for ind in range(len(mostc)-1):
              if len(mostc) == 1:
                  empatados.append(mostc[ind][0])
                  break
              else:
                  if mostc[ind][1] == mostc[ind+1][1]:
                      if empatados == []:    
                          empatados.append(mostc[ind][0])
                          empatados.append(mostc[ind+1][0])
                      else:
                          empatados.append(mostc[ind+1][0])
                  else:
                      if not empatados:
                          empatados.append(mostc[0][0])
                          break
                      else:
                          break
      #decisao: variavel auxiliar para armazenar o predict do atual exemplo
      #caso tenha mais de um no array empatados: vai usar o array ordem_classes para desempatar
      if len(empatados) != 1:                    
          for classe_ord in ordem_classes: #ordem_classes = array com a ordem para desempate obtido na base de treino original
              for classe_pred in empatados: #empatados= array de classes empatadas
                  #o desempate é feito escolhendo entre os empatados o primeiro que aparecer na lista de ordem obitidos
                  if classe_ord == classe_pred: 
                      decisao = classe_pred
                      break
              else:
                  continue
              break
      else: #com apenas um, 
          decisao = empatados[0]
              
      predicao.append(decisao)
      
  #predict = np.asarray(predicao)
  #print("predicao:",predicao)
  return np.asarray(predicao)