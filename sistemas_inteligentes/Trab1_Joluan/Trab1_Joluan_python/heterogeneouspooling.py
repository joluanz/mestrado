# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 09:25:32 2021

@author: jolua
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y
from collections import Counter

#definido o classificador heterogeneouspooling
class HeterogenousPoolingClassifier(BaseEstimator):
    def __init__(self, n_samples=None):
        super().__init__()
        self.n_samples = n_samples #apenas um hiperparametro
    
    #método fit() para treinamento dos classificadores base    
    def fit(self,x_train,y_train):
        classificadores = [] #array classifiacdores: array para salvar os classificadores base depois de treinados
        x_train,y_train = check_X_y(x_train,y_train)
        
        #_ordem_classes: array com as classes da base de treino original, em ordem decrescente(utlizado para desempate)
        df_aux = pd.DataFrame(data=y_train)
        df_ordem = df_aux.value_counts(df_aux[0])
        dados_ordem = pd.DataFrame(data=df_ordem)
        self._ordem_classes  = dados_ordem.index.array
        
        #loop para criar novas bases de treino a partir da original, e ir fitando os classifiacdores base
        for i in range(self.n_samples):
            if i == 0:
                sub_X, sub_y = x_train, y_train
            else:
                sub_X, sub_y = resample(x_train, y_train, random_state=i-1)

            self.DT = DecisionTreeClassifier()
            self.DT.fit(sub_X,sub_y)
            classificadores.append(self.DT)
            
            self.gNB = GaussianNB()
            self.gNB.fit(sub_X,sub_y)
            classificadores.append(self.gNB)
            
            self.kNN = KNeighborsClassifier(n_neighbors=1)
            self.kNN.fit(sub_X,sub_y)
            classificadores.append(self.kNN)
        
        self._classificadores = classificadores

    #método predict(): usado para predizer a classe dos exemplos do conjunto de teste
    def predict(self,x_test):
        ordem_classes = self._ordem_classes
        
        #predicts: vai conter uma lista com todos os predicts para cada classificador base
        predicts = []
        for classificador in self._classificadores:
            predict = classificador.predict(x_test)
            predicts.append(predict)
      
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
        return np.asarray(predicao)

