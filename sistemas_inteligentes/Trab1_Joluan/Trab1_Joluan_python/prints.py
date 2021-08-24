# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 11:17:17 2021

@author: jolua
"""

import seaborn as sns
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
import matplotlib.pyplot as plt
from misc import scor

def boxplot(scores_bagging, scores_adaboost, scores_randomforest, scores_hgpooling, dataset_name):
  data_box = {'Bagging':scores_bagging,
                   'AdaBoost':scores_adaboost,
                   'RandomForest':scores_randomforest,
                   'Hg. Pooling':scores_hgpooling}
  
  df_data_box = pd.DataFrame(data_box)
  
  plt.figure() 
  boxplot = sns.boxplot(data=df_data_box)
  arquivo = dataset_name + '.png'
  plt.savefig(arquivo)
  

def testehipotese(scores_bagging, scores_adaboost, scores_randomforest, scores_hgpooling, dataset_name):
    # Paired t Test
    s,pt_bag_ada = ttest_rel(scores_bagging, scores_adaboost)
    s,pt_bag_rf = ttest_rel(scores_bagging, scores_randomforest)
    s,pt_bag_hg = ttest_rel(scores_bagging, scores_hgpooling)
    s,pt_ada_rf = ttest_rel(scores_adaboost, scores_randomforest)
    s,pt_ada_hg = ttest_rel(scores_adaboost, scores_hgpooling)
    s,pt_rf_hg = ttest_rel(scores_randomforest,scores_hgpooling)
    
    # Wilcoxon
    s,pw_bag_ada = wilcoxon(scores_bagging, scores_adaboost)
    s,pw_bag_rf = wilcoxon(scores_bagging, scores_randomforest)
    s,pw_bag_hg = wilcoxon(scores_bagging, scores_hgpooling)
    s,pw_ada_rf = wilcoxon(scores_adaboost, scores_randomforest)
    s,pw_ada_hg = wilcoxon(scores_adaboost, scores_hgpooling)
    s,pw_rf_hg = wilcoxon(scores_randomforest,scores_hgpooling)
        
    data_p = [["Bagging",'{0:.5f}'.format(pt_bag_ada),'{0:.5f}'.format(pt_bag_rf),'{0:.5f}'.format(pt_bag_hg)], 
                        ['{0:.5f}'.format(pw_bag_ada),"AdaBoost",'{0:.5f}'.format(pt_ada_rf),'{0:.5f}'.format(pt_ada_hg)],
                        ['{0:.5f}'.format(pw_bag_rf),'{0:.5f}'.format(pw_ada_rf),"Random Forest",'{0:.5f}'.format(pt_rf_hg)],
                        ['{0:.5f}'.format(pw_bag_hg),'{0:.5f}'.format(pw_ada_hg),'{0:.5f}'.format(pw_rf_hg),"Heterogeneous Pooling"]]
    
    df_data_p = pd.DataFrame(data_p)
    print("\nTestes de hipotese na base",dataset_name)
    print(df_data_p.T)


def tabelascores(scores_bagging, scores_adaboost, scores_randomforest, scores_hgpooling, dataset_name):
    data_tabela = {'Método':['Acuracia Media', 'Desvio Padrao', 'Inferior', 'Superior'],
                'Bagging':scor(scores_bagging),
                'AdaBoost':scor(scores_adaboost),
                'RandomForest':scor(scores_randomforest),
                'HeterogeneousPooling':scor(scores_hgpooling)}
    
    df_data_tabela = pd.DataFrame(data_tabela)
    print("\nTabela das médias das acurácias da base",dataset_name)
    print(df_data_tabela.T)