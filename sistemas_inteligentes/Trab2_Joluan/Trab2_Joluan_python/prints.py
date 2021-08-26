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


def boxplot(scores_hgpooling_hillclimbing, scores_hgpooling_simulatedannealing, scores_hgpooling_genetic, dataset_name):
  data_box = {'Hill Climbing':scores_hgpooling_hillclimbing,
                   'Simulated Annealing':scores_hgpooling_simulatedannealing,
                   'Genetic':scores_hgpooling_genetic}
  
  df_data_box = pd.DataFrame(data_box)
  
  plt.figure() 
  boxplot = sns.boxplot(data=df_data_box)
  arquivo = dataset_name + '.png'
  plt.savefig(arquivo)
  

def testehipotese(scores_hgpooling_hillclimbing, scores_hgpooling_simulatedannealing, scores_hgpooling_genetic, dataset_name):
    # Paired t Test
    s,pt_hc_sa = ttest_rel(scores_hgpooling_hillclimbing, scores_hgpooling_simulatedannealing)
    s,pt_hc_ge = ttest_rel(scores_hgpooling_hillclimbing, scores_hgpooling_genetic)
    s,pt_sa_ge = ttest_rel(scores_hgpooling_simulatedannealing, scores_hgpooling_genetic)
    
    # Wilcoxon
    s,pw_hc_sa = wilcoxon(scores_hgpooling_hillclimbing, scores_hgpooling_simulatedannealing)
    s,pw_hc_ge = wilcoxon(scores_hgpooling_hillclimbing, scores_hgpooling_genetic)
    s,pw_sa_ge = wilcoxon(scores_hgpooling_simulatedannealing, scores_hgpooling_genetic)
        
    data_p = [["Hill Climbing",'{0:.5f}'.format(pt_hc_sa),'{0:.5f}'.format(pt_hc_ge)], 
                        ['{0:.5f}'.format(pw_hc_sa),"Simulated Annealing",'{0:.5f}'.format(pt_sa_ge)],
                        ['{0:.5f}'.format(pw_hc_ge),'{0:.5f}'.format(pw_sa_ge),"Genetic"]]
    
    df_data_p = pd.DataFrame(data_p)
    print("\nTestes de hipotese na base",dataset_name)
    print(df_data_p.T)


def tabelascores(scores_hgpooling_hillclimbing, scores_hgpooling_simulatedannealing, scores_hgpooling_genetic, dataset_name):
    data_tabela = {'Método':['Acuracia Media', 'Desvio Padrao', 'Inferior', 'Superior'],
                'Hill Climbing':scor(scores_hgpooling_hillclimbing),
                'Simulated Annealing':scor(scores_hgpooling_simulatedannealing),
                'Genetic':scor(scores_hgpooling_genetic)}
    
    df_data_tabela = pd.DataFrame(data_tabela)
    print("\nTabela das médias das acurácias da base",dataset_name)
    print(df_data_tabela.T)