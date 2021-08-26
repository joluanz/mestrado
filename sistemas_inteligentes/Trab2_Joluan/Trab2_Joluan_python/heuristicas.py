# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 10:08:37 2021

@author: jolua
"""
import time
from funcoes import *

def hillclimbing_deterministico(max_time,classificadores,x_test,y_train,ordem_classes):
        current_state = [0]*len(classificadores) #cria um state zerado [0,0,...,0] para usar como inicio
        optimal_acc = 0 #acuracia otima começa zerada junto com o state zerado
        optimal_state = current_state #estado otimo começa zerado junto com o state zerado
        valid_states = len(classificadores) #inicia valid_states, será usada como critério de parada da heuristica
        
        start = time.process_time() #start: tempo inicial
        end = 0                     #end:   tempo final

        #critérios de parada:
        #valid_states == 0 -> quando nenhum dos estados possiveis foi melhor que um estado de uma iteração anterior
        #end-start < max_time ->
        while valid_states != 0 and end-start <= max_time:
          #possible_states = todos os estados possiveis a partir do estado atual
          possible_states = generate_states(current_state)
          #valid_states = quantidade de estados possiveis
          valid_states = len(possible_states)

          #loop por todos os estados possiveis
          for state in possible_states:
              #aux_acc = acuracia do estado atual
              aux_acc = evaluate_state(state, classificadores,ordem_classes,x_test,y_train)
              
              #se acuracia atual for maior que a acuracia otima
              if aux_acc >= optimal_acc:
                  #atualiza a acuracia otima, o estado otimo, e o estado atual
                  optimal_acc = aux_acc
                  optimal_state = state
                  current_state = state
              else:
                  #senao, subtrai 1 do validstates (quando chega a 0 a heuristica para)
                  valid_states = valid_states - 1

          #end: tempo final
          end = time.process_time()

        #gen_state_comb = retorna os classificadores do estado otimo
        return gen_state_comb(optimal_state, classificadores)

def simulated_annealing(t,alfa,iter_max,max_time,classificadores,x_test,y_train,ordem_classes):
    current_state = generate_initial_state(classificadores)
    optimal_state = current_state
    current_acc = evaluate_state(current_state, classificadores,ordem_classes,x_test,y_train)
    optimal_acc = current_acc

    start = time.process_time() #start: tempo inicial
    end = 0                     #end:   tempo final
    
    
    while t >= 1 and end-start <= max_time and optimal_acc != 1:        
        
        for _ in range(iter_max):
            neighborhood = generate_neighborhood(current_state)
            if neighborhood == []:
                return gen_state_comb(optimal_state, classificadores)

            aux_state = rand_state(neighborhood) #escolhe um state aleatorio dentro dos vizinhos
            aux_acc = evaluate_state(aux_state, classificadores,ordem_classes,x_test,y_train) #calcula acuracia aux_state

            if aux_acc > current_acc:
                current_state = aux_state
                current_acc = aux_acc
                if aux_acc > optimal_acc:
                    optimal_state = aux_state
                    optimal_acc = aux_acc
            else:
                if change_probability(aux_acc,current_acc,t):
                    current_state = aux_state
                    current_acc = aux_acc
        
        t = t*alfa
        end = time.process_time()

    return gen_state_comb(optimal_state, classificadores)

def genetic(pop_size,max_iter,cross_ratio,mut_ratio,elite_pct,max_time,classificadores,x_test,y_train,ordem_classes):

    start = time.process_time()
    optimal_state = [0] * len(classificadores)
    optimal_acc = 0
    populacao = initial_population(pop_size, classificadores)
    conv = convergent(populacao)
    iter = 0    
    end = 0

    while not conv and iter < max_iter and end-start <= max_time and optimal_acc != 1:
        
        val_pop = evaluate_population(populacao, classificadores,ordem_classes,x_test,y_train)
        
        elitismo = elitism(val_pop, elite_pct)
        val_best = elitismo[0][0]

        new_pop = elitism_state(elitismo)
        best = new_pop[0]

        if (val_best > optimal_acc):
            optimal_state = best
            optimal_acc = val_best

        selected = selection(val_pop, pop_size - len(new_pop)) 
        crossed = crossover_step(selected, cross_ratio)
        mutated = mutation_step(crossed, mut_ratio)
        populacao = new_pop + mutated
        conv = convergent(populacao)
        iter+=1
        end = time.process_time()
        
    return gen_state_comb(optimal_state, classificadores)