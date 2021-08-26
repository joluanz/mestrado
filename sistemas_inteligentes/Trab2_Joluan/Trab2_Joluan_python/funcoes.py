# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 10:07:44 2021

@author: jolua
"""
from misc import predicao
from sklearn.metrics import accuracy_score
import random
import math
import numpy as np

#em comum
def gen_state_comb(state, items):
    combinado_atual = []
    for i in range(len(state)):
        if(state[i] == 1):
          combinado_atual.append(items[i])
    return combinado_atual

def evaluate_state(state, classificadores,ordem_classes,x_test,y_train):
    #comb_states = classificadores que estão no estado
    comb_states = gen_state_comb(state, classificadores)

    #predicts: vai conter todos os predicts para cada classificador
    predicts = []

    #loop por cada classificador que está no estado
    for classificador in comb_states:
        predict = classificador.predict(x_test)
        #calcula o predict do clasificador atual e salva em predicts
        predicts.append(predict)

    #predito: predicao vai escolher o mais votado entre os classificadores
    predito = predicao(predicts,ordem_classes)

    #calcula a acuracia do predito com o real
    value = accuracy_score(y_train, predito)
    
    return value

def generate_initial_state(items):
    #generate_initial_state gera um estado inicial aleatorio do mesmo tamanho dos classificadores(items)
    zero_state = [0]*len(items) #cria um state todo de 0's para testar
    
    initial_state = [] #initial_state = cria o array que vai ser o estado inicial
    for i in range(len(items)): #for da quantidade de itens nos classificadores (9,15,21)
        initial_state.append(random.randint(0, 1)) #random de 0 ou 1, e adiciona no initial_state
    
    #o bloco abaixo é para não deixar o que o initial_state seja composto só de 0's
    while np.array_equal(zero_state,initial_state): #continua rodando até que seja difenrete do zero_state
        initial_state = []
        zero_state = [0]*len(items)
        for i in range(len(items)):
            initial_state.append(random.randint(0, 1))

    #retorna o initial_state com comprimento igual aos classificadores, composto de 0 e 1
    return initial_state

#hill climbing
def generate_states(initial_state):
    #generate_states gera todos os estados possiveis a partir de um estado inicial(sempre 'pra frente')
    states = [] #array para salvar os estados gerados

    #for para cada elemento no estado inicial
    for i in range(len(initial_state)):
        aux = initial_state.copy() #aux = copia do estado inicial
        aux[i] = 1 #salva na posição i do estado copiado 1
        if(not np.array_equal(aux,initial_state)): #só salvar se o estado aux for diferente do estado inicial
          states.append(aux)
    #retorna o array de estados gerados
    return states

#simulatedannealing
def change_state(state,position,value):
    #change_state, recebe um estado, uma posição, e um valor
    state[position] = value #e vai alterar o valor da posição do estado
    #retorna o estado alterado
    return state


def generate_neighborhood(state):
    #generate_neighborhood gera todos os estados vizinhos ao estado recebido
    neighborhood = [] #neighborhood= array para salvar os vizinhos
    zero_state = [0]*len(state) #cria um state todo de 0's para testar

    #dois for's, o primeiro vai passar inserindo 1 a cada posição do estado inicial
    #o segundo vai passar inserindo 0
    for i in range(len(state)):
        aux = state.copy()
        new_state = change_state(aux,i,1)
        if not np.array_equal(state,new_state): #caso o estado não seja igual ao inicial
            neighborhood.append(new_state) #salva o estado com a alteração no array de vizinhos
    for i in range(len(state)):
        aux = state.copy()
        new_state = change_state(aux,i,0)
        if not np.array_equal(state,new_state): #caso o estado não seja igual ao inicial
            if not np.array_equal(zero_state,new_state): # e caso, não seja [0,0,0] só de 0's
                neighborhood.append(new_state) #salva o estado com a alteração no array de vizinhos
    #retorna todos os vizinhos pelo array neighborhood
    return neighborhood

def rand_state(states):
    #rand_state = escolhe um estado aleatorio, a partir de um array de estados(states)
    index = random.randint(0,len(states)-1) #gera um index aleatorio
    #retorna o estado do index gerado do states
    return states[index]

def change_probability(value,best_value,t):
    p = 1/(math.exp(1)**((best_value-value)/t))
    r = random.uniform(0,1)
    if r < p:
        return True
    else:
        return False
    
#genetic
def initial_population(n, items):
    #initial_population, gera uma população de estados de tamanho 'n' (n=hiperparametro do tamanho da população)
    pop = [] #pop = inicia o array que terá varios estados
    count = 0 #contador que irá até 'n'
    while count < n:
        individual = generate_initial_state(items) #individual = individuo gerado a partir do generate_initial state, que gera um estaod inicial aleatorio
        pop = pop + [individual] #insere o individual na populção
        count += 1 #contador++

    #retorna a população composto de 'n' individuos
    return pop

def convergent(population):
    #convergent verifica se a população convergiu para um estado(todos os individuos serem iguais)
    if population != []:
        base = population[0] #base= pega o primeiro inviduo para servir de comparação
        i = 0
        while i < len(population):
            if base != population[i]: #compara com o primeiro individuo, caso diferente, a população nao ocnvergiu
                return False #retorna false para população com individuos diferentes
            i += 1
        return True #retorna true para população com todos os individuos iguais

def evaluate_population (populacao, classificadores,ordem_classes,x_test,y_train):
    #evaluate_population calcula a acuracia de cada estado e salva em um array com
    #[(acuracia,estado),(acuracia,estado),...,(acuracia,estado)]
    eval = []
    for state in populacao:
        eval = eval + [(evaluate_state(state, classificadores,ordem_classes,x_test,y_train), state)]
    return eval

def first(x):
    return x[0]

def elitism (val_pop, pct):
    n = math.floor((pct/100)*len(val_pop))
    if n < 1:
        n = 1
    val_elite = sorted (val_pop, key = first, reverse = True)[:n]
    #elite = [s for v,s in val_elite]
    return val_elite

def elitism_state(val_elite):
    elite = [s for v,s in val_elite]
    return elite

def states_total_value(states):
    total_sum = 0
    for state in states:
        total_sum = total_sum + state[0]
    return total_sum

def roulette_construction(states):
    aux_states = []
    roulette = []
    total_value = states_total_value(states)
    for state in states:
        value = state[0]
        if total_value != 0:
            ratio = value/total_value
        else:
            ratio = 1
        aux_states.append((ratio,state[1]))
 
    acc_value = 0
    for state in aux_states:
        acc_value = acc_value + state[0]
        s = (acc_value,state[1])
        roulette.append(s)
    return roulette

def roulette_run (rounds, roulette):
    if roulette == []:
        return []
    selected = []
    while len(selected) < rounds:
        r = random.uniform(0,1)
        for state in roulette:
            if r <= state[0]:
                selected.append(state[1])
                break
    return selected

def selection(value_population,n):
    aux_population = roulette_construction(value_population)
    new_population = roulette_run(n, aux_population)
    return new_population

def crossover(dad,mom):
    zero_state = [0]*len(dad)
    r = random.randint(1, len(dad) - 1)
    son = dad[:r]+mom[r:]
    daug = mom[:r]+dad[r:]

    while np.array_equal(zero_state,son) or np.array_equal(zero_state,daug):
        r = random.randint(1, len(dad) - 1)
        son = dad[:r]+mom[r:]
        daug = mom[:r]+dad[r:]

    return son, daug

def crossover_step (population, crossover_ratio):
    new_pop = []

    if len(population) < 2:
        return population
    else:
        for _ in range (round(len(population)/2)):
            rand = random.uniform(0, 1)
            fst_ind = random.randint(0, len(population) - 1)
            scd_ind = fst_ind
            while fst_ind == scd_ind:
                scd_ind = random.randint(0, len(population) - 1)
            parent1 = population[fst_ind]
            parent2 = population[scd_ind]

            if rand <= crossover_ratio:
                offspring1, offspring2 = crossover(parent1, parent2)            
            else:
                offspring1, offspring2 = parent1, parent2

            new_pop = new_pop + [offspring1, offspring2]
        
    return new_pop

def mutation (indiv):
    zero_state = [0]*len(indiv)

    individual = indiv.copy()
    rand = random.randint(0, len(individual) - 1)
    if individual[rand] == 0:
        individual[rand] = 1
    else:
        individual[rand] = 0
    
    while np.array_equal(zero_state,individual):
        individual = indiv.copy()
        rand = random.randint(0, len(individual) - 1)
        if individual[rand] == 0:
            individual[rand] = 1
        else:
            individual[rand] = 0

    return individual

def mutation_step (population, mutation_ratio):
    ind = 0
    for individual in population:
        rand = random.uniform(0, 1)

        if rand <= mutation_ratio:
            mutated = mutation(individual)
            population[ind] = mutated
                
        ind+=1
        
    return population
