# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 10:11:16 2021

@author: jolua
"""

def Estados(action):
    if action == 'start':
        resposta = input('Olá! Vamos identificar um estado do Brasil (S/N)?')
    elif action == 'restart':
        resposta = input('Deseja identificar outro Estado (S/N)?')
    
    if resposta == 'S':
        vegetacao = input('Qual a vegetação predominante do estado (caatinga, cerrado, pampa, pantanal, floresta amazônica, mata atlântica, mata subtropical)?')
        if vegetacao == 'pampa':
            print('O estado é o Rio Grande do Sul.')
        elif vegetacao == 'pantanal':
            print('O estado é o Mato Grosso do Sul.')
        elif vegetacao == 'mata subtropical':
            print('O estado é o Paraná.')
        elif vegetacao == 'floresta amazônica':
            clima = input('Qual o clima do estado (tropical, equatorial)?')
            oceano = input('O estado é banhado pelo oceano atlantico? (S/N)?')
            if clima == 'tropical':
                if oceano == 'S':
                    print('O estado é o Maranhão.')
                elif oceano == 'N':
                    print('O estado é o Mato Grosso.')
            elif clima == 'equatorial':
                habitantes_menos_1milhao = input('A população do estado tem menos de 1 milhão de habitantes (S/N)?')
                if habitantes_menos_1milhao == 'S':
                    if oceano == 'S':
                        print('O estado é o Amapá.')
                    elif oceano == 'N':
                        area_menos_200mil = input('A área do estado tem menos de 200 mil km²? (S/N)?')
                        if area_menos_200mil == 'S':
                            print('O estado é o Acre.')
                        elif area_menos_200mil == 'N':
                            print('O estado é o Roraíma.')
                elif habitantes_menos_1milhao == 'N':
                    if oceano == 'S':
                        print('O estado é o Pará.')
                    elif oceano == 'N':
                        area_mais_1milhao = input('A área do estado tem mais de 1 milhão km²? (S/N)?')
                        if area_mais_1milhao == 'S':
                            print('O estado é o Amazonas.')
                        elif area_mais_1milhao == 'N':
                            print('O estado é o Rondônia.')
        elif vegetacao == 'mata atlântica':
            clima = input('Qual o clima do estado (subtropical, tropical)?')
            habitantes_mais_10milhoes = input('A população do estado tem mais de 10 milhões de habitantes (S/N)?')
            if clima == 'tropical':
                if habitantes_mais_10milhoes == 'S':
                    print('O estado é o Rio de Janeiro.')
                elif habitantes_mais_10milhoes == 'N':
                    print('O estado é o Espírito Santo.')
            elif clima == 'subtropical':
                if habitantes_mais_10milhoes == 'S':
                    print('O estado é São Paulo.')
                elif habitantes_mais_10milhoes == 'N':
                    print('O estado é o Santa Catarina.')
        elif vegetacao == 'cerrado':
            clima = input('Qual o clima do estado (semiárido, tropical)?')
            if clima == 'semiárido':
                print('O estado é o Piauí.')
            elif clima == 'tropical':
                habitantes_mais_5milhoes = input('A população do estado tem mais de 5 milhões de habitantes (S/N)?')
                if habitantes_mais_5milhoes == 'S':
                    municipios_mais_500 = input('O estado tem mais de 500 municipios? (S/N)?')
                    if municipios_mais_500 == 'S':
                        print('O estado é Minas Gerais.')
                    elif municipios_mais_500 == 'N':
                        print('O estado é o Goiás.')
                elif habitantes_mais_5milhoes == 'N':
                    area_menos_10mil = input('A área do estado tem menos de 10 mil km²? (S/N)?')
                    if area_menos_10mil == 'S':
                        print('O estado é o Distrito Federal.')
                    elif area_menos_10mil == 'N':
                        print('O estado é o Tocantins.')
        elif vegetacao == 'caatinga':
            clima = input('Qual o clima do estado (semiárido, tropical)?')
            habitantes_mais_5milhoes = input('A população do estado tem mais de 5 milhões de habitantes (S/N)?')
            if clima == 'semiárido':
                if habitantes_mais_5milhoes == 'S':
                    area_mais_100mil = input('A área do estado tem mais de 100 mil km²? (S/N)?')
                    if area_mais_100mil == 'S':
                        print('O estado é o Ceará.')
                    elif area_mais_100mil == 'N':
                        print('O estado é o Pernambuco.')
                elif habitantes_mais_5milhoes == 'N':
                    area_mais_50mil = input('A área do estado tem mais de 50 mil km²? (S/N)?')
                    if area_mais_50mil == 'S':
                        municipios_mais_200 = input('O estado tem mais de 200 municipios? (S/N)?')
                        if municipios_mais_200 == 'S':
                            print('O estado é a Paraíba.')
                        elif municipios_mais_200 == 'N':
                            print('O estado é o Rio Grande do Norte.')
                    elif area_mais_50mil == 'N':
                        print('O estado é o Alagoas.')
            elif clima == 'tropical':    
                if habitantes_mais_5milhoes == 'S':
                    print('O estado é a Bahia.')
                elif habitantes_mais_5milhoes == 'N':
                    print('O estado é o Sergipe.')
    elif resposta == 'N':
        print('Tchau.')
        return 'N'
    
    return 'S'


rec = Estados('start')
while rec == 'S':
    rec = Estados('restart')