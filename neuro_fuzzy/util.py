import csv
import random
from collections import Iterable

import numpy as np
from matplotlib import pyplot as plt


def normalizar(dado, minimo, maximo):
    if isinstance(dado, list) or isinstance(dado, tuple):
        dado = np.array(dado)
    return (dado - minimo) / (maximo - minimo)


def desnormalizar(dado_normalizado, minimo, maximo):
    if isinstance(dado_normalizado, list) or isinstance(dado_normalizado, tuple):
        dado_normalizado = np.array(dado_normalizado)
    return dado_normalizado * (maximo - minimo) + minimo


def erro_percentual_absoluto_medio(ydv, ys):
    somatorio_ep = 0
    total_itens = 0
    for k in range(len(ys)):
        # y = round(ydv[k], 5)
        y = ydv[k]
        if y != 0:
            # y_estimado = round(ys[k], 5)
            y_estimado = ys[k]
            ep = ((y - y_estimado) / y) * 100
            somatorio_ep += abs(ep)
            total_itens += 1
    epam = somatorio_ep / total_itens
    return epam


def embaralhar_dados(xt, ydt):
    temp = list(zip(xt, ydt))
    random.shuffle(temp)
    xt_embaralahdo, ydt_embaralhado = zip(*temp)
    return xt_embaralahdo, ydt_embaralhado


def exibir_resultado_desejado(xv, ydv, label_y_validacao):
    if len(xv) > 10000:
        largura = 16
    else:
        largura = 8
    plt.figure(figsize=(largura, 6))
    if(len(xv[0]) > 1):
        plt.plot(ydv, label=label_y_validacao)
        plt.xlabel("")
    else:
        plt.plot(xv, ydv, label=label_y_validacao)
        plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

def atrasar(dados, tamanho_atraso):
    dados_atrasados = []
    for i in range(len(dados)):
        if i < tamanho_atraso:
            dados_atrasados.append(0)
        else:
            dados_atrasados.append(dados[i - tamanho_atraso])
    return dados_atrasados

def avancar(dados, tamanho_avanco):
    n =  len(dados)
    dados_avancados = []
    for i in range(n):
        if i >= (n - tamanho_avanco):
            dados_avancados.append(0)
        else:
            dados_avancados.append(dados[i + tamanho_avanco])
    return dados_avancados

def ajustar(dados, tamanho_atraso, tamanho_avanco):
    n = len(dados)
    dados_ajustados = dados[tamanho_atraso:(n - tamanho_avanco)]
    return dados_ajustados

def criar_arquivo(nome_arquivo, dados):
    if isinstance(dados[0], Iterable) == False:
        dados = [[dados[i]] for i in range(len(dados))]
    with open(nome_arquivo, 'w', newline='\n') as arquivo:
        wr = csv.writer(arquivo)
        wr.writerows(dados)