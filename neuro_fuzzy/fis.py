from decimal import Decimal

import numpy as np
import matplotlib.pyplot as plt
from neuro_fuzzy.util import desnormalizar, erro_percentual_absoluto_medio, embaralhar_dados, normalizar
from sklearn.metrics import mean_squared_error

class FIS:

    def __init__(self, n_epoca, xt, ydt, xv, ydv, m, label_y_validacao, alfa=0.01, normalizar_dados = False):
        self.n_epoca = n_epoca
        self.n_pontos_treinamento = len(xt)
        self.n = len(xt[0])
        if normalizar_dados:
            self.xt_original = xt
            self.ydt_original = ydt
            self.xv_original = xv
            self.ydv_original = ydv
            self.ydt, self.ydv, self.y_minimo_original, self.y_maximo_original = self.nomalizar_y(self.ydt_original, self.ydv_original)
            self.xt, self.xv, self.x_minimo_original, self.y_maximo_original = self.normalizar_x(self.xt_original, self.xv_original)
        else:
            self.xt = np.array(xt)
            self.ydt = np.array(ydt)
            self.xv = np.array(xv)
            self.ydv = np.array(ydv)
        self.m = m
        self.alfa = alfa
        self.label_y_validacao = label_y_validacao
        self.treinado = False

    def exibir_resultado_validacao(self, desnormalizar_resultado =  False, minimo_desnomalizacao = 0, maximo_desnomalizacao = 0):
        ys = self.obter_ys_validacao()
        if desnormalizar_resultado:
            ys = desnormalizar(ys, minimo_desnomalizacao, maximo_desnomalizacao)
            ydv = desnormalizar(self.ydv, minimo_desnomalizacao, maximo_desnomalizacao)
        else:
            ydv = self.ydv
        if len(self.xv) > 10000:
            largura = 16
        else:
            largura = 8
        plt.figure(figsize=(largura, 6))
        if(len(self.xv[0]) > 1):
            plt.plot(ydv, label=self.label_y_validacao)
            plt.plot(ys, label='y estimado')
            plt.xlabel("")
        else:
            plt.plot(self.xv, ydv, label=self.label_y_validacao)
            plt.plot(self.xv, ys, label='y estimado')
            plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()

        rmse = mean_squared_error(ydv, ys, squared = False)
        rmse = round(rmse, 4)
        print('Root mean squared error (RMSE): ', rmse)

        epam = erro_percentual_absoluto_medio(ydv, ys)
        epam = round(epam, 4)
        print('Mean absolute percentage error (MAPE): {:.4f}%'.format(epam))

    def exibir_resultado_validacao_multiplos_treinos(self, numero_treinos = 10, desnormalizar_resultado =  False, minimo_desnomalizacao = 0, maximo_desnomalizacao = 0):
        rmse_treinos = []
        mape_treinos = []
        if self.treinado:
            rmse, mape = self.resultado_validacao(desnormalizar_resultado, minimo_desnomalizacao, maximo_desnomalizacao)
            rmse_treinos.append(rmse)
            mape_treinos.append(mape)
            numero_treinos -= 1
        for i in range(numero_treinos):
            self.reiniciar_parametros_aleatorios()
            self.treinar_gradiente(plota_resultado_epocas=False)
            rmse, mape = self.resultado_validacao(desnormalizar_resultado, minimo_desnomalizacao, maximo_desnomalizacao)
            rmse_treinos.append(rmse)
            mape_treinos.append(mape)

        treinos = [i + 1 for i in range(len(rmse_treinos))]

        rmse_treinos = [round(rmse_treinos[i], 4) for i in range(len(rmse_treinos))]
        rmse_medio = [round(np.mean(rmse_treinos), 4)] * len(rmse_treinos)
        rmse_desvio_padrao = round(np.std(rmse_treinos), 4)

        plt.figure(figsize=(8, 6))
        plt.plot(treinos, rmse_treinos, marker='o', label='RMSE do treino')
        plt.plot(treinos, rmse_medio, linestyle='--', label='RMSE médio')
        plt.xticks(treinos)
        plt.xlabel("Treino")
        plt.ylabel("Root mean squared error (RMSE)")
        plt.legend()
        plt.show()
        print('Média do RMSE: ', rmse_medio[0])
        print('Desvio padrão do RMSE: ', rmse_desvio_padrao)

        mape_treinos = [round(mape_treinos[i], 4) for i in range(len(mape_treinos))]
        mape_medio = [round(np.mean(mape_treinos), 4)] * len(mape_treinos)
        mape_desvio_padrao = round(np.std(mape_treinos), 4)

        plt.figure(figsize=(8, 6))
        plt.plot(treinos, mape_treinos, marker='o', label='MAPE do treino')
        plt.plot(treinos, mape_medio, linestyle='--', label='MAPE médio')
        plt.xticks(treinos)
        plt.xlabel("Treino")
        plt.ylabel("Mean absolute percentage error (MAPE)")
        plt.legend()
        plt.show()
        print('Média do MAPE: {:.4f}%'.format(mape_medio[0]))
        print('Desvio padrão do MAPE: {:.4f}%'.format(mape_desvio_padrao))

    def resultado_validacao(self, desnormalizar_resultado =  False, minimo_desnomalizacao = 0, maximo_desnomalizacao = 0):
        ys = self.obter_ys_validacao()
        if desnormalizar_resultado:
            ys = desnormalizar(ys, minimo_desnomalizacao, maximo_desnomalizacao)
            ydv = desnormalizar(self.ydv, minimo_desnomalizacao, maximo_desnomalizacao)
        else:
            ydv = self.ydv
        rmse = mean_squared_error(ydv, ys, squared=False)
        epam = erro_percentual_absoluto_medio(ydv, ys)
        return rmse, epam

    def exibir_erro_epoca(self, rmse_validacao_epoca, rmse_treino_epoca):
        epocas = [i+1 for i in range(self.n_epoca)]
        rmse_validacao_epoca = [round(rmse_validacao_epoca[i], 4) for i in range(self.n_epoca)]
        rmse_treino_epoca = [round(rmse_treino_epoca[i], 4) for i in range(self.n_epoca)]
        plt.figure(figsize=(8, 6))
        plt.plot(epocas, rmse_validacao_epoca, marker='o', label='RMSE dos dados de validação')
        plt.plot(epocas, rmse_treino_epoca, marker='x', linestyle=':', label='RMSE dos dados de treino')
        plt.xticks(epocas)
        plt.xlabel("Época de treino")
        plt.ylabel("Root mean squared error (RMSE)")
        plt.legend()
        plt.show()

    def nomalizar_y(self, ydt_original, ydv_original):
        minimo = min(min(ydt_original), min(ydv_original))
        maximo = max(max(ydt_original), max(ydv_original))
        ydt_normalizado = normalizar(ydt_original, minimo, maximo)
        ydv_normalizado = normalizar(ydv_original, minimo, maximo)
        return  ydt_normalizado, ydv_normalizado, minimo, maximo

    def normalizar_x(self, xt_original, xv_original):
        x = np.concatenate((xt_original, xv_original))
        x_transposto = x.T
        x_min = [x_transposto[i].min() for i in range(len(x_transposto))]
        x_max = [x_transposto[i].max() for i in range(len(x_transposto))]
        xt_transposto = np.array(xt_original).T
        xv_transposto = np.array(xv_original).T
        for i in range(self.n):
            xt_transposto[i] = normalizar(xt_transposto[i], x_min[i], x_max[i])
            xv_transposto[i] = normalizar(xv_transposto[i], x_min[i], x_max[i])
        return xt_transposto.T, xv_transposto.T, x_min, x_max