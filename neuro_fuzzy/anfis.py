from math import e
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from sklearn.metrics import mean_squared_error
from neuro_fuzzy.fis import FIS
from neuro_fuzzy.util import desnormalizar


class ANFIS(FIS):

    def __init__(self, n_epoca, xt, ydt, xv, ydv, m, label_y_validacao, alfa=0.01):
        super().__init__(n_epoca, xt, ydt, xv, ydv, m, label_y_validacao, alfa)
        self.c, self.s, self.p, self.q = self.gerar_parametros_aleatorios()
        #np.seterr(over='raise', under='raise')

    def reiniciar_parametros_aleatorios(self):
        self.c, self.s, self.p, self.q = self.gerar_parametros_aleatorios()

    def gerar_parametros_aleatorios(self):
        c = []
        s = []
        p = []
        q = []

        x_max = max(self.xt.tolist())
        x_min = min(self.xt.tolist())

        for i in range(self.n):
            media_entrada_regra = []
            desvio_padrao_entrada_regra = []
            p_entrada_regra = []
            for j in range(self.m):
                media = x_min[i] + ((x_max[i] - x_min[i]) * random.random())
                media_entrada_regra.append(media)
                desvio_padrao = (x_max[i] - x_min[i]) * random.random()
                desvio_padrao_entrada_regra.append(desvio_padrao)
                parametro_p = random.random()
                p_entrada_regra.append(parametro_p)
            c.append(media_entrada_regra)
            s.append(desvio_padrao_entrada_regra)
            p.append(p_entrada_regra)

        for j in range(self.m):
            q.append(random.random())

        return c, s, p, q

    def calcular_yj(self, x_ponto_validacao_treinamento):
        yj = []
        for j in range(self.m):
            y = self.q[j]
            for i in range(self.n):
                try:
                    y += self.p[i][j] * x_ponto_validacao_treinamento[i]
                except FloatingPointError:
                    y_temp = round(self.p[i][j], 4) * round(x_ponto_validacao_treinamento[i], 4)
                    y = round((y + y_temp), 4)
            yj.append(y)
        return yj

    def calcular_a(self, yj, wj):
        a = 0
        for j in range(self.m):
            a += (wj[j] * yj[j])
        return a

    def calcular_b(self, wj):
        return np.sum(wj)

    def calcular_pertinencia(self, x, media, desvio_padrao):
        try:
            pertinencia = e ** (-1 / 2 * ((x - media) / desvio_padrao) ** 2)
        except FloatingPointError:
            pertinencia = 0
        return pertinencia

    def calcular_wj(self, x_ponto_validacao_treinamento):
        wj = []
        for j in range(self.m):
            w = 1
            for i in range(self.n):
                pertinencia = self.calcular_pertinencia(x_ponto_validacao_treinamento[i], self.c[i][j], self.s[i][j])
                w *= pertinencia
            wj.append(w)
        return wj

    def calcular_ys(self, x_ponto_validacao_treinamento):
        wj = self.calcular_wj(x_ponto_validacao_treinamento)
        yj = self.calcular_yj(x_ponto_validacao_treinamento)
        a = self.calcular_a(yj, wj)
        b = self.calcular_b(wj)
        ys = a / b
        return ys, wj, b, yj

    def obter_ys_validacao(self):
        ys = []
        for k in range(len(self.xv)):
            x_ponto_validacao = self.xv[k]
            y, _, _, _ = self.calcular_ys(x_ponto_validacao)
            ys.append(y)
        return ys
    
    def obter_ys_treino(self):
        ys = []
        for k in range(len(self.xt)):
            x_ponto = self.xt[k]
            y, _, _, _ = self.calcular_ys(x_ponto)
            ys.append(y)
        return ys

    def calcular_dedys(self, ys, yd):
        return ys - yd

    def calcular_dysdyj(self, w, b):
        return w / b

    def calcular_dyjdqj(self):
        return 1

    def calcular_dedqj(self, w, ys, yd, b):
        try:
            dedqj = self.calcular_dedys(ys, yd) * self.calcular_dysdyj(w, b) * self.calcular_dyjdqj()
        except FloatingPointError:
            dedqj = 0
        return dedqj

    def calcualr_dysdwj(self, y, ys, b):
        return (y - ys) / b

    def calcular_dwjdcij(self, w, x, c, s):
        dwjdcij = w * ((x - c) / s ** 2)
        return dwjdcij

    def calcular_dedcij(self, ys, yd, y, b, w, x, c, s):
        try:
            dedcij = self.calcular_dedys(ys, yd) * self.calcualr_dysdwj(y, ys, b) * self.calcular_dwjdcij(w, x, c, s)
        except FloatingPointError:
            dedcij = 0
        return dedcij

    def calcular_dwjdsij(self, w, x, c, s):
        dwjdsij = w * ((x - c) ** 2) / (s ** 3)
        return dwjdsij

    def calcular_dedsij(self, ys, yd, y, b, w, x, c, s):
        try:
            dedsij = self.calcular_dedys(ys, yd) * self.calcualr_dysdwj(y, ys, b) * self.calcular_dwjdsij(w, x, c, s)
        except FloatingPointError:
            dedsij = 0
        return dedsij

    def calcular_dyjdpij(self, x):
        return x

    def calcular_dedpij(self, ys, yd, w, b, x):
        try:
            dedpij = self.calcular_dedys(ys, yd) * self.calcular_dysdyj(w, b) * self.calcular_dyjdpij(x)
        except FloatingPointError:
            dedpij = 0
        return dedpij

    def treinar_gradiente(self, plota_resultado_epocas=False, desnormalizar_apresentacao_resultado=False,
                          minimo_desnomalizacao=0, maximo_desnomalizacao=0):
        if plota_resultado_epocas:
            rmse_validacao_epocas = []
            rmse_treino_epocas = []
            if desnormalizar_apresentacao_resultado:
                ydv = desnormalizar(self.ydv, minimo_desnomalizacao, maximo_desnomalizacao)
                ydt = desnormalizar(self.ydt, minimo_desnomalizacao, maximo_desnomalizacao)
            else:
                ydv = self.ydv
                ydt = self.ydt
        for epoca in range(self.n_epoca):
            for k in range(self.n_pontos_treinamento):
                x_ponto_treino = self.xt[k]
                ys, wj, b, yj = self.calcular_ys(x_ponto_treino)
                yd = self.ydt[k]
                for j in range(self.m):
                    for i in range(self.n):
                        dedcij = self.calcular_dedcij(ys, yd, yj[j], b, wj[j], x_ponto_treino[i], self.c[i][j], self.s[i][j])
                        dedsij = self.calcular_dedsij(ys, yd, yj[j], b, wj[j], x_ponto_treino[i], self.c[i][j], self.s[i][j])
                        dedpij = self.calcular_dedpij(ys, yd, wj[j], b, x_ponto_treino[i])
                        self.c[i][j] = self.c[i][j] - self.alfa * dedcij
                        self.s[i][j] = self.s[i][j] - self.alfa * dedsij
                        self.p[i][j] = self.p[i][j] - self.alfa * dedpij
                    self.q[j] = self.q[j] - self.alfa * self.calcular_dedqj(wj[j], ys, yd, b)
            if plota_resultado_epocas == True:
                ys = self.obter_ys_validacao()
                yst = self.obter_ys_treino()
                if desnormalizar_apresentacao_resultado:
                    ys = desnormalizar(ys, minimo_desnomalizacao, maximo_desnomalizacao)
                    yst = desnormalizar(yst, minimo_desnomalizacao, maximo_desnomalizacao)
                rmse_validacao = mean_squared_error(ydv, ys, squared=False)
                rmse_validacao_epocas.append(rmse_validacao)
                rmse_treino = mean_squared_error(ydt, yst, squared=False)
                rmse_treino_epocas.append(rmse_treino)
                if epoca == 0:
                    if len(self.xv) > 10000:
                        largura = 16
                    else:
                        largura = 8
                    plt.figure(figsize=(largura, 6))
                    if (len(self.xv[0]) > 1):
                        plt.plot(ydv, label=self.label_y_validacao)
                        plt.xlabel("")
                    else:
                        plt.plot(self.xv, ydv, label=self.label_y_validacao)
                        plt.xlabel("x")
                    plt.ylabel("y")
                if (self.n > 1):
                    plt.plot(ys)
                else:
                    plt.plot(self.xv, ys)
                if epoca == (self.n_epoca - 1):
                    plt.legend()
                    plt.show()
        if plota_resultado_epocas == True:
            self.exibir_erro_epoca(rmse_validacao_epocas, rmse_treino_epocas)
        self.treinado = True