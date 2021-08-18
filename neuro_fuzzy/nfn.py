import random
import simpful as sf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from neuro_fuzzy.fis import FIS
from neuro_fuzzy.util import desnormalizar


class NFN(FIS):

    def __init__(self, n_epoca, xt, ydt, xv, ydv, m, label_y_validacao, alfa = 0.1):
        super().__init__(n_epoca, xt, ydt, xv, ydv, m, label_y_validacao, alfa)
        self.x_max = [max(idx) for idx in zip(*self.xt)]
        self.x_min = [min(idx) for idx in zip(*self.xt)]
        self.f = self.gerar_funcoes_pertinencia()
        self.w = self.gerar_w()

    def reiniciar_parametros_aleatorios(self):
        self.w = self.gerar_w()

    def gerar_funcoes_pertinencia(self):
        f = []
        nomes_funcoes = []
        for j in range(self.m):
            nomes_funcoes.append(str(j))
        for i in range(self.n):
            universo = [self.x_min[i], self.x_max[i]]
            antecedente = sf.AutoTriangle(self.m, terms=nomes_funcoes, universe_of_discourse=universo)
            f.append(antecedente)
        return f

    def gerar_w(self):
        w = []
        for i in range(self.n):
            wi = []
            delta = self.x_max[i] - self.x_min[i]
            for j in range(self.m):
                wij = self.x_min[i] + (random.random() * delta)
                wi.append(wij)
            w.append(wi)
        return w

    def calcular_ys(self, x_ponto_validacao_treinamento):
        ys = 0
        for i in range(self.n):
            ys += self.calcular_yi(x_ponto_validacao_treinamento[i], self.f[i], self.w[i])
        return ys

    def calcular_yi(self, xi, fi, wi):
        k = self.obter_indice_funcao_ativada(xi, fi)
        if k is None:
            x_min = fi._universe_of_discourse[0]
            if xi < x_min:
                return wi[0]
            else:
                return wi[len(fi._FSlist) - 1]
        else:
            yi = fi._FSlist[k].get_value(xi) * wi[k]
            if fi._FSlist[k].get_value(xi) < 1:
                yi += fi._FSlist[k+1].get_value(xi) * wi[k+1]
            return yi

    def obter_indice_funcao_ativada(self, xi, fi):
        x_min = fi._universe_of_discourse[0]
        x_max = fi._universe_of_discourse[1]
        if xi < x_min:
            return None
        elif xi > x_max:
            return None
        else:
            delta_i = (x_max - x_min) / (self.m - 1)
            k = int((xi - x_min) / delta_i)
            return k

    def obter_ys_validacao(self):
        ys = []
        for k in range(len(self.xv)):
            x_ponto_validacao = self.xv[k]
            y = self.calcular_ys(x_ponto_validacao)
            ys.append(y)
        return ys

    def obter_ys_treino(self):
        ys = []
        for k in range(len(self.xt)):
            x_ponto = self.xt[k]
            y = self.calcular_ys(x_ponto)
            ys.append(y)
        return ys

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
            for p in range(self.n_pontos_treinamento):
                x_ponto_treino = self.xt[p]
                ys = self.calcular_ys(x_ponto_treino)
                yd = self.ydt[p]
                for i in range(self.n):
                    xi = x_ponto_treino[i]
                    fi = self.f[i]
                    k = self.obter_indice_funcao_ativada(xi, fi)
                    if k is None:
                        x_min = fi._universe_of_discourse[0]
                        if xi < x_min:
                            k = 0
                            self.w[i][k] = self.w[i][k] - self.alfa * (ys - yd) * fi._FSlist[k].get_value(xi)
                        else:
                            k = len(fi._FSlist) - 1
                            self.w[i][k] = self.w[i][k] - self.alfa * (ys - yd) * fi._FSlist[k].get_value(xi)
                    else:
                        self.w[i][k] = self.w[i][k] - self.alfa * (ys - yd) * fi._FSlist[k].get_value(xi)
                        if fi._FSlist[k].get_value(xi) < 1:
                            self.w[i][k+1] = self.w[i][k+1] - self.alfa * (ys - yd) * fi._FSlist[k+1].get_value(xi)
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
                if (len(self.xv[0]) > 1):
                    plt.plot(ys)
                else:
                    plt.plot(self.xv, ys)
                if epoca == (self.n_epoca - 1):
                    plt.legend()
                    plt.show()
        if plota_resultado_epocas == True:
            self.exibir_erro_epoca(rmse_validacao_epocas, rmse_treino_epocas)
        self.treinado = True