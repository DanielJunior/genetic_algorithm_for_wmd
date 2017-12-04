# -*- coding: utf-8 -*-
import numpy as np, pickle, multiprocessing as mp
import sys
from random import randrange
import random
from operator import itemgetter
import time

####CONSTANTES####

K = 20
MAX_ITER = 50
MAX_GERACAO = 10
CHANCE = 0.5
##################

#####LEIA DADOS######
load_file = "files/book100_vec.pk"

print("Carregando dados...")
with open(load_file) as f:
    # X é o doc em word2vec, Bow é o bag of words, y é a classe codificada para inteiro,
    # C é o nome da classe, words são as palavras relevantes do doc
    [X, BOW_X, y, C, words] = pickle.load(f)

n = np.shape(X)
N = n[0]
print("Parseando dados...")
for i in xrange(N):
    X_i = X[i].T
    X_i = X_i.tolist()
    X[i] = X_i
D = np.zeros((N, K))
print("Calculando centróides...")
#####CALCULA CENTROIDES######
CENTROIDES = list(np.mean(doc, axis=0) for doc in X)


############################        GENETICO      ############################

###source e target são indices em X##########
def gerador_populacao_semi_guloso(source, target):
    populacao = []
    for i in xrange(MAX_ITER):
        T = np.zeros((len(X[source]), len(X[target])))
        words_source = T.shape[0]  #### quantidade de palavras
        words_target = T.shape[1]
        BOW_1_copy = np.copy(BOW_X[source])
        BOW_2_copy = np.copy(BOW_X[target])
        ##### a cada iteração do meu gerador esse sample vai selecionar linhas diferentes,
        # evitando ficar uma parte de T toda vazia...
        for i in random.sample(range(1, words_source), words_source - 1):
            ###calculando as distancias entre a palavra de i de source até j de target
            distancias = list(distance(X[source][i], X[target][j]) for j in xrange(words_target))
            #### tratando o caso de um documento ter menos que k palavras
            k_ = K
            if k_ > words_target:
                k_ = words_target
            #### indices das palavras que gasto menos pra 'viajar' de source pra target
            k_closers_words = np.argpartition(distancias, k_ - 1)
            k_closers_words = k_closers_words[1:len(k_closers_words)]
            for j in k_closers_words:
                #### se posso viajar
                if (BOW_1_copy[i] - BOW_2_copy[j] >= 0):
                    travel = BOW_2_copy[j]
                    BOW_1_copy[i] -= travel
                    BOW_2_copy[j] -= travel
                    T[i, j] = travel
        populacao.append(T)
    ####Preciso ordenar a populacao pelo T_sum_cost e os scores
    scores = []
    for t in populacao:
        scores.append(T_sum_cost(t, X[source], X[target]))
    idx_ordenados = np.argpartition(scores, len(scores) - 1)
    scores = np.array(scores)
    populacao = np.array(populacao)
    return populacao[idx_ordenados], scores[idx_ordenados]


def ok_out(T, source):
    # todas as palavras estão viajando respeitando sua cota de viagem?
    return any([T.sum(axis=1)[i] <= BOW_X[source][i] for i in xrange(T.shape[0])])


def ok_in(T, target):
    # todas as palavras recebendo viagem estão respeitando sua cota?
    return any([T.sum(axis=0)[i] <= BOW_X[target][i] for i in xrange(T.shape[1])])


# estratégia elitista
def crossover(membro1, membro2, source, target):
    # faço o crossover copiando a primeira linha e cruzando a ultima linha de cada individuo
    child1 = np.zeros(membro1.shape)
    child2 = np.zeros(membro2.shape)
    rows = membro1.shape[0]
    for i in xrange(rows):

        ch1 = np.copy(child1)
        ch2 = np.copy(child2)

        # devo checar se as operações vão me produzir uma solução válida
        ch1[i] = membro1[i]
        if (ok_out(ch1, source) and ok_in(ch1, target)):
            child1[i] = membro1[i]

        ch2[i] = membro2[i]
        if (ok_out(ch2, source) and ok_in(ch2, target)):
            child2[i] = membro2[i]

        ch1[rows - 1 - i] = membro2[rows - 1 - i]
        if (ok_out(ch1, source) and ok_in(ch1, target)):
            child1[i] = membro2[rows - 1 - i]

        ch2[rows - 1 - i] = membro1[rows - 1 - i]
        if (ok_in and ok_out):
            child2[i] = membro1[rows - 1 - i]

    return child1, child2


# estratégia elitista
def selecao_natural(filhos, populacao):
    populacao[8] = filhos[0]
    populacao[9] = filhos[1]
    return populacao


def mutacao(populacao):
    for i, member in enumerate(populacao):
        if random.random() < CHANCE:
            continue
        # seleciono duas linhas aleatórias
        row1 = random.randint(0, member.shape[0] - 1)
        col1 = random.randint(0, member.shape[1] - 1)
        col2 = random.randint(0, member.shape[1] - 1)
        row2 = random.randint(0, member.shape[0] - 1)
        # se a célula escolhida é > 0 posso trocar unidades entre colunas de linhas diferentes
        #tenho que garantir que não vou criar linha com número negativo!
        if member[row1][col1] > 0:
            member[row1][col1] -= 1
            member[row1][col2] += 1
            member[row2][col1] += 1
            member[row2][col2] -= 1
        elif member[row2][col2] > 0:
            member[row1][col1] += 1
            col2 = random.randint(0, member.shape[1] - 1)
            member[row1][col2] -= 1
            row2 = random.randint(0, member.shape[0] - 1)
            member[row2][col1] -= 1
            member[row2][col2] += 1
    return populacao


def avalia(populacao, source, target):
    ####Preciso ordenar a populacao pelo T_sum_cost e os scores
    scores = []
    for t in populacao:
        scores.append(T_sum_cost(t, X[source], X[target]))
    idx_ordenados = np.argpartition(scores, len(scores) - 1)
    scores = np.array(scores)
    populacao = np.array(populacao)
    return populacao[idx_ordenados], scores[idx_ordenados]


def criterio_parada(geracao, ultima_atualizacao):
    if geracao > MAX_GERACAO or ultima_atualizacao > MAX_GERACAO / 3:
        return True
    return False


###source e target são indices em X##########
def genetico_atualizado(source, target):
    geracao = 0
    ####retorno a populacao inicial em ordem decrescente de scores e os scores junto#######
    populacao, scores = gerador_populacao_semi_guloso(source, target)  #### populacao é uma lista de T's
    best_score = scores[0]
    best = populacao[0]
    ultima_atualizacao = 1
    while (not criterio_parada(geracao, ultima_atualizacao)):
        filhos = crossover(populacao[0], populacao[1], source, target)
        nova_populacao = selecao_natural(filhos, populacao)
        nova_populacao = mutacao(populacao)
        ####minha avaliação retorna o score e ordena internamente a nova_populacao######
        nova_populacao, novo_score = avalia(nova_populacao, source, target)  # minha avaliacao é o T_sum_cost ordenado
        novo_best_score = novo_score[0]
        ####MEU PROBLEMA É DE MINIMIZAÇÃO########
        if novo_best_score < best_score:
            best_score = novo_best_score
            best = nova_populacao[0]
            ultima_atualizacao = 1
        else:
            ####se não encontrei uma soluação melhor, levo o meu atual melhor pra proxima geracao#######
            nova_populacao[-1] = best
            ultima_atualizacao += 1
        populacao = nova_populacao
    return best


#################### FIM GENETICO #####################################

def distance(x1, x2):
    return np.sqrt(np.sum((np.array(x1) - np.array(x2)) ** 2))


def T_sum_cost(T, DOC1, DOC2):
    sum = 0
    for i, costs in enumerate(T):
        for j, cost in enumerate(costs):
            sum += cost * distance(DOC1[i], DOC2[j])
    return sum


def calcula_wmd(ix):
    print("Calculando WMD do %dº documento" % ix)
    source = X[ix]
    source_centroide = CENTROIDES[ix]
    # print("Calculando distancias...")
    ####CALCULA AS DISTANCIAS DOS OUTROS CENTROIDES PARA ESSE#####
    distancias = list(np.linalg.norm(source_centroide - target) for target in CENTROIDES)
    # print("Selecionando K candidatos...")
    #########RECUPERA O INDICES DOS K DOCS MAIS PRÓXIMOS###########
    candidatos = np.argpartition(distancias, K + 1)[0:K + 1]
    ###### O MAIS PRÓXIMO É ELE MESMO, ENTÃO REMOVO ELE ###################
    candidatos = candidatos[1:len(candidatos)]
    Di = np.zeros((1, K))
    #######geração do vetor de custos associado ao doc origem#########
    for j, candidato in enumerate(candidatos):
        #####ix é o índice do doc origem e candidato é o indice do doc target########
        T = genetico_atualizado(ix, candidato)
        #### calculo o valor associado a matriz gerada pelo genetico########
        cost = T_sum_cost(T, X[ix], X[candidato])
        Di[0, j] = cost
    ####retorno tupla com o wmd desse doc e os candidatos usados#########
    return Di, candidatos


def main():
    pool = mp.Pool(processes=8)

    pool_outputs = pool.map(calcula_wmd, list(range(N)))
    pool.close()
    pool.join()

    WMD_D = np.zeros((N, K))
    CANDIDATOS_WMD = np.zeros((N, K))

    start_time = time.time()
    for i in xrange(N):
        WMD_D[i], CANDIDATOS_WMD[i] = pool_outputs[i]
        # WMD_D[i], CANDIDATOS_WMD[i] = calcula_wmd(i)
        print(WMD_D[i])
    print("--- %s seconds ---" % (time.time() - start_time))

    #######SALVANDO RESULTADOS PARA USAR NO RECOMENDADOR##############
    print("Salvando resultados...")
    with open('wmd.pk', 'w') as f:
        pickle.dump(WMD_D, f)

    with open('candidatos.pk', 'w') as f:
        pickle.dump(CANDIDATOS_WMD, f)


if __name__ == "__main__":
    main()
    print("Fim!")
