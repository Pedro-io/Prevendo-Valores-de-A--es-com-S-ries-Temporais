#%%
# Importa as bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Define o caminho do arquivo CSV
caminho_arquivo = "C:\\Users\\maype\\Desktop\\projetos\\Prevendo Valores de Ações com Séries Temporais\\Data\\Eletrobras ELET3 - Histórico  InfoMoney.csv"

# Lê o arquivo CSV em um DataFrame pandas
df = pd.read_csv(caminho_arquivo, sep=",", index_col="DATA", parse_dates=True)

# Mostra as primeiras linhas do DataFrame
print(df.head())

# Visualiza a série temporal do preço de fechamento
plt.figure(figsize=(12, 6))
plt.plot(df['FECHAMENTO'])
plt.title('Preço de Fechamento da Ação ELET3')
plt.xlabel('Data')
plt.ylabel('Preço (R$)')
plt.grid(True)
plt.show()

#%%
# Decomposição da série temporal em tendência, sazonalidade e ruído
decomposicao = seasonal_decompose(df['FECHAMENTO'], model='additive')

# Plota os componentes da decomposição
plt.figure(figsize=(12, 8))

# Tendência
plt.subplot(4, 1, 1)
plt.plot(decomposicao.trend)
plt.title('Tendência')
plt.grid(True)

# Sazonalidade
plt.subplot(4, 1, 2)
plt.plot(decomposicao.seasonal)
plt.title('Sazonalidade')
plt.grid(True)

# Ruído
plt.subplot(4, 1, 3)
plt.plot(decomposicao.resid)
plt.title('Ruído')
plt.grid(True)

# Série temporal original
plt.subplot(4, 1, 4)
plt.plot(df['FECHAMENTO'])
plt.title('Série Temporal Original')
plt.grid(True)

plt.tight_layout()
plt.show()

#%%
# Identifica os pontos de alta e baixa
# Usando uma janela deslizante para encontrar picos e vales
def encontrar_picos_valleys(data, janela):
  """Identifica os pontos de alta e baixa usando uma janela deslizante.

  Args:
    data: A série temporal.
    janela: O tamanho da janela deslizante.

  Returns:
    picos: Um array de índices dos pontos de alta.
    valleys: Um array de índices dos pontos de baixa.
  """
  picos, valleys = [], []
  for i in range(janela, len(data) - janela):
    janela_atual = data[i - janela:i + janela + 1]
    if data[i] == max(janela_atual):
      picos.append(i)
    if data[i] == min(janela_atual):
      valleys.append(i)
  return np.array(picos), np.array(valleys)

# Define o tamanho da janela deslizante
janela = 5

# Encontra os pontos de alta e baixa
picos, valleys = encontrar_picos_valleys(df['FECHAMENTO'], janela)

# Plota a série temporal com os pontos de alta e baixa marcados
plt.figure(figsize=(12, 6))
plt.plot(df['FECHAMENTO'], label='Preço de Fechamento')
plt.scatter(df.index[picos], df['FECHAMENTO'][picos], c='green', label='Pontos de Alta')
plt.scatter(df.index[valleys], df['FECHAMENTO'][valleys], c='red', label='Pontos de Baixa')
plt.title('Preço de Fechamento da Ação ELET3 - Pontos de Alta e Baixa')
plt.xlabel('Data')
plt.ylabel('Preço (R$)')
plt.legend()
plt.grid(True)
plt.show()

#%%
# Teste de raiz unitária de Dickey-Fuller (ADF)
# Verifica se a série temporal é estacionária

resultado_adf = adfuller(df['FECHAMENTO'])
print('Estatística ADF:', resultado_adf[0])
print('Valor p:', resultado_adf[1])
print('Valores críticos:')
for chave, valor in resultado_adf[4].items():
    print(f'\t{chave}: {valor}')

# Interpreta os resultados
if resultado_adf[1] <= 0.05:
    print('A série temporal é estacionária (rejeita a hipótese nula).')
else:
    print('A série temporal não é estacionária (não rejeita a hipótese nula).')