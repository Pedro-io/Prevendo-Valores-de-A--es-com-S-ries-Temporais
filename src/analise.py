#%% 
import pandas as pd 
import matplotlib.pyplot as plt

#%% 
arquivo = "..\Data\Eletrobras ELET3 - Histórico  InfoMoney.csv"

df = pd.read_csv(arquivo, index_col=None)

# %%
df.head()
# %%
df['time'] = pd.to_datetime(df['time'])  # Converte a coluna time para o formato datetime
df.set_index('time', inplace=True)       # Define a coluna time como índice

#%%
# Plotar as colunas open, high, low, close como linhas
plt.figure(figsize=(14, 7))  # Tamanho do gráfico
plt.plot(df['open'], label='Open', color='blue')
plt.plot(df['high'], label='High', color='green')
plt.plot(df['low'], label='Low', color='red')
plt.plot(df['close'], label='Close', color='black')

# Adicionar títulos e legendas
plt.title('Série Temporal - Abertura, Alta, Baixa e Fechamento')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='best')
plt.grid(True)  # Adiciona uma grade para facilitar a leitura dos dados
plt.show()
# %%
