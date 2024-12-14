# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from statsmodels.tsa.stattools import adfuller
from numpy.fft import fft
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# Carregar dados usando pandas
df = pd.read_csv('..\\Data\\Eletrobras ELET3 - Histórico  InfoMoney.csv')

# Converter virgulas para pontos e substituir n/d para NaN
for col in ['ABERTURA', 'FECHAMENTO', 'VARIAÇÃO', 'MÍNIMO', 'MÁXIMO']:
    df[col] = df[col].str.replace(",", ".").replace("n/d", np.nan).astype(float)

# Função para converter volume
def convert_volume(volume):
    if isinstance(volume, str):
      volume = volume.upper()
      if "M" in volume:
        return float(volume.replace("M", "").replace(",", ".")) * 1000000
      elif "B" in volume:
        return float(volume.replace("B", "").replace(",", ".")) * 1000000000
      else:
        return np.nan
    else:
        return np.nan

# Converter a coluna 'VOLUME'
df['VOLUME'] = df['VOLUME'].apply(convert_volume)

# Converter coluna data para data
df['DATA'] = pd.to_datetime(df['DATA'], dayfirst=True)

# Definir a data como indice
df = df.set_index('DATA')

# %%
# Gráfico da Série Original
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['MÁXIMO'], linestyle='-', color='green', label='Máximo de ELET3')
plt.title('Variação do Valor Máximo de ELET3 ao Longo do Tempo', fontsize=14)
plt.xlabel('2023 - 2024', fontsize=12)
plt.ylabel('Valor Máximo (USD)', fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## **Remoção da Tendência da Série**

# %%
def regressao_linear(x, y):
    x_sum = sum(x)
    y_sum = sum(y)
    x_sum2 = x_sum ** 2
    x2 = [xi ** 2 for xi in x]
    x2_sum = sum(x2)
    xy = [xi * yi for xi, yi in zip(x, y)]
    xy_sum = sum(xy)
    n = len(x)
    b1 = (x_sum * y_sum - n * xy_sum)  / (x_sum2 - n * x2_sum)
    b0 = (y_sum - b1 * x_sum) / n
    ym = [b0 + b1 * xi for xi in x]
    e2 = [(yi - ymi) ** 2 for yi, ymi in zip(y, ym)]
    e2_sum = sum(e2)
    y2 = [yi ** 2 for yi in y]
    y2_sum = sum(y2)
    y_sum2 = y_sum ** 2
    r2 = 1 - (e2_sum)/(y2_sum - y_sum2/n)
    return b0, b1, r2

# %%
# Utilizamos o indice do DataFrame para calcular a regressão
b0, b1, r2 = regressao_linear(range(len(df)), df['MÁXIMO'])

# Calcular valores ajustados (tendência)
tendencia = [b0 + b1 * xi for xi in range(len(df))]

# Remoção da tendência (resíduos)
serie_sem_tendencia = [yi - y_aj for yi, y_aj in zip(df['MÁXIMO'], tendencia)]
df['sem_tendencia'] = serie_sem_tendencia


# Visualização
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['MÁXIMO'], label="Série Original", marker="o")
plt.plot(df.index, tendencia, label="Tendência Ajustada", linestyle="--")
plt.plot(df.index, serie_sem_tendencia, label="Série Sem Tendência", marker="x")
plt.legend()
plt.title("Remoção de Tendência com Regressão Linear")
plt.xlabel("Tempo")
plt.ylabel("Valores")
plt.grid()
plt.show()


# %%
X = range(len(df))
Y = df['MÁXIMO']

# Modelo linear por regressão linear
b0, b1, r2 = regressao_linear(X, Y)
print(b0)
print(b1)
print(r2)

# Curva usando modelo linear
ymodelo = [b0 + b1 * xi for xi in X]
df['Comp.Tendencia'] = ymodelo

# Série Real
plt.figure(figsize=(10, 6))
plt.plot(df.index, Y, label='Série Real', color='green')

# Tendência - Função Personalizada
plt.plot(df.index, ymodelo, 'r-', label='Tendência (Regressão Linear Customizada)')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Comparação da Série Real com as Tendências Extraídas')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nDados com Tendências:")
print(df)

# %%
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['sem_tendencia'], label='Série Sem Tendência', color='green')
plt.xlabel('2023 - 2024')
plt.ylabel('Valores para Y (Sem Tendência)')
plt.title('Série Temporal Sem Tendência')
plt.legend()
plt.show()

# %%
from statsmodels.tsa.stattools import adfuller

estacionaria = df['sem_tendencia']
adf_test = adfuller(estacionaria)
print("Teste ADF (Série Sem Tendência):")
print(f"ADF Statistic: {adf_test[0]}")
print(f"p-value: {adf_test[1]}")
if adf_test[1] < 0.05:
    print("A série sem tendência é estacionária.")
else:
    print("A série sem tendência não é estacionária. Diferenciação necessária.")

# %% [markdown]
# **Calculando a Primeira Diferença**

# %%
df['primeira_diferenca'] = df['sem_tendencia'].diff().dropna()

# %%
df = df.dropna(subset=['primeira_diferenca'])
# %%
estacionaria1 = df['primeira_diferenca']
adf_test = adfuller(estacionaria1)
print("Teste ADF (Primeira Diferença):")
print(f"ADF Statistic: {adf_test[0]}")
print(f"p-value: {adf_test[1]}")
if adf_test[1] < 0.05:
    print("A série diferenciada é estacionária.")
else:
    print("A série diferenciada não é estacionária. Diferenciação necessária.")

# %%
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['primeira_diferenca'], label='Série Primeira Diferença', color='green')
plt.xlabel('2023 - 2024')
plt.ylabel('Valores para Y')
plt.title('Série Temporal - Primeira Diferença')
plt.legend()
plt.show()

# %% [markdown]
# # **Aplicando a FFT**

# %%
dados = df['primeira_diferenca'].values
x = list(range(len(df)))

sr = 10
ts = 1.0 / sr
t = np.array(x)

X = fft(dados)
N = len(X)
n = np.arange(N)
T = N / sr
freqFft = n / T

amplitude = np.abs(X) / (N/2)
phase = np.angle(X, deg=True)
plt.figure(figsize=(10, 7))

plt.subplot(2,1,1)
plt.stem(freqFft, np.abs(X) / (N/2), 'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |X(freq)|')

plt.subplot(2,1,2)
plt.stem(freqFft, np.angle(X, deg=True), 'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Phase (X(freq))')
plt.show()

# %%
def find_high_amplitude_phase(ampli, phase, ffreq, threshold=0.124):
    high_amplitude_indices = np.where(ampli > threshold)
    high_amplitude_frequencies = ffreq[high_amplitude_indices]
    high_amplitude_phases = phase[high_amplitude_indices]
    ampli_indice = ampli[high_amplitude_indices]
    return high_amplitude_frequencies, high_amplitude_phases,  ampli_indice

valores = find_high_amplitude_phase(amplitude, phase, freqFft, threshold=0.124)

df_resultado = pd.DataFrame({
    "Frequência (Hz)": valores[0],
    "Fase (radianos)": valores[1],
    "Amplitude": valores[2]
})

# Exibindo a tabela
print("Frequências com amplitudes maiores que 0.124:")
print(df_resultado)

# %%
# Reconstrução da Série Usando FFT
fft_formula = ""
for i in range(len(valores[0])):
    amplitude_i = valores[2][i]
    frequency_i = valores[0][i]
    phase_i = valores[1][i]
    phase_rad = phase_i * (np.pi / 180)
    fft_formula += f"{amplitude_i:.6f} * np.cos(2 * np.pi * {frequency_i:.6f} * t - ({phase_rad:.6f})) + "
fft_formula = fft_formula.rstrip(" + ")  #

print("Fórmula Gerada:")
print(fft_formula)

try:
  X_fft = eval(fft_formula)
except SyntaxError as e:
    print(f"Erro de sintaxe na fórmula: {e}")
    X_fft = np.zeros(len(df))
# %%
X_real = df['primeira_diferenca'].values

fig, axs = plt.subplots(3, 1, figsize=(10, 7))

# Gráfico da série real
axs[0].plot(t, X_real, label='Série Real', color='blue')
axs[0].set_title('Série Real')
axs[0].set_xlabel('Time (x)')
axs[0].set_ylabel('Amplitude')
axs[0].legend()
axs[0].grid(True)

# Gráfico da série deduzida
axs[1].plot(t, X_fft, label='Série Deduzida', color='red')
axs[1].set_title('Série Deduzida')
axs[1].set_xlabel('Time (x)')
axs[1].set_ylabel('Amplitude')
axs[1].legend()
axs[1].grid(True)

# Gráfico comparativo entre a série real e deduzida
axs[2].plot(t, X_real, label='Série Real', color='blue')
axs[2].plot(t, X_fft, label='Série Deduzida', color='red', linestyle='--')
axs[2].set_title('Comparação Séries')
axs[2].set_xlabel('Time (x)')
axs[2].set_ylabel('Amplitude')
axs[2].legend()
axs[2].grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
#  **APLICAÇÃO DOS MODELOS DE PREVISÃO - ARIMAX**

# %%
plot_acf(df['primeira_diferenca'])
# %%
plot_pacf(df['primeira_diferenca'])
# %%
# Definindo a variável alvo e as variáveis exógenas
target = 'MÁXIMO'
features = ['ABERTURA', 'FECHAMENTO', 'VARIAÇÃO', 'MÍNIMO', 'VOLUME']

# Remover NaN da serie
df_full = df.dropna(subset=[target] + features)

# Criar Lags das variáveis exógenas
df_full[[f'{col}_lag_1' for col in features]] = df_full[features].shift(1)
df_full = df_full.dropna()

# Separar dados
train_size = int(len(df_full) * 0.8)
train_data = df_full.iloc[:train_size]
test_data = df_full.iloc[train_size:]

# Normalização/Padronização
scaler_x = StandardScaler()
train_x = scaler_x.fit_transform(train_data[[f'{col}_lag_1' for col in features]])
test_x = scaler_x.transform(test_data[[f'{col}_lag_1' for col in features]])

scaler_y = StandardScaler()
train_y = scaler_y.fit_transform(train_data[target].values.reshape(-1,1)).flatten()
test_y = scaler_y.transform(test_data[target].values.reshape(-1,1)).flatten()

# 1. ARIMAX
order = (1, 0, 1)  # Define a ordem do modelo ARIMA
seasonal_order = (1, 1, 1, 4)  # Define a ordem da parte sazonal s=4
arimax_model = SARIMAX(train_y,
                       exog=train_x,
                       order=order,
                       seasonal_order=seasonal_order,
                       enforce_stationarity=False,
                       enforce_invertibility=False)
arimax_fit = arimax_model.fit()

arimax_predictions = arimax_fit.predict(start=0,
                                        end=len(train_y)-1,
                                        exog=train_x)

arimax_predictions_test = arimax_fit.predict(start=len(train_y),
                                                end=len(df_full)-1,
                                                exog=test_x)

arimax_predictions_desnormalized = scaler_y.inverse_transform(arimax_predictions.reshape(-1, 1)).flatten()
arimax_predictions_desnormalized_test = scaler_y.inverse_transform(arimax_predictions_test.reshape(-1, 1)).flatten()

arimax_rmse = np.sqrt(mean_squared_error(test_y, arimax_predictions_test))
print(f"ARIMAX RMSE: {arimax_rmse}")

# Exibir o gráfico com as previsões
plt.figure(figsize=(10, 6))
plt.plot(df_full.index, df_full[target], label='Original', color='blue')
plt.plot(train_data.index, arimax_predictions_desnormalized, label='ARIMAX Forecast Train', color='green', linestyle = '--')
plt.plot(test_data.index, arimax_predictions_desnormalized_test, label='ARIMAX Forecast Test', color='red', linestyle = '--')
plt.title('Comparação entre valor original e previsões')
plt.xlabel('Data')
plt.ylabel('Valor Máximo')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
# %%
