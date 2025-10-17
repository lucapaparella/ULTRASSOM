import numpy as np
import matplotlib.pyplot as plt

# Parâmetros do sinal
fs = 20e6         # frequência de amostragem (20 MHz)
t = np.arange(0, 10e-6, 1/fs)  # 10 µs de duração
fc = 5e6          # frequência central (5 MHz)

# Pulso ultrassônico (seno com janela gaussiana)
envelope = np.exp(-((t - 5e-6)**2) / (0.5e-6)**2)
rf = np.sin(2*np.pi*fc*t) * envelope

# "time_zero" informado pelo HDF5 (pulso começa 2 µs após início da aquisição)
time_zero = 2e-6

# Eixo de tempo com e sem inversão
t_sem_corr = t + time_zero       # sem multiplicar por -1
t_corr = t - time_zero           # com multiplicação por -1

plt.figure(figsize=(10,5))
plt.plot(t_sem_corr*1e6, rf, 'b-', label='Sem inverter time_zero (+)')
plt.plot(t_corr*1e6, rf, 'r--', label='Com inverter time_zero (-)')
plt.axvline(0, color='k', linestyle=':', label='t = 0 (emissão)')
plt.xlabel("Tempo relativo (µs)")
plt.ylabel("Amplitude")
plt.title("Efeito da inversão de time_zero")
plt.legend()
plt.grid(True)
plt.show()
