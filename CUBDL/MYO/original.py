import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

#limpar a tela
os.system("clear")

# Caminho do arquivo HDF5
path = r"/home/users/lpaparella/ULTRASSOM/IMAGENS/1_CUBDL_Task1_Data/MYO005.hdf5"

# Abre o arquivo
with h5py.File(path, "r") as f:
    Id = np.array(f["beamformed_Idata"])
    Qd = np.array(f["beamformed_Qdata"])

# Combina as partes I e Q em um sinal complexo
bimg = Id + 1j * Qd

# Calcula a magnitude (amplitude)
bimg = np.abs(bimg)

# Normaliza e converte para dB
bimg /= (bimg.max() + 1e-12)
bimg_db = 20 * np.log10(bimg + 1e-12)

# Limita a faixa dinâmica (por exemplo, -60 dB)
bimg_db = np.clip(bimg_db, -60, 0)

print(f"BIM DB => {bimg_db.shape}")

#salvar arquivo
def save_fig(fig=None, nome_base="reconstrucao", pasta="IMAGENS_SALVAS", dpi=200):
    """Salva a figura atual em ./IMAGENS_SALVAS/<nome_base>[_N].png e imprime o caminho."""
    fig = fig or plt.gcf()
    Path(pasta).mkdir(parents=True, exist_ok=True)
    i = 0
    while True:
        sufixo = "" if i == 0 else f"_{i}"
        path = Path(pasta) / f"{nome_base}{sufixo}.png"
        if not path.exists():
            break
        i += 1
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[OK] Figura salva em: {path}")

# Exibe a imagem
plt.figure(figsize=(6, 6))
plt.imshow(bimg_db.T, cmap="gray", vmin=-60, vmax=0, aspect="auto")
plt.colorbar(label="dB")
plt.title("Imagem Beamformada (Magnitude em dB)")
plt.xlabel("Amostras (x)")
plt.ylabel("Linhas (z)")
plt.show()
save_fig(nome_base="original")