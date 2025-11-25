import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

#limpar a tela
os.system("clear")

# Caminho do arquivo HDF5
path = r"/home/users/lpaparella/ULTRASSOM/IMAGENS/1_CUBDL_Task1_Data/EUT003.hdf5"

# Abre o arquivo
with h5py.File(path, "r") as f:
    # Id = np.array(f["beamformed_Idata"])
    # Qd = np.array(f["beamformed_Qdata"])

# Combina as partes I e Q em um sinal complexo
    bimg = np.array(f["beamformed_data"])



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
bimg_log = 20 * np.log10(np.abs(bimg) + 1e-12)
bimg_log -= bimg_log.max()   # normaliza para 0 dB
plt.imshow(bimg_log.T, cmap="gray", aspect="auto")
plt.colorbar()
plt.show()
save_fig(nome_base="original")