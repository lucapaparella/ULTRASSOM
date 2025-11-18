import h5py, numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
from pathlib import Path
import os, matplotlib

# matplotlib.use("Agg")     # sem X11 → salva em arquivo

#limpar a tela
os.system("clear")

# Caminho do arquivo
path = r"/home/users/lpaparella/ULTRASSOM/IMAGENS/1_CUBDL_Task1_Data/UFL001.hdf5"
# Abrindo o arquivo

arquivo = h5py.File(path, "r")

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


print("PASSO 1 => EXPLORANDO O ARQUIVO HDF5".center(64))
print("*"*80)

def nome_e_tipo(nome, obj):
  if isinstance(obj, h5py.Group):
      print(f"GRUPO => {nome}")
  elif isinstance(obj, h5py.Dataset):
      print(f"DATASET => {nome.center(35)} | {str(obj.shape).center(15)} | {obj.dtype}")

arquivo.visititems(nome_e_tipo)

print("*"*80)