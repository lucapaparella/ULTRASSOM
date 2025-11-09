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
path = r"/home/users/lpaparella/ULTRASSOM/IMAGENS/1_CUBDL_Task1_Data/INS004.hdf5"
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
print("*"*64)

def nome_e_tipo(nome, obj):
  if isinstance(obj, h5py.Group):
      print(f"GRUPO => {nome}")
  elif isinstance(obj, h5py.Dataset):
      print(f"DATASET => {nome.center(25)} | {str(obj.shape).center(15)} | {obj.dtype}")

arquivo.visititems(nome_e_tipo)

print("*"*64)

posicoes_elementos = np.array(arquivo["element_positions"], dtype="float32")

posicoes_elementos[0] -= np.mean(posicoes_elementos[0])
print(f"posicoes_elementos => {posicoes_elementos}")

print(f"posicoes_elementos => {posicoes_elementos[-1, 0]}")

transmit_direction = np.array(arquivo["transmit_direction"], dtype="float32")
# print(f"transmit_direction => {transmit_direction}")

transmit_count = np.array(arquivo["transmit_count"], dtype="float32")
# print(f"transmit_count => {transmit_count}")

pixel_positions = np.array(arquivo["pixel_positions"], dtype="float32")
# print(f"pixel_positions => {pixel_positions}")

time_zero = -1 * np.array(arquivo["start_time"], dtype="float32")
# print(f"time_zero => {time_zero}")