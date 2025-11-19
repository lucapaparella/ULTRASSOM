import h5py, numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
from pathlib import Path
import os, matplotlib

#limpar a tela
os.system("clear")
acq = 1
# Caminho do arquivo
caminho = r"/home/users/lpaparella/ULTRASSOM/IMAGENS/1_CUBDL_Task1_Data/"
# Make sure the selected dataset is valid
dataset = "UFL{:03d}".format(acq) + ".hdf5"
path = caminho + dataset
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

# --------- Backend: CUDA (CuPy) se disponível; caso contrário NumPy ----------
USE_CUDA = True
xp = np
to_cpu = lambda a: a
hilbert_xp = None

if USE_CUDA:
    try:
        import cupy as cp
        from cupyx.scipy.signal import hilbert as c_hilbert
        if cp.cuda.runtime.getDeviceCount() > 0:
            xp = cp
            hilbert_xp = c_hilbert
            to_cpu = cp.asnumpy
            dev = cp.cuda.Device()
            dev.use()
            print(f"[CUDA] Usando GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
        else:
            print("[CUDA] Nenhuma GPU detectada. Usando CPU (NumPy).")
    except Exception as e:
        print(f"[CUDA] CuPy indisponível ({e}). Usando CPU (NumPy).")

if hilbert_xp is None:
    # fallback para CPU
    from scipy.signal import hilbert as s_hilbert
    hilbert_xp = s_hilbert

# --------- FIM Backend: CUDA (CuPy) se disponível; caso contrário NumPy ----------

 # Phantom-specific parameters
if acq == 1:
    sound_speed = 1526
elif acq == 2 or acq == 4 or acq == 5:
    sound_speed = 1523
else:
    sound_speed = 1525

# Get data
idata = xp.array(arquivo["channel_data"], dtype="float32")
qdata = xp.imag(hilbert_xp(idata, axis=-1))

# transforma ângulos que estão em graus para ângulos em radianos
angles = xp.array(arquivo["angles"]) * np.pi / 180
# print(f"angles => {angles}")

fc = xp.array(arquivo["modulation_frequency"]).item()
fs = xp.array(arquivo["channel_data_sampling_frequency"]).item()
c = sound_speed  # np.array(arquivo["sound_speed"]).item()
time_zero = -1 * np.array(arquivo["channel_data_t0"], dtype="float32")
fdemod = fc

# Make the element positions based on LA533 geometry
pitch = 0.245e-3
nelems = idata.shape[1]
xpos = xp.arange(nelems) * pitch
xpos -= xp.mean(xpos)
ele_pos = xp.stack([xpos, 0 * xpos, 0 * xpos], axis=1)
