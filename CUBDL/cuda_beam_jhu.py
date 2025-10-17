import h5py, numpy as np, matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os

#limpar a tela
os.system("cls")

# Caminho do arquivo
path = r"C:\Users\lucap\Documents\CUBDL_Data\CUBDL_Data\2_Post_CUBDL_JHU_Breast_Data\JHU030.hdf5"
# Abrindo o arquivo

arquivo = h5py.File(path, "r")
print("PASSO 1 => EXPLORANDO O ARQUIVO HDF5".center(64))
print("*"*64)

def nome_e_tipo(nome, obj):
  if isinstance(obj, h5py.Group):
      print(f"GRUPO => {nome}")
  elif isinstance(obj, h5py.Dataset):
      print(f"DATASET => {nome.center(25)} | {str(obj.shape).center(15)} | {obj.dtype}")

arquivo.visititems(nome_e_tipo)

print("*"*64)
input("Aperte ENTER para continuar...")



print("PASSO 2 => ATIVANDO GPU SE EXISTIR".center(64))
print("*"*64)
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
input("Aperte ENTER para continuar...")
print("PASSO 2 => CRIANDO OS VETORES".center(64))
print("*"*64)
 # Get data
idata = np.array(arquivo["channel_data"], dtype="float32")
print(f"idata => {idata.shape}")
# qdata = np.imag(hilbert_xp(idata, axis=-1))
# sinal analítico por canal (interp fracionária estável de fase)
angles = np.array(arquivo["angles"])
fc = np.array(arquivo["modulation_frequency"]).item()
fs = np.array(arquivo["sampling_frequency"]).item()
c = np.array(arquivo["sound_speed"]).item()

# [Gravação começa] ----(ex.: espera 2 µs)---- [Pulso é emitido] ---- ecos retornam ---->
# o -1 faz que o tempo do pulso emitido seja 0
tempo_zero = -1 * np.array(arquivo["time_zero"], dtype="float32")
print(f"tempo_zero => {tempo_zero.shape}")

fdemod = 0

# Transforma o dataset em uma lista de posições dos elementos (128,) => (128, 3) com [x, y, z] ????????
posicoes_elementos = np.array(arquivo["element_positions"], dtype="float32")
print(f"posicoes_elementos => {posicoes_elementos.shape}")
# print(f"posicoes_elementos => {posicoes_elementos}")
# ele_pos = np.stack([posicoes_elementos, 0 * posicoes_elementos, 0 * posicoes_elementos], axis=1)
# print(f"ele_pos=> {ele_pos.shape}")
# print(f"ele_pos dados => {ele_pos}")

# GRADE de imagem (CPU primeiro; depois mandamos p/ GPU se houver)
# Profundidade (zlims)
limite_z = np.array([0e-3, idata.shape[2] * c / fs / 2])
print(f"zlims => {limite_z}")
# Largura Sonda (xlims)
limite_x = np.array([posicoes_elementos[0], posicoes_elementos[-1]])
print(f"xlims => {limite_x}")



# --- Helper para criar array no backend correto (NumPy ou CuPy)
def as_xp(a, dtype=None):
    if xp is np:
        return np.asarray(a, dtype=dtype)
    else:
        return xp.asarray(a, dtype=dtype)
idata = as_xp(idata, dtype=(xp.float32 if xp is not np else np.float32))
# sinal analítico por canal (interp fracionária estável de fase)
qdata = hilbert_xp(idata, axis=-1).astype(xp.complex64)
print(f"qdata => {qdata.shape}")
print(f"angles => {angles.shape}")

