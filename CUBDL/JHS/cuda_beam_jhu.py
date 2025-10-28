import h5py, numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
from pathlib import Path
import os, matplotlib

# matplotlib.use("Agg")     # sem X11 → salva em arquivo

#limpar a tela
os.system("cls")

# Caminho do arquivo
path = r"/home/users/lpaparella/ULTRASSOM/IMAGENS/2_Post_CUBDL_JHU_Breast_Data/JHU030.hdf5"
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
tamanho_pixel = np.array(arquivo["pixel_d"]).item()
print(f"Tamanho pixel => {tamanho_pixel}")
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

#Amostras 
amostras = idata.shape[2] 

# GRADE de imagem (CPU primeiro; depois mandamos p/ GPU se houver)

# Profundidade Fisica em metros
z_max= amostras * c / fs / 2
print(f"z_max  => {z_max}")
# Posição minima e mxima dos eleentr=so do trasdutor
x_min = posicoes_elementos[0]
x_max = posicoes_elementos[-1]


# Essa fórmula calcula quantos pixels (nx) cabem na largura total (x_max - x_min) 
# o +1 é utilizado para incluir a borda
nx = round((x_max - x_min)/tamanho_pixel) + 1
print(f"nx => {nx}")
# Mesma lógica para profundidade:
nz = round(z_max/tamanho_pixel) + 1
print("nz =", nz)

# Largura Sonda (xlims)
limite_x = np.linspace(x_min, x_max, nx)
print(f"limite_x => {limite_x}")
# limites da profondidade
limite_z = np.linspace(0e-3, z_max,nz)
print(f"limite_z=> {limite_z}")



# --- HELPER para criar array no backend correto (NumPy ou CuPy)
def as_xp(a, dtype=None):
    if xp is np:
        return np.asarray(a, dtype=dtype)
    else:
        return xp.asarray(a, dtype=dtype)
    
#chamada do HELPER
idata = as_xp(idata, dtype=(xp.float32 if xp is not np else np.float32))

# sinal analítico por canal (interp fracionária estável de fase)
qdata = hilbert_xp(idata, axis=-1).astype(xp.complex64)
print(f"qdata => {qdata.shape}")
print(f"angles => {angles.shape}")

escala = 1e3
img = arquivo["/beamformed_data"][:]



img = 10* np.log10(np.abs(img.T) / np.max(np.abs(img)))


plt.figure(figsize=(6, 8))
plt.imshow(img, cmap="gray", origin="upper", aspect="auto",
           extent=[x_min*escala, x_max*escala, z_max*escala, 0*escala])
# plt.imshow(img, cmap="gray", origin="upper", aspect="auto")
plt.title("Imagem beamformed (transposta)")
plt.xlabel("Lateral (x)")
plt.ylabel("Profundidade (z)")
plt.show()

# >>> Salva em ./IMAGENS_SALVAS/reconstrucao.png (ou reconstrucao_1.png, etc.)
save_fig(nome_base="reconstrucao")
