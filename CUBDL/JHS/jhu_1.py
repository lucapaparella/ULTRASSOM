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

# Entradas esperadas

# P: objeto com metadados e parâmetros do exame:
P = np.array(arquivo["channel_data"], dtype="float32")
# P.angles (nangles,), radianos
angles = xp.array(arquivo["angles"])
print(f"ANGLES => {angles.shape}")
# P.ele_pos (nelems, 3), posições (x,y,z) dos elementos
ele_pos = xp.array(arquivo["element_positions"], dtype="float32")
print(f"ELE_POS => {ele_pos.shape}")
# P.fc, P.fs, P.fdemod, P.c, P.time_zero (arrays/escalares)
fc = xp.array(arquivo["modulation_frequency"]).item()
fs = xp.array(arquivo["sampling_frequency"]).item()
fdemod = 0
time_zero = -1 * np.array(arquivo["time_zero"], dtype="float32")
print(f"FC => {fc}")
print(f"FS => {fs}")
print(f"FDEMOD => {fdemod}")
print(f"TIME_ZERO => {time_zero.shape}")
# grid: (ncols, nrows, 3) com (x,y,z) dos pixels (pode ser 2D ou 3D — o código achata).


# => garantir que as variáveis ang_list e ele_list tenham um formato consistente (iterável)
# ang_list: índices de ângulos que serão usados (se None, usa todos).
ang_list = range(angles.shape[0])
print(f"ang_list => {ang_list}")
# ele_list: índices de elementos Rx (se None, usa todos).
ele_list = range(ele_pos.shape[0])
print(f"ele_list => {ele_list}")

c =xp.array(arquivo["sound_speed"]).item()
tamanho_pixel = xp.array(arquivo["pixel_d"]).item()
print(f"Tamanho pixel => {tamanho_pixel}")
# [Gravação começa] ----(ex.: espera 2 µs)---- [Pulso é emitido] ---- ecos retornam ---->
# o -1 faz que o tempo do pulso emitido seja 0

#============================================
#   DESENCOLVENDO O GRID                    
#============================================
# eps serve para:
eps = 1e-10
# Em Python (e em quase todas as linguagens), números em ponto flutuante sofrem com pequenos erros de precisão
# corrigir erros de arredondamento;
# garantir que o limite máximo (xmax, zmax) entre no vetor;
# e evitar comparações com zero que poderiam gerar divisões por zero (em outros contextos)

amostras = P.shape[2] 

# GRADE de imagem (CPU primeiro; depois mandamos p/ GPU se houver)

# Profundidade Fisica em metros
z_max= amostras * c / fs / 2
print(f"z_max  => {z_max}")
# Posição minima e mxima dos eleentr=so do trasdutor
x_min = ele_pos[0]
x_max = ele_pos[-1]
# Essa fórmula calcula quantos pixels (nx) cabem na largura total (x_max - x_min) 
# o +1 é utilizado para incluir a borda
dx = xp.round((x_max - x_min)/tamanho_pixel) + 1 # round => arredondamento para interiro
dx = int(dx) # converter em inteiro para não dar erro no cupy
# Mesma lógica para profundidade:
dz = xp.round(z_max/tamanho_pixel) + 1
dz = int(dz)

# Largura Sonda (xlims)
limite_x = xp.linspace(x_min, x_max, dx)
print(f"limite_x => {limite_x}")
# limites da profondidade
limite_z = xp.linspace(0e-3, z_max,dz)
print(f"limite_z=> {limite_z}")

zz, xx = np.meshgrid(limite_z, limite_x, indexing="ij")
yy = 0 * xx
grid = np.stack((xx, yy, zz), axis=-1)
print(f"GRID => {grid.shape}")
grid_reshape = grid.reshape(-1,3)
print(f"GRID RESHAPE => {grid_reshape.shape}")
grid_out = grid.shape[:-1] # TUPLA
print(f"GRID OUT => {grid_out}")
#=======================================================================

# Numero de angulos
nangles = len(ang_list)
print(f"Numero de angulos => {nangles}") 
# Numero de elementos
nelems  = len(ele_list)
print(f"Numero de elementos => {nelems}") 
# Numero de pixel
npixels   = grid_reshape.shape[0]
print(f"Numero de Pixels => {npixels}") 

# Initialize delays, apodizations, output array
txdel = xp.zeros(nangles, npixels)
rxdel = xp.zeros(nelems, npixels)
txapo = xp.ones(nangles, npixels)
rxapo = xp.ones(nelems, npixels)


#atraso de onda plana
## Compute distance to user-defined pixels for plane waves
# Expects all inputs to be numpy arrays specified in SI units.
# INPUTS
#   grid    Pixel positions in x,y,z    [npixels, 3]
#   angles  Plane wave angles (radians) [nangles]
# OUTPUTS
#   dist    Distance from each pixel to each element [nelems, npixels]

def delay_plane(grid, angles):
    # Use broadcasting to simplify computations
    x = xp.expand_dims(grid[:, 0], 0) # “esticando” (broadcasting)
    z = xp.expand_dims(grid[:, 2], 0) # “esticando” (broadcasting)
    # For each element, compute distance to pixels
    dist = x * xp.sin(angles) + z * xp.cos(angles)
    return dist

# Ela cria uma máscara (apodização)

for i, tx in enumerate(ang_list):
    txdel[i] = delay_plane(grid, angles[tx])
    txdel[i] += time_zero[tx] * c
    txapo[i] = apod_plane(grid, angles[tx], limite_x)

for j, rx in enumerate(ele_list):
    rxdel[j] = delay_focus(self.grid, self.ele_pos[rx])
    rxapo[j] = apod_focus(self.grid, self.ele_pos[rx])



