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
# input("Aperte ENTER para continuar...")
#------------------------------------------------------------------------------------
idata = xp.array(arquivo["channel_data"], dtype="float32")
print(f"IDATA => {idata.shape}")
# P.angles (nangles,), radianos
angles = xp.array(arquivo["angles"])
print(f"ANGLES => {angles.shape}")
# P.ele_pos (nelems, 3), posições (x,y,z) dos elementos
ele_pos = xp.array(arquivo["element_positions"], dtype="float32")
print(f"ELE_POS => {ele_pos}")


# P.fc, P.fs, P.fdemod, P.c, P.time_zero (arrays/escalares)
fc = xp.array(arquivo["modulation_frequency"]).item()
fs = xp.array(arquivo["sampling_frequency"]).item()
c =xp.array(arquivo["sound_speed"]).item()
fdemod = 1

# [Gravação começa] ----(ex.: espera 2 µs)---- [Pulso é emitido] ---- ecos retornam ---->
# o -1 faz que o tempo do pulso emitido seja 0
tempo_zero = -1 * np.array(arquivo["time_zero"], dtype="float32")
print(f"tempo_zero => {tempo_zero.shape}")

print(f"FC => {fc}")
print(f"FS => {fs}")
print(f"FDEMOD => {fdemod}")

# grid: (ncols, nrows, 3) com (x,y,z) dos pixels (pode ser 2D ou 3D — o código achata).


# => garantir que as variáveis ang_list e ele_list tenham um formato consistente (iterável)
# ang_list: índices de ângulos que serão usados (se None, usa todos).
ang_list = range(angles.shape[0])
print(f"ang_list => {ang_list}")
# ele_list: índices de elementos Rx (se None, usa todos).
ele_list = range(ele_pos.shape[0])
print(f"ele_list => {ele_list}")
# tamanho_pixel = xp.array(arquivo["pixel_d"]).item()
# print(f"Tamanho pixel => {tamanho_pixel}")

#-------------------------------------------------------
# P = PICMUSData(database_path, acq, target, dtype)
P = xp.array(arquivo["channel_data"], dtype="float32")
#-------------------------------------------------------
# Define pixel grid limits (assume y == 0)
ele_pos = xp.array(arquivo["element_positions"], dtype="float32")
xlims = [ele_pos[0], ele_pos[-1]]
print(f"XLIMS =>{xlims}")
# "Eu quero que a imagem de ultrassom comece a 5 mm de profundidade e termine a 55 mm "
# "de profundidade."
# Profundidade Fisica em metros

# profundidade mínima por ângulo (amostra n=0)
z_start_per_angle = xp.maximum(0.0, (-tempo_zero[0]) * c/2)

# profundidade máxima por ângulo (amostra n=Ns-1)
z_end_per_angle = (((idata[2] - 1) / fs) - tempo_zero[0]) * (c/2)

# intervalo comum a todos os ângulos
zmin = float(xp.max(z_start_per_angle))
zmax = float(xp.min(z_end_per_angle))

# limites da profondidade
zlims = (zmin, zmax)
print("zlims (m) =", zlims[1])




#-------------------------------------------------------
# calculo do pixel
#-------------------------------------------------------
c =xp.array(arquivo["sound_speed"]).item()
fc = xp.array(arquivo["modulation_frequency"]).item()
wvln = c / fc #comprimento da onda
# Isso define o tamanho horizontal de cada pixel
# O código define a largura do pixel como um terço (1/3) do comprimento de onda.
# É uma regra de amostragem (conhecida como critério de Nyquist). 
# Para reconstruir digitalmente uma onda sem perder informação, 
# você precisa de pelo menos 2 amostras (pixels) por comprimento de onda. 
# sar 3 (como em wvln / 3) é mais seguro e garante uma imagem de melhor qualidade.
dx = wvln / 3 #largura do pixel
# o código está simplesmente garantindo que os pixels sejam quadrados
dz = dx  # Use square pixels
print(f"DZ => {dz}")
#-------------------------------------------------------
tamanho_pixel = xp.array(arquivo["pixel_d"]).item()
print(f"Tamanho pixel => {tamanho_pixel}")
# grid = make_pixel_grid(xlims, zlims, dx, dz)
# fnum = 1

eps = 1e-10
# A imagem não vai mostrar o que está colado no transdutor (de 0 a 5 mm), e também 
# não vai mostrar nada além de 55 mm de profundidade.
x = xp.arange(xlims[0], xlims[-1] + eps, dx)
print(f"X => {x.shape}")
z = xp.arange(zlims[1], zlims[0] + eps, dz)
print(f"Z => {z.shape}")
zz, xx = xp.meshgrid(z, x, indexing="ij")
print(f"ZZ => {zz.shape}")
print(f"XX => {xx.shape}")
yy = 0 * xx
grid = xp.stack((xx, yy, zz), axis=-1)
print(f"GRID => {grid.shape}")


qdata = xp.imag(hilbert_xp(idata, axis=-1))
print(f"IDATA => {idata.shape}")
print(f"QDATA => {qdata.shape}")
rf_complexo = (idata,qdata)
# print(f"RF complexo = > {rf_complexo}")

# Convert grid to tensor
grid = xp.reshape(grid, (-1, 3))
out_shape = grid.shape[:-1]
print(f"GRID RESHAPE => {grid.shape}")
print(f"out_shaoe=> {out_shape}")
 # o 0 e o 2 porque são as colunas x e z ...a y não interessa...o ultimo zero faz broadcasting

def calcula_distancia(grid, angles): #TX
    x_pixels = np.expand_dims(grid[:, 0], 0)
    z_pixels = np.expand_dims(grid[:, 2], 0)
    # print(f"AX => {x_pixels.shape}")
    # print(f"AZ => {z_pixels.shape}")
    distancia = x_pixels * xp.sin(angles) + z_pixels * xp.cos(angles)
    return distancia

def apod_plane(grid, angles, xlims):
    pix = xp.expand_dims(grid, 0)
    ang = xp.reshape(angles, (-1, 1, 1))
    # Project pixels back to aperture along the defined angles
    x_proj = pix[:, :, 0] - pix[:, :, 2] * np.tan(ang)
    # Select only pixels whose projection lie within the aperture, with fudge factor
    mask = (x_proj >= xlims[0] * 1.2) & (x_proj <= xlims[1] * 1.2)
    # Convert to float and normalize across angles (i.e., delay-and-"average")
    apod = xp.array(mask, dtype="float32")
    # Output has shape [nangles, npixels]
    return apod


 # Compute delays in meters
nangles = len(ang_list)
nelems = len(ele_list)
npixels = grid.shape[0]

# Initialize delays, apodizations, output array
txdel = xp.zeros((nangles, npixels), dtype="float")
rxdel = xp.zeros((nelems, npixels), dtype="float")

# o apodizado muda os 1 para o quando fora do feixe
txapo = xp.ones((nangles, npixels), dtype="float")
rxapo = xp.ones((nelems, npixels), dtype="float")


# Compute transmit and receive delays and apodizations
for i, tx in enumerate(ang_list):
    txdel[i] = calcula_distancia(grid, angles[tx])
    txdel[i] += tempo_zero[tx] * c
    txapo[i] = apod_plane(grid, angles[tx], xlims)
print(f"TX DEL => {txdel.shape}")
