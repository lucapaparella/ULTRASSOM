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

print(f"IMAGEM ORIGINAL=> {arquivo["/beamformed_data"].shape}")

# P.angles (nangles,), radianos
angles = xp.array(arquivo["angles"])
print(f"ANGLES => {angles.shape}")

# P.ele_pos (nelems, 3), posições (x,y,z) dos elementos
ele_pos = xp.array(arquivo["element_positions"], dtype="float32")
ele_pos = xp.stack([ele_pos, xp.zeros_like(ele_pos), xp.zeros_like(ele_pos)], axis=1)
print(f"ELE_POS => {ele_pos.shape}")
# teste
# teste_xlims = np.zeros((3, 3), dtype="float32")
teste_xlims = [ele_pos[0, 0].item(), ele_pos[-1, 0].item()]
print(f"TESTE XLIMS => {teste_xlims}")

pitch = xp.array(arquivo["pitch"]).item()
print(f"pitch => {pitch}")
# P.fc, P.fs, P.fdemod, P.c, P.time_zero (arrays/escalares)
fc = xp.array(arquivo["modulation_frequency"]).item()
fs = xp.array(arquivo["sampling_frequency"]).item()
c =xp.array(arquivo["sound_speed"]).item()
fdemod = 1

# [Gravação começa] ----(ex.: espera 2 µs)---- [Pulso é emitido] ---- ecos retornam ---->
# o -1 faz que o tempo do pulso emitido seja 0
tempo_zero =-1*xp.array(arquivo["time_zero"], dtype="float32")
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
# ele_pos = xp.array(arquivo["element_positions"], dtype="float32")
xlims = [ele_pos[0, 0].item(), ele_pos[-1, 0].item()]
print(f"XLIMS =>{xlims}")
# "Eu quero que a imagem de ultrassom comece a 5 mm de profundidade e termine a 55 mm "
# "de profundidade."
# Profundidade Fisica em metros

# profundidade mínima por ângulo (amostra n=0)
z_start_per_angle = xp.maximum(0.0, -tempo_zero[0]) * (c/2)

# profundidade máxima por ângulo (amostra n=Ns-1)
z_end_per_angle = (((idata[2] - 1) / fs) - tempo_zero[0]) * (c/2)

# intervalo comum a todos os ângulos
zmin = float(xp.max(z_start_per_angle))
zmax = float(xp.min(z_end_per_angle))
print(f"ZMIN => {zmin}")
print(f"ZMAX => {zmax}")
# limites da profondidade
zlims = (zmin, zmax)
print("zlims (m) =", zlims)




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
# dx = wvln / 3 #largura do pixel
# o código está simplesmente garantindo que os pixels sejam quadrados
# dz = dx  # Use square pixels
# print(f"DZ => {dz}")
#-------------------------------------------------------
tamanho_pixel = xp.array(arquivo["pixel_d"]).item()
print(f"Tamanho pixel => {tamanho_pixel}")
dx = tamanho_pixel
dz = dx
# grid = make_pixel_grid(xlims, zlims, dx, dz)
fnum = 1

eps = 1e-10
# A imagem não vai mostrar o que está colado no transdutor (de 0 a 5 mm), e também 
# não vai mostrar nada além de 55 mm de profundidade.
x = xp.arange(xlims[0], xlims[-1] + eps, dx)
print(f"X => {x.shape}")
# z = xp.arange(zlims[0], zlims[1] + eps, dz)
z = xp.arange(zmax, zmin + eps, dz)
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
print(f"TX APO => {txapo.shape}")

def apod_focus(grid, ele_pos, fnum=1, min_width=1e-3):
    # Get vector between elements and pixels via broadcasting
    ppos = xp.expand_dims(grid, axis=0)
    epos = xp.reshape(ele_pos, (-1, 1, 3))

    v = ppos - epos
    # Select (ele,pix) pairs whose effective fnum is greater than fnum
    mask = xp.abs(v[:, :, 2] / (v[:, :, 0] + 1e-30)) > fnum
    mask = mask | (xp.abs(v[:, :, 0]) <= min_width)
    # Also account for edges of aperture
    mask = mask | ((v[:, :, 0] >= min_width) & (ppos[:, :, 0] <= epos[0, 0, 0]))
    mask = mask | ((v[:, :, 0] <= -min_width) & (ppos[:, :, 0] >= epos[-1, 0, 0]))
    # Convert to float and normalize across elements (i.e., delay-and-"average")
    apod = xp.array(mask, dtype="float32")
    # Output has shape [nelems, npixels]
    return apod

def delay_focus(grid, ele_pos):
    # Get norm of distance vector between elements and pixels via broadcasting
    dist = xp.linalg.norm(grid - xp.expand_dims(ele_pos, 0), axis=-1)
    return dist

for j, rx in enumerate(ele_list):
    rxdel[j] = delay_focus(grid, ele_pos[rx])
    rxapo[j] = apod_focus(grid, ele_pos[rx])

print(f"RX DEL => {rxdel.shape}")
print(f"RX APO => {rxapo.shape}")

 # Converter em amostras
txdel *= fs / c
rxdel *= fs / c

# Initialize the output array
idas = xp.zeros(npixels, dtype="float")
qdas = xp.zeros(npixels, dtype="float")
for t, td, ta in tqdm(zip(ang_list, txdel, txapo),
                      total=len(ang_list),
                      desc="Beamforming TX angles"):
    for r, rd, ra in zip(ele_list, rxdel, rxapo):
        # Dados IQ do disparo t e elemento r
        i_chan = idata[t, r]   # (Nsamples,)
        q_chan = qdata[t, r]

        # Soma dos atrasos (índices fracionários)
        delays = td + rd       # (Npixels,)

        # Interpolação linear 1D
        samples = xp.arange(i_chan.size)
        ifoc = xp.interp(delays, samples, i_chan, left=0.0, right=0.0)
        qfoc = xp.interp(delays, samples, q_chan, left=0.0, right=0.0)

        # Rotação de fase (caso demodulada)
        if fdemod != 0:
            tshift = delays / fs - grid[:, 2] * 2 / c
            theta = 2 * xp.pi * fdemod * tshift
            cos_t, sin_t = xp.cos(theta), xp.sin(theta)
            ifoc, qfoc = ifoc * cos_t - qfoc * sin_t, ifoc * sin_t + qfoc * cos_t

        # Apodização TX × RX e soma
        apods = ta * ra
        idas += ifoc * apods
        qdas += qfoc * apods

# helper para converter CuPy→NumPy (ou deixar como está se já for NumPy)
to_np = (lambda a: a.get()) if xp.__name__ == "cupy" else (lambda a: np.asarray(a))

# --- depois do loop de beamforming ---
idas = to_np(idas)
qdas = to_np(qdas)

# sinal complexo e imagem (em 2D!)
iq = idas + 1j * qdas

# você gerou x, z e meshgrid (zz, xx) antes:
nx = int(x.shape[0])      # número de colunas (lateral)
nz = int(z.shape[0])      # número de linhas (profundidade)

# iq veio de grid achatado -> reshape para (nz, nx)
bimg = np.abs(iq).reshape(nz, nx)


# log-compress  (uma vez só) + normalização
# bimg = bimg / (bimg.max() + 1e-12)
# bimg_db = 20 * np.log10(bimg + 1e-12)
# bimg_db = xp.clip(bimg_db, -60, 0)   # janela dinâmica opcional
# bimg_db = 10* np.log10(np.abs(bimg.T) / np.max(np.abs(bimg)))

bimg /= (bimg.max() + 1e-12)
bimg_db = 20*np.log10(bimg + 1e-12)  # ou 10*log10(power)


print(f"BIM DB => {bimg_db.shape}")
# extent (tudo em NumPy/CPU)
x_cpu = to_np(x)
z_cpu = to_np(z)
xmin, xmax = float(x_cpu.min()), float(x_cpu.max())
zmin, zmax = float(z_cpu.min()), float(z_cpu.max())

# origin='upper' → use [xmin, xmax, zmax, zmin] para profundidade “pra baixo”
extent = [xmin*1e3, xmax*1e3,zmin*1e3,zmax*1e3]

plt.figure(figsize=(6, 8))
plt.imshow(bimg_db, cmap="gray", origin="upper", aspect="auto", extent=extent)
plt.title("Imagem B-mode")
plt.xlabel("Lateral")
plt.ylabel("Profundidade")
plt.show()

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


# # >>> Salva em ./IMAGENS_SALVAS/reconstrucao.png (ou reconstrucao_1.png, etc.)
save_fig(nome_base="reconstrucao")



