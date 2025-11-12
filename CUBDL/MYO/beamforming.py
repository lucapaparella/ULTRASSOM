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
path = r"/home/users/lpaparella/ULTRASSOM/IMAGENS/1_CUBDL_Task1_Data/MYO005.hdf5"
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


idata = xp.array(arquivo["channel_data"], dtype="float32")
qdata = xp.imag(hilbert_xp(idata, axis=-1))
print(f"idata => {idata.shape}")
print(f"qdata => {qdata.shape}")

angles = xp.array(arquivo["angles"])
fc = xp.array(arquivo["modulation_frequency"]).item()

fs = xp.array(arquivo["sampling_frequency"]).item()
print(f"fs => {fs}")


c = xp.array(arquivo["sound_speed"]).item()
print(f"c => {c}")
# c = 1580.0
# Make the element positions based on L11-4v geometry
# o datasets não fornece as posições dos elementos
# eles precisame ser calculados usando um pitch padrão 
pitch = 0.3e-3
nelems = idata.shape[1]
print(f"nelems => {nelems}")
xpos = xp.arange(nelems) * pitch
xpos -= xp.mean(xpos)
ele_pos = xp.stack([xpos, 0 * xpos, 0 * xpos], axis=1)
print(f"ele_pos => {ele_pos.shape}")

fdemod = 0

# O time_zero não é fornecido, portanto precisa calcular
time_zero = xp.zeros((len(angles),), dtype="float32")

for i, a in enumerate(angles):
    time_zero[i] = -1*ele_pos[-1, 0] * np.abs(np.sin(a)) / c

print(f"time_zero => {time_zero.shape}")

# zlims = xp.array([0e-3, idata.shape[2] * c / fs / 2])
zlims = [5e-3, 55e-3]

print(f"zlims => {zlims}")
xlims = xp.array([ele_pos[0, 0], ele_pos[-1, 0]])
print(f"xlims => {xlims}")

# Define pixel grid limits (assume y == 0)
wvln = c / fc
dx = wvln / 2.5
print(f"dx => {dx}")
dz = dx  # Use square pixels
# grid = make_pixel_grid(xlims, zlims, dx, dz)
fnum = 1

eps = 1e-10
# eps vem de epsilon, Ele é usado para evitar erros de arredondamento nas funções arange.
x = xp.arange(xlims[0], xlims[1] + eps, dx)
z = xp.arange(zlims[0], zlims[1] + eps, dz)
zz, xx = xp.meshgrid(z, x, indexing="ij")
yy = 0 * xx
grid = xp.stack((xx, yy, zz), axis=-1)
print(f"grid => {grid.shape}")

# grid = xp.constant(grid, dtype=xp.float32)
grid = xp.reshape(grid, (-1, 3))
out_shape = grid.shape[:-1]

ang_list = range(angles.shape[0])
print(f"ang_list => {ang_list}")

ele_list = range(ele_pos.shape[0])
print(f"ele_list => {ele_list}")

nangles = len(ang_list)
print(f"nangles => {nangles}")
nelems = len(ele_list)
print(f"nelems => {nelems}")
npixels = grid.shape[0]
print(f"npixels => {npixels}")


# Initialize delays, apodizations, output array
txdel = xp.zeros((nangles, npixels), dtype="float32")
print(f"txdel => {txdel.shape}")
rxdel = xp.zeros((nelems, npixels), dtype="float32")
print(f"rxdel => {rxdel.shape}")
txapo = xp.ones((nangles, npixels), dtype="float32")
print(f"txapo => {txapo.shape}")
rxapo = xp.ones((nelems, npixels), dtype="float32")
print(f"rxapo => {rxapo.shape}")

#   grid    Pixel positions in x,y,z    [npixels, 3]
#   angles  Plane wave angles (radians) [nangles]
# OUTPUTS
#   dist    Distance from each pixel to each element [nelems, npixels]
def delay_plane(grid, angles):
    # Use broadcasting to simplify computations
    x = xp.expand_dims(grid[:, 0], 0)
    z = xp.expand_dims(grid[:, 2], 0)
    # For each element, compute distance to pixels
    dist = x * xp.sin(angles) + z * xp.cos(angles)
    return dist

#   grid    Pixel positions in x,y,z            [npixels, 3]
#   angles  Plane wave angles (radians)         [nangles]
#   xlims   Azimuthal limits of the aperture    [2]
# OUTPUTS
#   apod    Apodization for each angle to each element  [nangles, npixels]
def apod_plane(grid, angles, xlims):
    pix = xp.expand_dims(grid, 0)
    ang = xp.reshape(angles, (-1, 1, 1))
    # Project pixels back to aperture along the defined angles
    x_proj = pix[:, :, 0] - pix[:, :, 2] * xp.tan(ang)
    # Select only pixels whose projection lie within the aperture, with fudge factor
    mask = (x_proj >= xlims[0] * 1.2) & (x_proj <= xlims[1] * 1.2)
    # Convert to float and normalize across angles (i.e., delay-and-"average")
    apod = xp.array(mask, dtype="float32")
    # Output has shape [nangles, npixels]
    return apod

for i, tx in enumerate(ang_list):
    txdel[i] = delay_plane(grid, angles[tx])
    txdel[i] += time_zero[tx] * c
    txapo[i] = apod_plane(grid, angles[tx], xlims)

print(f"txdel => {txdel.shape}")
print(f"txapo => {txapo.shape}")

#   grid    Pixel positions in x,y,z    [npixels, 3]
#   ele_pos Element positions in x,y,z  [nelems, 3]
# OUTPUTS
#   dist    Distance from each pixel to each element [nelems, npixels]
def delay_focus(grid, ele_pos):
    # Get norm of distance vector between elements and pixels via broadcasting
    dist = xp.linalg.norm(grid - xp.expand_dims(ele_pos, 0), axis=-1)
    return dist

#   grid        Pixel positions in x,y,z        [npixels, 3]
#   ele_pos     Element positions in x,y,z      [nelems, 3]
#   fnum        Desired f-number                scalar
#   min_width   Minimum width to retain         scalar
# OUTPUTS
#   apod    Apodization for each pixel to each element  [nelems, npixels]
def apod_focus(grid, ele_pos, fnum=1, min_width=1e-3):
    # Get vector between elements and pixels via broadcasting
    ppos = xp.expand_dims(grid, 0)
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



for j, rx in enumerate(ele_list):
    rxdel[j] = delay_focus(grid, ele_pos[rx])
    rxapo[j] = apod_focus(grid, ele_pos[rx])

print(f"rxdel => {rxdel.shape}")
print(f"txapo => {txapo.shape}")

# Convert to samples
txdel *= fs / c
rxdel *= fs / c

# Make data torch tensors
# iqdata = (idata, qdata)

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

        # FDEMOD = 0 NÃO PRECISA DESSE IF
        # # Rotação de fase (caso demodulada)
        # if fdemod != 0:
        #     tshift = delays / fs - grid[:, 2] * 2 / c
        #     theta = 2 * xp.pi * fdemod * tshift
        #     cos_t, sin_t = xp.cos(theta), xp.sin(theta)
        #     ifoc, qfoc = ifoc * cos_t - qfoc * sin_t, ifoc * sin_t + qfoc * cos_t

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
bimg_db = np.clip(bimg_db, -50, 0)

print(f"BIM DB => {bimg_db.shape}")

#------------------------------------------------------------------------

nx, nz = len(x), len(z)
Lx = float(x.max()-x.min())   # m
Lz = float(z.max()-z.min())   # m
print(f"nx={nx}, nz={nz},  Lx={Lx*1e3:.1f} mm, Lz={Lz*1e3:.1f} mm,  aspect_phys={Lx/Lz:.3f}")

assert bimg.shape == (nz, nx), f"bimg {bimg.shape} != (nz,nx)=({nz},{nx})"
#------------------------------------------------------------------------

# extent (tudo em NumPy/CPU)
x_cpu = to_np(x)
z_cpu = to_np(z)
xmin, xmax = float(x_cpu.min()), float(x_cpu.max())
zmin, zmax = float(z_cpu.min()), float(z_cpu.max())

# origin='upper' → use [xmin, xmax, zmax, zmin] para profundidade “pra baixo”
extent = [xmin*1e3, xmax*1e3,zmax*1e3,zmin*1e3]

plt.figure(figsize=(6, 8))
plt.imshow(bimg_db, cmap="gray", origin="upper", aspect="auto", extent=extent)
plt.title("Imagem B-mode")
plt.xlabel("Lateral")
plt.ylabel("Profundidade")
plt.colorbar(label='dB')
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
