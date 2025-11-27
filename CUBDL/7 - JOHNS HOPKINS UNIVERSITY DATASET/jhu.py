# ============================================================
# RECONSTRUÇÃO DE IMAGEM ULTRASSÔNICA B-MODE (DAS / PLANE WAVE)
#
# Este script foi reelaborado a partir de materiais e arquivos
# disponibilizados pelo CUBDL (Challenge on Ultrasound Beamforming),
# incluindo adaptações, reorganização das etapas, melhorias de
# legibilidade, tradução de comentários e integração com backend
# NumPy/CuPy para execução em CPU ou GPU.
#
# Uso educacional e de pesquisa.
# ============================================================

import h5py, numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
from pathlib import Path
# Ignorar avisos do cupyx
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# SCRIPT DE RECONSTRUÇÃO B-MODE (PLANE WAVE / DAS)
# ------------------------------------------------------------
# - Carrega um arquivo HDF5 da base CUBDL (MYOxxx.hdf5)
# - Configura backend NumPy/CuPy (CPU ou GPU)
# - Monta grade de pixels (x,z)
# - Calcula atrasos TX/RX + apodizações
# - Faz o beamforming somando I/Q em cada pixel
# - Gera imagem B-mode em dB e plota/salva
# ============================================================

# Limpar a tela do terminal (opcional, apenas estético)
os.system("clear")

flag= True # para ver somente uma vez a estrutura do dataset

# ============================================================
# BACKEND NUMPY/CUPY (CPU OU GPU)
# ============================================================
USE_CUDA = True         # se True, tenta usar GPU com CuPy
xp = np                 # por padrão, usamos NumPy
to_cpu = lambda a: a    
hilbert_xp = None       # será preenchido com hilbert da CPU ou GPU

if USE_CUDA:
    try:
        import cupy as cp
        from cupyx.scipy.signal import hilbert as c_hilbert
        if cp.cuda.runtime.getDeviceCount() > 0:
            # Pelo menos uma GPU disponível
            xp = cp
            hilbert_xp = c_hilbert
            to_cpu = cp.asnumpy
            dev = cp.cuda.Device()
            dev.use()
            print(f"[CUDA] Usando GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}".center(64))
            print("*" * 64)
        else:
            print("[CUDA] Nenhuma GPU detectada. Usando CPU (NumPy)".center(64))
            print("*" * 64)
    except Exception as e:
        print(f"[CUDA] CuPy indisponível ({e}). Usando CPU (NumPy).")
        print("*" * 64)

# Se não deu para usar CuPy, cai pro SciPy/NumPy na CPU
if hilbert_xp is None:
    from scipy.signal import hilbert as s_hilbert
    hilbert_xp = s_hilbert

# ============================================================
# PARÂMETROS FÍSICOS E CARREGAMENTO DOS DADOS
# ============================================================

# Lista de aquisições a processar.
# Aqui, imagens = [0, 1, ..., 10], permitindo reconstruir múltiplos
imagens= list(range(24, 35))

# Para cada número da lista, executa a reconstrução correspondente,
for acq in imagens:
    # Caminho do arquivo
    caminho = r"/home/users/lpaparella/ULTRASSOM/IMAGENS/2_Post_CUBDL_JHU_Breast_Data/"
    dataset = "JHU{:03d}".format(acq) + ".hdf5"
    path = caminho + dataset

    # ------------------------------------------------------------
    # Abertura do arquivo HDF5 e inspeção básica da estrutura
    # ------------------------------------------------------------
    arquivo = h5py.File(path, "r")

    def nome_e_tipo(nome, obj):
        """
        Função de callback para 'visititems' do h5py:
        - Imprime o nome de cada GRUPO e DATASET encontrado no arquivo.
        - Mostra também shape e tipo (dtype) dos datasets.
        """
        if isinstance(obj, h5py.Group):
            print(f"GRUPO    => {nome}")
        elif isinstance(obj, h5py.Dataset):
            print(f"DATASET  => {nome.center(25)} | {str(obj.shape).center(15)} | {obj.dtype}")

    if flag:
        print("EXPLORANDO O ARQUIVO HDF5".center(64))
        print("*" * 64)
        arquivo.visititems(nome_e_tipo)
        print("*" * 64)
        flag=False


    # Ajuste da velocidade do som de acordo com a aquisição
    idata = xp.array(arquivo["channel_data"], dtype="float32")
    qdata = xp.imag(hilbert_xp(idata, axis=-1))
    angles = xp.array(arquivo["angles"])
    fc = xp.array(arquivo["modulation_frequency"]).item()
    fs = xp.array(arquivo["sampling_frequency"]).item()
    c = xp.array(arquivo["sound_speed"]).item()
    time_zero = -1 * xp.array(arquivo["time_zero"], dtype="float32")
    fdemod = 0

    xpos = xp.array(arquivo["element_positions"], dtype="float32").T
    ele_pos = xp.stack([xpos, 0 * xpos, 0 * xpos], axis=1)

    xlims = xp.array([ele_pos[0, 0], ele_pos[-1, 0]])
    zlims = [3e-3, 30e-3]

    # Define pixel grid limits (assume y == 0)
    wvln = c / fc
    dx = wvln / 2.5
    dz = dx  # Use square pixels
    fnum = 1

    # OUTPUTS
    # grid    Pixel grid of size (nx, nz, 3)
    # eps vem de epsilon, Ele é usado para evitar erros de arredondamento nas funções arange.
    eps = 1e-10
    x = xp.arange(xlims[0], xlims[1] + eps, dx)
    z = xp.arange(zlims[0], zlims[1] + eps, dz)
    zz, xx = xp.meshgrid(z, x, indexing="ij")
    yy = 0 * xx
    grid = xp.stack((xx, yy, zz), axis=-1)

    # grid = xp.constant(grid, dtype=xp.float32)
    grid = xp.reshape(grid, (-1, 3))
    out_shape = grid.shape[:-1]

    ang_list = range(angles.shape[0])

    ele_list = range(ele_pos.shape[0])

    nangles = len(ang_list)
    nelems = len(ele_list)
    npixels = grid.shape[0]
    xlims_1 = (ele_pos[0, 0], ele_pos[-1, 0])  # Aperture width

    # Initialize delays, apodizations, output array
    txdel = xp.zeros((nangles, npixels), dtype="float32")
    rxdel = xp.zeros((nelems, npixels), dtype="float32")
    txapo = xp.ones((nangles, npixels), dtype="float32")
    rxapo = xp.ones((nelems, npixels), dtype="float32")
    

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

    # Convert to samples
    txdel *= fs / c
    rxdel *= fs / c

    # Initialize the output array
    idas = xp.zeros(npixels, dtype="float")
    qdas = xp.zeros(npixels, dtype="float")
    for t, td, ta in tqdm(zip(ang_list, txdel, txapo),
                        total=len(ang_list),
                        desc=f"Beamforming => IMAGEM {acq}"):
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
    bimg /= (bimg.max() + 1e-12)
    bimg_db = 20*np.log10(bimg + 1e-12)  # ou 10*log10(power)

    # extent (tudo em NumPy/CPU)
    x_cpu = to_np(x)
    z_cpu = to_np(z)
    xmin, xmax = float(x_cpu.min()), float(x_cpu.max())
    zmin, zmax = float(z_cpu.min()), float(z_cpu.max())

    # origin='upper' → use [xmin, xmax, zmax, zmin] para profundidade “pra baixo”
    extent = [xmin*1e3, xmax*1e3,zmax*1e3,zmin*1e3]

    plt.figure(figsize=(6, 8))
    plt.imshow(bimg_db, cmap="gray", origin="upper", aspect="equal", extent=extent)
    plt.title("Imagem B-mode")
    plt.xlabel("Lateral")
    plt.ylabel("Profundidade")
    plt.show()

    #salvar arquivo
    def save_fig(fig=None, nome_base="reconstrucao", pasta="IMAGENS", dpi=200):
        """Salva a figura atual em ./IMAGENS_SALVAS/<nome_base>[_N].png e imprime o caminho."""
        fig = fig or plt.gcf()
        Path(pasta).mkdir(parents=True, exist_ok=True)
        
        while True:
            sufixo = acq
            path = Path(pasta) / f"{nome_base}{sufixo}.png"
            if not path.exists():
                break
            
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"[OK] Figura salva em: {path}")


    # # >>> Salva em ./IMAGENS_SALVAS/reconstrucao.png (ou reconstrucao_1.png, etc.)
    # save_fig(nome_base="reconstrucao")