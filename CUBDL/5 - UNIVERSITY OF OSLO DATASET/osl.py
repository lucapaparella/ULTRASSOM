import h5py, numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
from pathlib import Path


#limpar a tela
os.system("clear")
imagens= [10]
# Caminho do arquivo
for acq in imagens:
    caminho = r"/home/users/lpaparella/ULTRASSOM/IMAGENS/1_CUBDL_Task1_Data/OSL010/"
    # Make sure the selected dataset is valid
    dataset = "OSL{:03d}".format(acq) + ".hdf5"
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
    convolve_xp = None

    if USE_CUDA:
        try:
            import cupy as cp
            from cupyx.scipy.signal import hilbert as c_hilbert
            from cupyx.scipy.ndimage import convolve as c_convolve
            
            if cp.cuda.runtime.getDeviceCount() > 0:
                xp = cp

                hilbert_xp = c_hilbert
                convolve_xp = c_convolve

                to_cpu = cp.asnumpy
                dev = cp.cuda.Device()
                dev.use()
                print(f"[CUDA] Usando GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
            else:
                print("[CUDA] Nenhuma GPU detectada. Usando CPU (NumPy).")
        except Exception as e:
            print(f"[CUDA] CuPy indisponível => Usando CPU (NumPy).")

    if hilbert_xp is None:
        # fallback para CPU
        from scipy.signal import hilbert as s_hilbert
        hilbert_xp = s_hilbert

    if convolve_xp is None:
        from scipy.ndimage import convolve1d as s_convolve
        convolve_xp = s_convolve

    # --------- FIM Backend: CUDA (CuPy) se disponível; caso contrário NumPy ----------
    

    # Phantom-specific parameters
    if acq == 2:
        sound_speed = 1536
    elif acq == 3:
        sound_speed = 1543
    elif acq == 4:
        sound_speed = 1538
    elif acq == 5:
        sound_speed = 1539
    elif acq == 6:
        sound_speed = 1541
    elif acq == 7:
        sound_speed = 1540
    else:
        sound_speed = 1540

    # Get data
    idata = xp.array(arquivo["channel_data"], dtype="float32")
    qdata = xp.imag(hilbert_xp(idata, axis=-1))
    angles = xp.array(arquivo["transmit_direction"][0], dtype="float32")
    fc = xp.array(arquivo["modulation_frequency"]).item()
    fs = xp.array(arquivo["sampling_frequency"]).item()
    c = sound_speed  # xp.array(arquivo["sound_speed"]).item()
    time_zero = -1 * xp.array(arquivo["start_time"], dtype="float32")[0]
    fdemod = 0
    ele_pos = xp.array(arquivo["element_positions"], dtype="float32").T
    ele_pos[:, 0] -= xp.mean(ele_pos[:, 0])

    xlims = [ele_pos[0, 0], ele_pos[-1, 0]]
    zlims = [10e-3, 65e-3]
    if acq == 10:
        zlims = [5e-3, 50e-3]


 # Define pixel grid limits (assume y == 0)
    wvln = c / fc
    dx = wvln / 4
    dz = dx  # Use square pixels
    eps = 1e-10
    x = xp.arange(xlims[0], xlims[1] + eps, dx)
    z = xp.arange(zlims[0], zlims[1] + eps, dz)
    zz, xx = xp.meshgrid(z, x, indexing="ij")
    yy = 0 * xx
    grid = xp.stack((xx, yy, zz), axis=-1)
    fnum = 1

    ang_list = range(angles.shape[0])
    # print(f"ang_list => {ang_list}")
    ele_list = range(ele_pos.shape[0])
    # print(f"ele_list => {ele_list}")


    out_shape = grid.shape[:-1]
    # print(f"out_shape => {out_shape}")

    # grid = xp.constant(grid, dtype=xp.float32)
    grid = xp.reshape(grid, (-1, 3))
    print(f"grid  => {grid}")

    nangles = len(ang_list)
    # print(f"nangles => {nangles}")
    nelems = len(ele_list)
    # print(f"nelems => {nelems}")
    npixels = grid.shape[0]
    # print(f"npixels => {npixels}")


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
        txdel[i] = delay_plane(grid, angles[[tx]])
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

    def apply_delays(iq, d):
        """Aplica atrasos no tempo usando interpolação linear (NumPy/CuPy)."""
        # iq: (Nbatch, Nsamples, 2)
        # d:  (Nbatch, Npix) ou (Nbatch, Npix, 1)  -> índices fracionários

        # Garante que d tenha shape (Nbatch, Npix) removendo eixo extra se existir
        if d.ndim == 3 and d.shape[-1] == 1:
            d = d[..., 0]   # (Nbatch, Npix)

        # Índices inferiores e superiores (parte inteira do atraso)
        d0 = xp.floor(d).astype(xp.int32)   # (Nbatch, Npix)
        d1 = d0 + 1                         # (Nbatch, Npix)

        # ----- Equivalente ao tf.gather_nd(..., batch_dims=1) -----
        # iq  → (Nbatch, Nsamples, 2)
        # d0  → (Nbatch, Npix)   (índices na dimensão das amostras)
        # Resultado → (Nbatch, Npix, 2)

        # Índice do batch com mesmo primeiro eixo de d0
        b = xp.arange(iq.shape[0])[:, None]  # (Nbatch, 1)

        # Para cada batch n e pixel p:
        #   iq0[n, p, :] = iq[n, d0[n, p], :]
        iq0 = iq[b, d0]   # (Nbatch, Npix, 2)
        iq1 = iq[b, d1]   # (Nbatch, Npix, 2)
        # -----------------------------------------------------------

        # Converte índices para float
        d0f = d0.astype(xp.float32)         # (Nbatch, Npix)
        d1f = d1.astype(xp.float32)         # (Nbatch, Npix)
        df  = d.astype(xp.float32)          # (Nbatch, Npix)

        # Monta pesos com eixo extra na última dimensão para casar com (Nbatch, Npix, 2)
        w0 = (d1f - df)[..., None]          # (Nbatch, Npix, 1)
        w1 = (df - d0f)[..., None]          # (Nbatch, Npix, 1)
        # print(f"w0 => {w0.shape}")
        # Interpolação linear: out = w0 * iq0 + w1 * iq1
        out = w0 * iq0 + w1 * iq1           # (Nbatch, Npix, 2)
        # print(f"out => {out.shape}")
        # Separa os canais I e Q (última dimensão: 0 = I, 1 = Q)
        ifoc = out[:, :, 0]
        qfoc = out[:, :, 1]
        # print(f"ifoc => {ifoc.shape}")
        # print(f"qfoc => {qfoc.shape}")
        return ifoc, qfoc

    def _complex_rotate(I, Q, theta):
        Ir = I * xp.cos(theta) - Q * xp.sin(theta)
        Qr = Q * xp.cos(theta) + I * xp.sin(theta)
        return Ir, Qr




    iqdata = xp.stack((idata, qdata), axis=-1)
    # print(f"iqdata => {iqdata.shape}")

    # Initialize the output array
    idas = xp.zeros(npixels, dtype="float")
    qdas = xp.zeros(npixels, dtype="float")

    #___________________________________________________________

    # só para usar como default; com CuPy passe xp=cp

    def grid_sample_xp(input, grid, align_corners=False):
        """
        Aproxima torch.nn.functional.grid_sample (modo bilinear, padding zeros).

        Espera:
            input: (N, C, H, W)
                N = batch (ex: ângulos)
                C = canais (ex: elementos, I/Q, etc.)
                H = altura  (ex: amostras no tempo)
                W = largura (ex: colunas)

            grid:  (N, H_out, W_out, 2), com:
                grid[..., 0] = x normalizado em [-1,1]  -> eixo W (colunas)
                grid[..., 1] = y normalizado em [-1,1]  -> eixo H (linhas)

        Retorna:
            out: (N, C, H_out, W_out)
        """
        # Shapes de entrada
        N, C, H, W = input.shape
        Ng, H_out, W_out, two = grid.shape
        assert Ng == N,  "grid e input devem ter o mesmo N"
        assert two == 2, "última dimensão de grid deve ser 2 (x,y)"

        # Separa coordenadas normalizadas
        x = grid[..., 0]  # (N, H_out, W_out)
        y = grid[..., 1]  # (N, H_out, W_out)

        # Converte de coordenadas normalizadas [-1,1] para índice de pixel [0, W-1] e [0, H-1]
        if align_corners:
            ix = (x + 1) * (W - 1) / 2.0
            iy = (y + 1) * (H - 1) / 2.0
        else:
            ix = (x + 1) * W / 2.0 - 0.5
            iy = (y + 1) * H / 2.0 - 0.5

        # Índices inteiros dos 4 vizinhos
        ix0 = xp.floor(ix).astype(xp.int64)
        iy0 = xp.floor(iy).astype(xp.int64)
        ix1 = ix0 + 1
        iy1 = iy0 + 1

        # Máscaras de pontos dentro da imagem (para padding zero)
        inside_x0 = (ix0 >= 0) & (ix0 < W)
        inside_x1 = (ix1 >= 0) & (ix1 < W)
        inside_y0 = (iy0 >= 0) & (iy0 < H)
        inside_y1 = (iy1 >= 0) & (iy1 < H)

        # Clampa índices para garantir que estão em [0, W-1] e [0, H-1]
        ix0_cl = xp.clip(ix0, 0, W - 1)
        ix1_cl = xp.clip(ix1, 0, W - 1)
        iy0_cl = xp.clip(iy0, 0, H - 1)
        iy1_cl = xp.clip(iy1, 0, H - 1)

        # # Distâncias fracionárias para pesos bilineares
        # wx1 = ix - ix0.astype(ix.dtype)  # distância até x1
        # wx0 = 1.0 - wx1                  # distância até x0
        # wy1 = iy - iy0.astype(iy.dtype)  # distância até y1
        # wy0 = 1.0 - wy1                  # distância até y0

        # Distâncias fracionárias para pesos bilineares
        # (convertemos os inteiros de volta pra float para subtrair de ix/iy)
        ix0_f = ix0.astype(ix.dtype)
        iy0_f = iy0.astype(iy.dtype)

        wx1 = ix - ix0_f          # distância até x1
        wx0 = 1.0 - wx1           # distância até x0
        wy1 = iy - iy0_f          # distância até y1
        wy0 = 1.0 - wy1           # distância até y0

        # Pesos 2D (N, H_out, W_out)
        wa = wx0 * wy0  # (x0, y0)
        wb = wx1 * wy0  # (x1, y0)
        wc = wx0 * wy1  # (x0, y1)
        wd = wx1 * wy1  # (x1, y1)

        # Zera pesos onde o índice sai da imagem (padding zeros)
        wa *= inside_x0 & inside_y0
        wb *= inside_x1 & inside_y0
        wc *= inside_x0 & inside_y1
        wd *= inside_x1 & inside_y1

        # Índices de batch: (N,1,1) para broadcast
        n_idx = xp.arange(N, dtype=xp.int64)[:, None, None]

        # Pega os 4 vizinhos: resultado (N, C, H_out, W_out)
        Ia = input[n_idx, :, iy0_cl, ix0_cl]  # (N,C,H_out,W_out)
        Ib = input[n_idx, :, iy0_cl, ix1_cl]
        Ic = input[n_idx, :, iy1_cl, ix0_cl]
        Id = input[n_idx, :, iy1_cl, ix1_cl]

        Ia = xp.transpose(Ia, (0, 3, 1, 2))  # -> (N, C, H_out, W_out)
        Ib = xp.transpose(Ib, (0, 3, 1, 2))
        Ic = xp.transpose(Ic, (0, 3, 1, 2))
        Id = xp.transpose(Id, (0, 3, 1, 2))

        # Adapta pesos para broadcast com C
        wa = wa[:, None, :, :]  # (N,1,H_out,W_out)
        wb = wb[:, None, :, :]
        wc = wc[:, None, :, :]
        wd = wd[:, None, :, :]

        # Combinação bilinear final
        out = wa * Ia + wb * Ib + wc * Ic + wd * Id  # (N, C, H_out, W_out)
        return out

    #_______________________________________________________________________

            
            # Loop over angles and elements
    for t, td, ta in tqdm(zip(ang_list, txdel, txapo),
                        total=len(ang_list),
                        desc="Beamforming TX angles"):
        # for t, td, ta in tqdm(zip(ang_list, txdel, txapo), total=nangles):
        for r, rd, ra in zip(ele_list, rxdel, rxapo):
            # Grab data from t-th Tx, r-th Rx
            # iq: (1, 2, 1, nsamps)
            iq = xp.stack([idata[t, r], qdata[t, r]], axis=0).reshape(1, 2, 1, -1)
            # print(f"iq  => {iq.shape}")
            delays = td + rd
            # print(f"delays  => {delays.shape}")
            # dgs: normaliza delays (tempo) para [-1,1] e coloca como eixo x
            dgs_time = ((delays.reshape(1, 1, -1, 1) * 2 + 1) / idata.shape[-1]) - 1  # (1,1,npix,1)
            # print(f"dgs_time  => {dgs_time.shape}")
            # segundo eixo (y) = 0, porque H=1 → efeito 1D no tempo
            dgs = xp.concatenate([dgs_time, 0 * dgs_time], axis=-1)  # (1,1,npix,2)

            # aplica nosso grid_sample_xp
            out = grid_sample_xp(iq, dgs, align_corners=False)  # (1,2,1,npix)
            # reshape para ficar igual ao PyTorch .view(2, -1)
            out = out.reshape(2, -1)
            ifoc = out[0]
            qfoc = out[1]
            # Apply phase-rotation if focusing demodulated data
            if fdemod != 0:
                tshift = delays.reshape(-1) / fs - grid[:, 2] * 2 / c
                theta = 2 * xp.pi * fdemod * tshift
                ifoc, qfoc = _complex_rotate(ifoc, qfoc, theta)
            # Apply apodization, reshape, and add to running sum
            apods = ta * ra
            idas += ifoc * apods
            qdas += qfoc * apods

    # Finally, restore the original pixel grid shape and convert to numpy array
    idas = idas.reshape(out_shape)
    qdas = qdas.reshape(out_shape)
    
    # idas = xp.reshape(idas, out_shape)
    # qdas = xp.reshape(qdas, out_shape)
    print(f"idas => {idas.shape}")

    print(f"qdas => {qdas.shape}")

    # # helper para converter CuPy→NumPy (ou deixar como está se já for NumPy)
    to_np = (lambda a: a.get()) if xp.__name__ == "cupy" else (lambda a: np.asarray(a))

    # # --- depois do loop de beamforming ---
    # idas = to_np(idas)
    # qdas = to_np(qdas)
    # # sinal complexo e imagem (em 2D!)
    iq = idas + 1j * qdas
    iq = to_np(iq)
    print(f"iq => {iq.shape}")
    # # você gerou x, z e meshgrid (zz, xx) antes:
    # nx = int(x.shape[0])      # número de colunas (lateral)
    # nz = int(z.shape[0])      # número de linhas (profundidade)

    # # iq veio de grid achatado -> reshape para (nz, nx)
    bimg_db = 20 * np.log10(np.abs(iq))  # Log-compress
    bimg_db -= np.amax(bimg_db)

    # bimg_db = np.abs(iq)
    # bimg_db /= np.amax(bimg_db)

    # Calcula a magnitude (amplitude)

    # bimg = np.abs(iq)

    # # Normaliza e converte para dB
    # bimg /= (bimg.max() + 1e-12)
    # bimg_db = 20 * np.log10(bimg + 1e-12)

    # # Limita a faixa dinâmica (por exemplo, -60 dB)
    bimg_db = np.clip(bimg_db, -60, 0)
    print(f"BIM DB => {bimg_db.shape}")

    #------------------------------------------------------------------------

    # nx, nz = len(x), len(z)
    # Lx = float(x.max()-x.min())   # m
    # Lz = float(z.max()-z.min())   # m
    # print(f"nx={nx}, nz={nz},  Lx={Lx*1e3:.1f} mm, Lz={Lz*1e3:.1f} mm,  aspect_phys={Lx/Lz:.3f}")

    # assert bimg.shape == (nz, nx), f"bimg {bimg.shape} != (nz,nx)=({nz},{nx})"
    # #------------------------------------------------------------------------

    # extent (tudo em NumPy/CPU)
    x_cpu = to_np(x)
    z_cpu = to_np(z)
    xmin, xmax = float(x_cpu.min()), float(x_cpu.max())
    zmin, zmax = float(z_cpu.min()), float(z_cpu.max())

    # origin='upper' → use [xmin, xmax, zmax, zmin] para profundidade “pra baixo”
    extent = [xmin*1e3, xmax*1e3,zmax*1e3,zmin*1e3]
    plt.figure()
    plt.imshow(bimg_db, cmap="gray", origin="upper", extent=extent)
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
        
        while True:
            sufixo = acq
            path = Path(pasta) / f"{nome_base}{sufixo}.png"
            if not path.exists():
                break
            
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"[OK] Figura salva em: {path}")


    # # >>> Salva em ./IMAGENS_SALVAS/reconstrucao.png (ou reconstrucao_1.png, etc.)
    save_fig(nome_base="reconstrucao")


