import h5py
import numpy as np
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
flag= True

imagens= list(range(24, 35))

for acq in imagens:
    # Caminho do arquivo
    caminho = r"/home/users/lpaparella/ULTRASSOM/IMAGENS/2_Post_CUBDL_JHU_Breast_Data/"
    # Make sure the selected dataset is valid
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

    # ============================================================
    # BACKEND NUMPY/CUPY (CPU OU GPU)
    # ============================================================
    USE_CUDA = True     # se True, tenta usar GPU com CuPy
    xp = np             # por padrão, usamos NumPy
    to_cpu = lambda a: a
    hilbert_xp = None   # será preenchido com hilbert da CPU ou GPU

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
                print(f"[CUDA] Usando GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
            else:
                print("[CUDA] Nenhuma GPU detectada. Usando CPU (NumPy).")
        except Exception as e:
            print(f"[CUDA] CuPy indisponível ({e}). Usando CPU (NumPy).")

    # Se não deu para usar CuPy, cai pro SciPy/NumPy na CPU
    if hilbert_xp is None:
        from scipy.signal import hilbert as s_hilbert
        hilbert_xp = s_hilbert

    # ============================================================
    # PARÂMETROS FÍSICOS E CARREGAMENTO DOS DADOS
    # ============================================================

    # Dados de canal (I) e geração de Q via transformada de Hilbert
    idata = xp.array(arquivo["channel_data"], dtype="float32")  # dados de RF (canal x tempo)
    qdata = xp.imag(hilbert_xp(idata, axis=-1))                 # componente em quadratura (Q)

    # Ângulos dos planos de onda (em radianos)
    angles = xp.array(arquivo["angles"])

    # Frequência de modulação (fc) e amostragem (fs)
    fc = xp.array(arquivo["modulation_frequency"]).item()
    fs = xp.array(arquivo["sampling_frequency"]).item()

    # Velocidade do som (m/s)
    c = xp.array(f["sound_speed"]).item()

    # Vetor de time-zero (um valor por ângulo)
    time_zero = xp.zeros((len(angles),), dtype="float32")

    # Frequência de demodulação (caso seja usada rotação de fase posterior)
    fdemod = 0

    # ============================================================
    # GEOMETRIA DO TRANSDUTOR (L11-4v) E TIME ZERO
    # ============================================================

    pitch = 0.3e-3                     # passo entre elementos (m)
    nelems = idata.shape[1]            # número de elementos no array
    xpos = xp.arange(nelems) * pitch   # posição x de cada elemento
    xpos -= xp.mean(xpos)              # centraliza em x=0

    # Posições dos elementos no espaço (x,y,z)
    ele_pos = xp.stack([xpos, 0 * xpos, 0 * xpos], axis=1)  # shape: (nelems, 3)

    # Para este dataset, o time-zero é o ponto central da abertura
    for i, a in enumerate(angles):
        # componente de atraso em função do ângulo (aprox. geometria)
        time_zero[i] = ele_pos[-1, 0] * xp.abs(xp.sin(a)) / c

    # Limites laterais (x) da abertura física
    xlims = [ele_pos[0, 0], ele_pos[-1, 0]]

    # Limites em profundidade (z) em metros
    zlims = [8e-3, 55e-3]

    # ============================================================
    # GRADE DE PIXELS (GRADE CARTESIANA EM X-Z)
    # ============================================================

    wvln = c / fc      # comprimento de onda
    dx = wvln / 4      # passo lateral (pixel) ~ λ/4
    dz = dx            # passo em profundidade igual a dx → pixels aproximadamente quadrados

    # eps vem de "epsilon": um valor bem pequeno usado para evitar problemas de arredondamento
    # nas funções arange (por exemplo, garantir que o limite superior seja incluído mesmo com
    # erros numéricos de ponto flutuante).
    eps = 1e-10

    # Vetores de coordenadas em x e z
    x = xp.arange(xlims[0], xlims[1] + eps, dx)
    z = xp.arange(zlims[0], zlims[1] + eps, dz)

    # Gera malha (zz, xx). indexing="ij" → 1º eixo = z (linhas), 2º eixo = x (colunas)
    zz, xx = xp.meshgrid(z, x, indexing="ij")
    yy = 0 * xx  # assumimos variação apenas em x-z, y=0
    grid = xp.stack((xx, yy, zz), axis=-1)   # shape: (nz, nx, 3)

    fnum = 1  # f-number desejado para apodização focada

    ang_list = range(angles.shape[0])   # índices de ângulos TX
    ele_list = range(ele_pos.shape[0])  # índices de elementos RX

    # Mantemos out_shape = (nz, nx) para reorganizar a imagem depois
    out_shape = grid.shape[:-1]

    # Achata grade em lista de pixels: [npixels, 3]
    grid = xp.reshape(grid, (-1, 3))

    # Quantidade de disparos (planos de onda) em transmissão
    nangles = len(ang_list)
    # Quantidade de elementos do transdutor em recepção
    nelems = len(ele_list)
    # Quantidade total de pixels na grade (grid achatado: npixels = nz * nx)
    npixels = grid.shape[0]

    # ============================================================
    # INICIALIZAÇÃO DE ATRASOS E APODIZAÇÕES
    # ============================================================

    # Atrasos (em unidades de tempo, depois convertidos para amostras)
    txdel = xp.zeros((nangles, npixels), dtype="float32")  # atrasos TX por ângulo
    rxdel = xp.zeros((nelems, npixels), dtype="float32")   # atrasos RX por elemento

    # Apodizações (janelas) TX e RX
    txapo = xp.ones((nangles, npixels), dtype="float32")
    rxapo = xp.ones((nelems, npixels), dtype="float32")

    # ------------------------------------------------------------
    # Funções auxiliares para atrasos/apodizações em onda plana
    # ------------------------------------------------------------
    def delay_plane(grid, angles):
        """
        Calcula atraso de transmissão (distância projetada) para onda plana.

        Parâmetros:
            grid   : posições dos pixels [npixels, 3]
            angles : ângulo(s) do plano de onda (rad) [nangles] ou [1]

        Retorna:
            dist   : distância "efetiva" ao longo da direção do plano de onda
                    shape: [nangles, npixels]
        """
        x = xp.expand_dims(grid[:, 0], 0)  # (1, npixels)
        z = xp.expand_dims(grid[:, 2], 0)  # (1, npixels)
        dist = x * xp.sin(angles) + z * xp.cos(angles)
        return dist

    def apod_plane(grid, angles, xlims):
        """
        Calcula apodização em transmissão para onda plana.

        - Projeta cada pixel de volta na abertura física ao longo do ângulo.
        - Mantém apenas pixels cuja projeção cai dentro [xlims[0]*1.2, xlims[1]*1.2].

        Parâmetros:
            grid   : posições dos pixels [npixels, 3]
            angles : ângulos dos disparos (rad) [nangles]
            xlims  : limites laterais da abertura [2]

        Retorna:
            apod   : máscara (0/1) por ângulo e pixel [nangles, npixels]
        """
        pix = xp.expand_dims(grid, 0)           # (1, npixels, 3)
        ang = xp.reshape(angles, (-1, 1, 1))    # (nangles,1,1)

        # Projeção da posição do pixel na abertura, ao longo do ângulo
        x_proj = pix[:, :, 0] - pix[:, :, 2] * xp.tan(ang)

        # Seleciona pixels cuja projeção cai dentro da abertura (com fator de "folga")
        mask = (x_proj >= xlims[0] * 1.2) & (x_proj <= xlims[1] * 1.2)

        apod = xp.array(mask, dtype="float32")
        return apod  # [nangles, npixels]

    # Preenche atrasos e apodização de TX para cada ângulo
    for i, tx in enumerate(ang_list):
        # atraso geométrico da onda plana
        txdel[i] = delay_plane(grid, angles[[tx]])
        # correção de time-zero
        txdel[i] += time_zero[tx] * c
        # apodização de TX
        txapo[i] = apod_plane(grid, angles[tx], xlims)

    # ------------------------------------------------------------
    # Funções auxiliares para atrasos/apodizações em foco (RX)
    # ------------------------------------------------------------
    def delay_focus(grid, ele_pos):
        """
        Calcula a distância entre cada pixel e um elemento específico.

        Parâmetros:
            grid    : posições dos pixels [npixels, 3]
            ele_pos : posição de UM elemento [3] ou shape (3,)

        Retorna:
            dist    : distância pixel-elemento [npixels]
        """
        # xp.expand_dims(ele_pos, 0) → shape (1,3)
        # grid - ele_pos → vetor posição entre pixel e elemento
        dist = xp.linalg.norm(grid - xp.expand_dims(ele_pos, 0), axis=-1)
        return dist

    def apod_focus(grid, ele_pos, fnum=1, min_width=1e-3):
        """
        Calcula apodização de recepção (focada) para um elemento.

        Ideia:
        - Calcula vetor entre elemento e pixel (v = ppos - epos).
        - Impõe critério de f-number: |z/x| > fnum para manter contribuições válidas.
        - Garante largura mínima de abertura (min_width) e trata bordas.

        Parâmetros:
            grid      : posições dos pixels [npixels, 3]
            ele_pos   : posições de TODOS elementos [nelems, 3]
            fnum      : f-number alvo (controla abertura efetiva)
            min_width : largura mínima para manter contribuição

        Retorna:
            apod      : máscara 0/1 [nelems, npixels]
        """
        ppos = xp.expand_dims(grid, 0)          # (1, npixels, 3)
        epos = xp.reshape(ele_pos, (-1, 1, 3))  # (nelems, 1, 3)
        v = ppos - epos                         # (nelems, npixels, 3)

        # Critério de f-number: |z/x| > fnum  (evita raios muito inclinados)
        mask = xp.abs(v[:, :, 2] / (v[:, :, 0] + 1e-30)) > fnum

        # Mantém também pontos próximos em x (largura mínima da abertura)
        mask = mask | (xp.abs(v[:, :, 0]) <= min_width)

        # Tratamento de bordas da abertura (evita "cortar" demais nos extremos)
        mask = mask | ((v[:, :, 0] >= min_width) & (ppos[:, :, 0] <= epos[0, 0, 0]))
        mask = mask | ((v[:, :, 0] <= -min_width) & (ppos[:, :, 0] >= epos[-1, 0, 0]))

        apod = xp.array(mask, dtype="float32")
        return apod  # [nelems, npixels]

    # Preenche atrasos e apodização de RX para cada elemento
    for j, rx in enumerate(ele_list):
        rxdel[j] = delay_focus(grid, ele_pos[rx])
        rxapo[j] = apod_focus(grid, ele_pos[rx])

    # ------------------------------------------------------------
    # Conversão de atrasos de distância (m) para amostras
    # ------------------------------------------------------------
    txdel *= fs / c
    rxdel *= fs / c

    # ============================================================
    # FUNÇÃO DE APLICAÇÃO DE ATRASOS (INTERPOLAÇÃO LINEAR EM IQ)
    # ============================================================
    def apply_delays(iq, d):
        """
        Aplica atrasos de tempo usando interpolação linear (NumPy/CuPy).

        Parâmetros:
            iq : tensores IQ, shape (Nbatch, Nsamples, 2)
                - Nbatch: batch (ex.: ângulos ou combinações TX/RX)
                - Nsamples: amostras no tempo
                - canal 0: I, canal 1: Q
            d  : atrasos em índices fracionários, shape (Nbatch, Npix) ou (Nbatch, Npix, 1)

        Retorna:
            ifoc, qfoc : componentes I e Q focadas, shape (Nbatch, Npix)
        """
        # Garante que d tenha shape (Nbatch, Npix)
        if d.ndim == 3 and d.shape[-1] == 1:
            d = d[..., 0]   # (Nbatch, Npix)

        # Índices inteiros inferior (d0) e superior (d1)
        d0 = xp.floor(d).astype(xp.int32)   # (Nbatch, Npix)
        d1 = d0 + 1                         # (Nbatch, Npix)

        # Índices de batch (para simular tf.gather_nd com batch_dims=1)
        b = xp.arange(iq.shape[0])[:, None]  # (Nbatch, 1)

        # iq0[n, p, :] = iq[n, d0[n, p], :]
        iq0 = iq[b, d0]   # (Nbatch, Npix, 2)
        iq1 = iq[b, d1]   # (Nbatch, Npix, 2)

        # Converte atrasos para float
        d0f = d0.astype(xp.float32)
        d1f = d1.astype(xp.float32)
        df  = d.astype(xp.float32)

        # Pesos da interpolação linear (eixo extra para casar com canal 2)
        w0 = (d1f - df)[..., None]          # (Nbatch, Npix, 1)
        w1 = (df - d0f)[..., None]          # (Nbatch, Npix, 1)

        # Interpolação linear entre iq0 e iq1
        out = w0 * iq0 + w1 * iq1           # (Nbatch, Npix, 2)

        # Separa I e Q
        ifoc = out[:, :, 0]
        qfoc = out[:, :, 1]

        return ifoc, qfoc

    # ============================================================
    # FUNÇÃO DE AMOSTRAGEM BILINEAR EM UM MAPA DE COORDENADAS
    # ============================================================
    def grid_sample_xp(input, grid, align_corners=False):
        """
        Realiza amostragem bilinear (interp. 2D) em um mapa de coordenadas normalizadas.

        A função recebe:
            input: tensor de entrada no formato (N, C, H, W)
                N = tamanho do batch
                C = número de canais
                H = altura da imagem (ex.: profundidade, tempo, etc.)
                W = largura da imagem (ex.: amostras, colunas)

            grid: coordenadas em que a imagem será amostrada,
                formato (N, H_out, W_out, 2)
                - grid[..., 0] contém coordenadas x normalizadas em [-1, 1]
                - grid[..., 1] contém coordenadas y normalizadas em [-1, 1]

        O que a função faz:
            • Converte as coordenadas normalizadas do grid para índices reais em
            [0, H-1] e [0, W-1].
            • Encontra os quatro vizinhos inteiros ao redor de cada posição (x0,y0),
            (x1,y0), (x0,y1), (x1,y1).
            • Calcula os pesos bilineares baseados na distância até cada vizinho.
            • Interpola os valores de input nesses quatro vizinhos usando os pesos.
            • Realiza padding implícito com zeros para coordenadas fora da imagem.

        Retorna:
            out: tensor interpolado no formato (N, C, H_out, W_out),
                contendo os valores da entrada reamostrados
                nas posições especificadas pelo grid.
        """
        # Shapes de entrada
        N, C, H, W = input.shape
        Ng, H_out, W_out, two = grid.shape

        assert Ng == N,  "grid e input devem ter o mesmo N"
        assert two == 2, "última dimensão de grid deve ser 2 (x,y)"

        # Coordenadas normalizadas (x,y) do grid
        x = grid[..., 0]  # (N, H_out, W_out)
        y = grid[..., 1]  # (N, H_out, W_out)

        # Converte coordenadas normalizadas [-1,1] para índices [0,W-1] e [0,H-1]
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

        # Máscaras de pontos dentro da imagem (para aplicar padding zero)
        inside_x0 = (ix0 >= 0) & (ix0 < W)
        inside_x1 = (ix1 >= 0) & (ix1 < W)
        inside_y0 = (iy0 >= 0) & (iy0 < H)
        inside_y1 = (iy1 >= 0) & (iy1 < H)

        # Clamping dos índices para ficarem em [0, W-1] e [0, H-1]
        ix0_cl = xp.clip(ix0, 0, W - 1)
        ix1_cl = xp.clip(ix1, 0, W - 1)
        iy0_cl = xp.clip(iy0, 0, H - 1)
        iy1_cl = xp.clip(iy1, 0, H - 1)

        # Distâncias fracionárias para pesos bilineares
        ix0_f = ix0.astype(ix.dtype)
        iy0_f = iy0.astype(iy.dtype)

        wx1 = ix - ix0_f    # distância até x1
        wx0 = 1.0 - wx1     # distância até x0
        wy1 = iy - iy0_f    # distância até y1
        wy0 = 1.0 - wy1     # distância até y0

        # Pesos 2D (N, H_out, W_out)
        wa = wx0 * wy0  # (x0, y0)
        wb = wx1 * wy0  # (x1, y0)
        wc = wx0 * wy1  # (x0, y1)
        wd = wx1 * wy1  # (x1, y1)

        # Zera pesos onde o índice sai da imagem (padding zero)
        wa *= inside_x0 & inside_y0
        wb *= inside_x1 & inside_y0
        wc *= inside_x0 & inside_y1
        wd *= inside_x1 & inside_y1

        # Índices de batch (N,1,1) para broadcast
        n_idx = xp.arange(N, dtype=xp.int64)[:, None, None]

        # Pega os 4 vizinhos em input → (N, C, H_out, W_out)
        Ia = input[n_idx, :, iy0_cl, ix0_cl]
        Ib = input[n_idx, :, iy0_cl, ix1_cl]
        Ic = input[n_idx, :, iy1_cl, ix0_cl]
        Id = input[n_idx, :, iy1_cl, ix1_cl]

        # Transpõe eixos para (N, C, H_out, W_out) (já estão assim, mas mantemos padrão)
        Ia = xp.transpose(Ia, (0, 3, 1, 2))
        Ib = xp.transpose(Ib, (0, 3, 1, 2))
        Ic = xp.transpose(Ic, (0, 3, 1, 2))
        Id = xp.transpose(Id, (0, 3, 1, 2))

        # Adapta pesos para broadcast em C
        wa = wa[:, None, :, :]  # (N,1,H_out,W_out)
        wb = wb[:, None, :, :]
        wc = wc[:, None, :, :]
        wd = wd[:, None, :, :]

        # Combinação bilinear final
        out = wa * Ia + wb * Ib + wc * Ic + wd * Id  # (N, C, H_out, W_out)
        return out

    # ============================================================
    # LOOP PRINCIPAL DE BEAMFORMING (TX x RX)
    # ============================================================

    # iqdata não é usada diretamente no loop, mas fica como referência (I/Q agrupados)
    iqdata = xp.stack((idata, qdata), axis=-1)  # shape: (nangles, nelems, nsamps, 2)

    # Vetores de saída (somatório DAS) em formato achatado (npixels)
    idas = xp.zeros(npixels, dtype="float")
    qdas = xp.zeros(npixels, dtype="float")

    # Loop sobre ângulos de transmissão e elementos de recepção
    for t, td, ta in tqdm(
        zip(ang_list, txdel, txapo),
        total=len(ang_list),
        desc="Beamforming TX angles"
    ):
        for r, rd, ra in zip(ele_list, rxdel, rxapo):
            # Extrai dados do ângulo t e elemento r
            # idata[t, r] e qdata[t, r] -> (nsamps,)
            # Monta tensor (N=1, C=2, H=1, W=nsamps) para usar grid_sample_xp
            iq = xp.stack([idata[t, r], qdata[t, r]], axis=0).reshape(1, 2, 1, -1)

            # Atraso total = TX + RX para cada pixel
            delays = td + rd

            # Converte atrasos (em amostras) para coordenadas normalizadas [-1,1] no eixo tempo (W)
            # Coloca no eixo x do grid; eixo y é 0 (H=1, efeito 1D no tempo)
            dgs_time = ((delays.reshape(1, 1, -1, 1) * 2 + 1) / idata.shape[-1]) - 1  # (1,1,npix,1)

            # Segundo canal de grid (y) fica zerado, pois não varremos altura
            dgs = xp.concatenate([dgs_time, 0 * dgs_time], axis=-1)  # (1,1,npix,2)

            # Aplica nossa implementação de grid_sample em NumPy/CuPy
            out = grid_sample_xp(iq, dgs, align_corners=False)  # (1,2,1,npix)

            # Achata resultado para (2, npix): canal 0 = I, canal 1 = Q
            out = out.reshape(2, -1)
            ifoc = out[0]
            qfoc = out[1]

            # Caso fosse aplicada rotação de fase (demodulação focal) seria aqui:
            # if fdemod != 0:
            #     tshift = delays.reshape(-1) / fs - grid[:, 2] * 2 / c
            #     theta = 2 * np.pi * fdemod * tshift
            #     ifoc, qfoc = _complex_rotate(ifoc, qfoc, theta)

            # Apodização total (TX * RX) para este par (t,r)
            apods = ta * ra

            # Soma Delay-And-Sum (DAS) sobre TX e RX
            idas += ifoc * apods
            qdas += qfoc * apods

    # ============================================================
    # REORGANIZAÇÃO, CONVERSÃO PARA dB E PLOT DA IMAGEM
    # ============================================================

    # Volta ao formato 2D (nz, nx)
    idas = idas.reshape(out_shape)
    qdas = qdas.reshape(out_shape)

    # Helper para converter CuPy → NumPy (se for o caso)
    to_np = (lambda a: a.get()) if xp.__name__ == "cupy" else (lambda a: np.asarray(a))

    # Sinal complexo IQ da imagem focada
    iq = idas + 1j * qdas
    iq = to_np(iq)  # garante que está em NumPy


    # Imagem em dB (compressão logarítmica)
    bimg_db = 20 * np.log10(np.abs(iq) + 1e-12)  # +eps para evitar log(0)
    bimg_db -= np.amax(bimg_db)                  # normaliza para que o máximo seja 0 dB

    # Limita faixa dinâmica (ex.: -60 dB a 0 dB)
    bimg_db = np.clip(bimg_db, -60, 0)


    # ------------------------------------------------------------
    # Configura eixo físico em mm para plot
    # ------------------------------------------------------------
    x_cpu = to_np(x)
    z_cpu = to_np(z)
    xmin, xmax = float(x_cpu.min()), float(x_cpu.max())
    zmin, zmax = float(z_cpu.min()), float(z_cpu.max())

    # extent: [lateral_min_mm, lateral_max_mm, profundidade_max_mm, profundidade_min_mm]
    # origin="upper" → profundidade aumenta "para baixo" na figura
    extent = [xmin * 1e3, xmax * 1e3, zmax * 1e3, zmin * 1e3]

    plt.figure()
    plt.imshow(
        bimg_db,
        cmap="gray",
        origin="upper",
        extent=extent
    )
    plt.title("Imagem B-mode")
    plt.xlabel("Lateral (mm)")
    plt.ylabel("Profundidade (mm)")
    plt.colorbar(label="dB")
    plt.show()

    # ------------------------------------------------------------
    # Função para salvar figura com nome incremental
    # ------------------------------------------------------------
    def save_fig(fig=None, nome_base="reconstrucao", pasta="IMAGENS", dpi=200):
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

    # Salva a figura gerada na pasta IMAGENS_SALVAS
    # save_fig(nome_base="reconstrucao")
