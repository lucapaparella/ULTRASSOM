
# ============================================================
# RECONSTRUÇÃO DE IMAGEM ULTRASSÔNICA B-MODE (DAS / PLANE WAVE)
#
# Este script faz a reconstrução de imagens ultrassônicas em modo B
# usando beamforming do tipo DAS (Delay-And-Sum) com ondas planas
# (Plane Wave Imaging).
#
# Ele foi cuidadosamente reorganizado a partir de códigos fornecidos
# pelo CUBDL (Challenge on Ultrasound Beamforming), com foco didático:
#   - comentários detalhados em português;
#   - separação clara das etapas do processamento;
#   - opção de usar CPU (NumPy) ou GPU (CuPy) de forma transparente;
#   - explicação dos principais conceitos físicos e numéricos.
#
# A ideia é servir como um roteiro de estudo de um pipeline real de
# reconstrução B-mode, indo desde a leitura dos dados brutos (I/Q)
# até a formação e exibição da imagem final em dB.
# ============================================================

import h5py, numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
from pathlib import Path
# Ignorar avisos de recursos experimentais do CuPy (apenas para evitar poluição do terminal)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



# ============================================================
# OBJETIVO GERAL DO SCRIPT
# ------------------------------------------------------------
# Etapas principais:
# 1. Carregar dados brutos de RF/IQ de um arquivo .hdf5 do CUBDL
# 2. Detectar automaticamente se há GPU disponível:
#       - se não houver → usar NumPy (CPU);
#       - se houver e a flag permitir → usar CuPy (GPU).
# 3. Criar a grade de pixels (x,z) da região de imagem a ser reconstruída
# 4. Calcular atrasos de:
#       - transmissão (TX) das ondas planas;
#       - recepção (RX) de cada elemento do transdutor.
# 5. Calcular as apodizações (pesos) de TX e RX
# 6. Executar o beamforming DAS: para cada pixel, somar os sinais I/Q
#    atrasados e ponderados dos vários ângulos e elementos
# 7. Converter a amplitude complexa em dB (log-compressão),
#    exibir a imagem e salvar em arquivo.
# ============================================================

# Limpa a tela apenas por organização visual do terminal.
# Não interfere nos cálculos nem na lógica do script.
os.system("clear")

# Esta flag serve para imprimir a estrutura interna do primeiro arquivo
# HDF5 apenas uma única vez. Depois, evita repetir a mesma listagem para
# todas as imagens seguintes.
flag = True

# ============================================================
# SELEÇÃO AUTOMÁTICA DO BACKEND (CPU OU GPU)
# ------------------------------------------------------------
# A ideia aqui é criar uma camada de abstração:
#   - usamos sempre "xp" no código (pode ser np ou cp);
#   - usamos "hilbert_xp" para a transformada de Hilbert;
#   - usamos "to_cpu" para garantir que, ao final, os dados estejam
#     em NumPy para exibição com matplotlib.
# ============================================================
USE_CUDA = True         # Se True, tenta usar CuPy/GPU. Se falhar, volta para NumPy/CPU.
xp = np                 # Por padrão, assume NumPy (CPU)
to_cpu = lambda a: a    # Função neutra: se já está em NumPy, retorna o próprio array
hilbert_xp = None       # Será definida depois (versão CPU ou GPU da Hilbert)

if USE_CUDA:
    try:
        import cupy as cp
        from cupyx.scipy.signal import hilbert as c_hilbert

        # Verifica se há pelo menos uma GPU CUDA disponível
        if cp.cuda.runtime.getDeviceCount() > 0:
            # Se houver, trocamos a "biblioteca numérica padrão" para CuPy (GPU)
            xp = cp
            hilbert_xp = c_hilbert       # Versão da transformada de Hilbert que roda na GPU
            to_cpu = cp.asnumpy          # Função para converter de CuPy → NumPy
            dev = cp.cuda.Device()
            dev.use()                    # Seleciona o dispositivo padrão

            # Imprime o nome da GPU detectada (meramente informativo)
            print(f"[CUDA] Usando GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}".center(74))
            print("*" * 74)

        else:
            # CuPy está instalado, mas nenhuma GPU foi detectada
            # → Mantém o processamento na CPU com NumPy
            print("[CUDA] Nenhuma GPU encontrada. Executando no CPU (NumPy)".center(74))
            print("*" * 74)

    except Exception as e:
        # Algum problema ao importar CuPy ou ao acessar a GPU
        # → Faz o fallback automático para NumPy/CPU
        print(f"[CUDA] Erro ao tentar usar CuPy ({e}). Executando no CPU (NumPy).")
        print("*" * 74)

# Se, ao final da tentativa de usar CUDA, ainda não tivermos definido
# "hilbert_xp", significa que ficaremos no CPU e devemos usar a Hilbert
# do SciPy (versão original).
if hilbert_xp is None:
    from scipy.signal import hilbert as s_hilbert
    hilbert_xp = s_hilbert

# ============================================================
# DEFINIÇÃO DAS IMAGENS A PROCESSAR
# ------------------------------------------------------------
# Aqui dizemos quais arquivos de aquisição iremos processar.
# O dataset segue o padrão "TSHXXX.hdf5", onde XXX é o número da imagem.
# ============================================================

imagens= list(range(3, 21))

# Loop principal: reconstrói uma imagem por vez
for acq in imagens:

    # ------------------------------------------------------------
    # ABERTURA DO ARQUIVO HDF5 E INSPEÇÃO DA ESTRUTURA
    # ------------------------------------------------------------
    # "caminho" é o diretório onde estão os arquivos .hdf5 da base TSH
    caminho = r"/home/users/lpaparella/ULTRASSOM/IMAGENS/3_Additional_CUBDL_Data/Plane_Wave_Data/TSH/"
    dataset = "TSH{:03d}".format(acq) + ".hdf5"
    path = caminho + dataset
    
    # Abre o arquivo HDF5 em modo somente leitura ("r")
    arquivo = h5py.File(path, "r")

    def nome_e_tipo(nome, obj):
        """
        Função auxiliar usada com 'arquivo.visititems' para imprimir
        a estrutura interna do arquivo .hdf5.

        Para cada objeto (grupo ou dataset) dentro do arquivo:
          - se for h5py.Group   → exibe que é um GRUPO (similar a uma pasta);
          - se for h5py.Dataset → exibe o nome, o shape (dimensões) e o tipo
                                   (dtype), como se fosse uma matriz salva.

        Isso é útil para:
          - entender como o CUBDL organizou os dados;
          - descobrir nomes e tamanhos dos datasets que iremos ler.
        """
        if isinstance(obj, h5py.Group):
            print(f"GRUPO => {nome}")
        elif isinstance(obj, h5py.Dataset):
            print(f"DATASET => {nome.center(35)} | {str(obj.shape).center(15)} | {obj.dtype}")

    # Mostra a estrutura do arquivo apenas na primeira imagem processada.
    # Depois disso, a flag passa a False e não volta a imprimir.
    if flag:
        print("EXPLORANDO O ARQUIVO HDF5".center(80))
        print("*" * 74)
        arquivo.visititems(nome_e_tipo)
        print("*" * 74)
        flag = False

    # ------------------------------------------------------------
    # CARREGAMENTO DOS DADOS FÍSICOS E DO SINAL
    # ------------------------------------------------------------
    # Vetor de ângulos das ondas planas (em radianos) para cada transmissão
    angles = xp.array(arquivo["angles"])

    # Sinal bruto de RF do dataset:
    #   "channel_data" vem em um formato achatado, que vamos reorganizar.
    #   Carregamos como float32 para reduzir memória e padronizar o tipo.
    idata = xp.array(arquivo["channel_data"], dtype="float32")

    # Reorganiza a dimensão dos dados assumindo:
    #   - 128 elementos (canais) no transdutor
    #   - len(angles) ângulos de transmissão
    #   - "-1" deixa o xp calcular automaticamente o número de amostras no tempo
    # Resultado:
    #   idata.shape = (128, n_angles, n_amostras)
    idata = xp.reshape(idata, (128, len(angles), -1))

    # Troca a ordem dos eixos para ficar no formato:
    #   (n_angles, n_elementos, n_amostras)
    # Isso facilita percorrer primeiro pelos ângulos e depois pelos elementos.
    idata = xp.transpose(idata, (1, 0, 2))

    # Gera a componente Q (parte imaginária) aplicando a transformada de Hilbert
    # ao longo do eixo do tempo (axis=-1). No final teremos:
    #   - idata → componente I
    #   - qdata → componente Q
    qdata = xp.imag(hilbert_xp(idata, axis=-1))

    # Frequência central (fc) do transdutor em Hz
    fc = xp.array(arquivo["modulation_frequency"]).item()

    # Frequência de amostragem (fs) do sinal em Hz
    fs = xp.array(arquivo["sampling_frequency"]).item()

    # Velocidade do som (c) utilizada na aquisição, em m/s
    c = xp.array(arquivo["sound_speed"]).item()

    # Vetor de "time_zero" (referência temporal) para cada ângulo.
    # Aqui é inicializado com zeros, ou seja, não estamos aplicando
    # nenhum ajuste de tempo específico por ângulo.
    time_zero = xp.zeros((len(angles),), dtype="float32")

    # Frequência de demodulação. Como está em zero, significa que
    # não estamos fazendo demodulação para banda-base neste script.
    fdemod = 0


    # ------------------------------------------------------------
    # Construção das posições físicas dos elementos do transdutor
    # ------------------------------------------------------------
    # Muitos datasets do CUBDL usam a geometria do transdutor L11-4v.
    # Ele é um transdutor linear (elementos alinhados ao longo do eixo x).

    # Distância (passo) entre o centro de cada elemento, em metros.
    # O L11-4v possui pitch ≈ 0.3 mm.
    pitch = 0.3e-3   # 0.3 mm

    # Número de elementos do transdutor.
    # Aqui pegamos da segunda dimensão de idata:
    #   idata.shape = (n_angles, n_elementos, n_amostras)
    nelems = idata.shape[1]

    # Coordenadas em x de cada elemento:
    #   - começa em 0
    #   - incrementa pitch a cada elemento
    # Ex.: [0, pitch, 2*pitch, ..., (nelems-1)*pitch]
    xpos = xp.arange(nelems) * pitch

    # Centraliza a abertura em x = 0:
    #   Subtraímos a média de xpos para que os elementos fiquem
    #   distribuídos simetricamente ao redor da origem.
    xpos -= xp.mean(xpos)

    # Cria um vetor 3D de posições dos elementos: (x, y, z)
    # Como o transdutor é linear e está na superfície, adotamos:
    #   y = 0  (sem variação lateral fora do plano)
    #   z = 0  (elementos posicionados no nível da superfície)
    ele_pos = xp.stack([xpos, 0 * xpos, 0 * xpos], axis=1)


    # ------------------------------------------------------------
    # Ajuste de "time zero" para cada ângulo de transmissão
    # ------------------------------------------------------------
    # Neste dataset específico, o 'time_zero' representa um deslocamento
    # temporal necessário para alinhar corretamente o início da aquisição.
    # Ele compensa o fato de que, para ondas planas inclinadas, a frente
    # de onda atinge o campo de visão em tempos diferentes.
    #
    # A regra adotada aqui é: o time_zero deve corresponder ao tempo que
    # a extremidade da abertura (elemento mais à direita) leva para alinhar
    # a frente de onda ao centro da imagem.
    #
    # Para cada ângulo 'a', calculamos um pequeno tempo extra para
    # sincronizar todos os ângulos, garantindo que a propagação comece
    # consistentemente no centro.
    # ------------------------------------------------------------
    for i, a in enumerate(angles):
        # ele_pos[-1, 0]  → posição x do último elemento do transdutor
        #
        # xp.abs(xp.sin(a)) → componente horizontal da onda plana
        #                     (quanto mais inclinado o ângulo, maior o atraso
        #                      necessário para alinhar o centro com a borda)
        #
        # Dividimos por 'c' (velocidade do som) para converter distância → tempo.
        #
        # Resultado: atraso temporal necessário para "trazer" a onda inclinada
        # ao mesmo ponto de referência do centro da abertura.
        time_zero[i] = ele_pos[-1, 0] * xp.abs(xp.sin(a)) / c


    # Limites laterais da imagem (em x):
    # usam a posição do primeiro e do último elemento do transdutor.
    xlims = [ele_pos[0, 0], ele_pos[-1, 0]]

    # Limites de profundidade da imagem (em z):
    # começa em 10 mm e vai até 45 mm.
    zlims = [10e-3, 45e-3]

    # ------------------------------------------------------------
    # CRIAÇÃO DA GRADE DE PIXELS (COORDENADAS X E Z)
    # ------------------------------------------------------------
    # Nesta etapa construímos o grid (x,z) da imagem.
    # Cada ponto desse grid representa um pixel onde o DAS será calculado.
    # Assumimos y = 0 (formação de imagem em um plano 2D x-z).
    # ------------------------------------------------------------
    # Comprimento de onda λ = c / fc
    wvln = c / fc
    # Espaçamento lateral e axial dos pixels:
    #   - aqui usamos λ/2.5 para garantir amostragem suficientemente densa
    #     na reconstrução (evitando aliasing espacial).
    dx = wvln / 2.5
    dz = dx  # Usamos pixels quadrados: mesmo passo em x e z

    # eps vem de “epsilon”: um valor bem pequeno usado aqui para compensar
    # possíveis erros de arredondamento no uso de xp.arange, garantindo que
    # o limite superior seja incluído corretamente.
    eps = 1e-10

    # Vetores de coordenadas x e z dos pixels (em metros)
    x = xp.arange(xlims[0], xlims[1] + eps, dx)
    z = xp.arange(zlims[0], zlims[1] + eps, dz)

    # Cria uma malha 2D (z,x) → xx e zz têm shape (nz, nx)
    zz, xx = xp.meshgrid(z, x, indexing="ij")
    # Coordenada y é assumida zero para todos os pontos (imagem 2D)
    yy = 0 * xx

    # Empilha (x,y,z) para cada pixel, resultando em um array:
    #   grid.shape = (nz, nx, 3)
    grid = xp.stack((xx, yy, zz), axis=-1)

    # f-number (fnum) usado na apodização focalizada (RX)
    fnum = 1

    # Lista de índices dos ângulos de transmissão disponíveis
    ang_list = range(angles.shape[0])
    # Lista de índices dos elementos do transdutor
    ele_list = range(ele_pos.shape[0])

    # out_shape guarda o shape original da grade 2D (nz, nx),
    # para que possamos “desachatar” o resultado no final.
    out_shape = grid.shape[:-1]

    # Reformata a grade de pixels para um vetor 2D:
    #   grid.shape → (npixels, 3), onde:
    #       npixels = nz * nx
    # Cada linha é um pixel com coordenadas (x,y,z).
    grid = xp.reshape(grid, (-1, 3))

    # Quantidade de ângulos de onda plana
    nangles = len(ang_list)
    # Quantidade de elementos do transdutor
    nelems = len(ele_list)
    # Quantidade total de pixels na imagem
    npixels = grid.shape[0]

    # ------------------------------------------------------------
    # ALOCAÇÃO DOS ARRAYS DE ATRASOS E APODIZAÇÕES
    # ------------------------------------------------------------
    # txdel: atrasos de transmissão (TX) para cada ângulo e pixel
    txdel = xp.zeros((nangles, npixels), dtype="float32")
    # rxdel: atrasos de recepção (RX) para cada elemento e pixel
    rxdel = xp.zeros((nelems, npixels), dtype="float32")
    # txapo: apodização (peso) de TX para cada ângulo e pixel
    txapo = xp.ones((nangles, npixels), dtype="float32")
    # rxapo: apodização (peso) de RX para cada elemento e pixel
    rxapo = xp.ones((nelems, npixels), dtype="float32")

    # ============================================================
    # FUNÇÕES PARA CÁLCULO DE ATRASOS E APODIZAÇÕES DE TX (PLANE WAVE)
    # ============================================================

    def delay_plane(grid, angles):
        """
        Calcula os atrasos de TEMPO DE PROPAGAÇÃO para ondas planas.

        ENTRADAS:
          - grid   : coordenadas dos pixels em (x,y,z), shape (npixels, 3)
          - angles : ângulos da onda plana (em radianos), shape (nangles)

        SAÍDA:
          - dist   : "distância efetiva" (projeção) de cada pixel ao plano de onda
                     para cada ângulo → shape (nangles, npixels)

        Ideia:
          Para uma onda plana que se propaga com ângulo 'theta', a frente de onda
          pode ser representada por uma superfície onde:
              dist = x * sin(theta) + z * cos(theta)
          Isso fornece um termo proporcional ao tempo de chegada da frente de onda
          naquele pixel, usado depois para calcular o atraso em amostras.
        """
        # Separa as coordenadas x e z dos pixels, com eixo extra para broadcast
        x = xp.expand_dims(grid[:, 0], 0)   # shape (1, npixels)
        z = xp.expand_dims(grid[:, 2], 0)   # shape (1, npixels)
        # Usa broadcasting entre (nangles,1) e (1,npixels) para obter
        # dist.shape = (nangles, npixels)
        dist = x * xp.sin(angles) + z * xp.cos(angles)
        return dist

    def apod_plane(grid, angles, xlims):
        """
        Calcula a apodização (peso) dos pixels para cada ângulo de onda plana.

        ENTRADAS:
          - grid   : coordenadas dos pixels em (x,y,z), shape (npixels, 3)
          - angles : ângulos de onda plana (rad), shape (nangles,)
          - xlims  : limites laterais da abertura [x_min, x_max]

        SAÍDA:
          - apod   : máscara de apodização, shape (nangles, npixels),
                     com valores 0 ou 1 (neste código) indicando se um pixel
                     contribui ou não para aquele ângulo de transmissão.

        Ideia:
          Para cada ângulo, "projetamos" o pixel de volta na abertura física.
          Se a projeção cair dentro dos limites da abertura (xlims), o pixel
          recebe peso 1; caso contrário, peso 0. Isso limita o campo de visão.
        """
        # Adiciona eixo de ângulo na frente para operar em batch nos pixels
        pix = xp.expand_dims(grid, 0)             # (1, npixels, 3)
        ang = xp.reshape(angles, (-1, 1, 1))      # (nangles, 1, 1)

        # Projeta os pixels de volta para o plano da abertura em função do ângulo
        # Fórmula aproximada: x_proj = x_pixel - z_pixel * tan(angulo)
        x_proj = pix[:, :, 0] - pix[:, :, 2] * xp.tan(ang)

        # Cria máscara booleana para saber se a projeção cai dentro dos limites
        # da abertura (multiplicamos por 1.2 como “folga” de margem).
        mask = (x_proj >= xlims[0] * 1.2) & (x_proj <= xlims[1] * 1.2)

        # Converte a máscara booleana para float32 (0.0 ou 1.0).
        # Resultado final: apod.shape = (nangles, npixels)
        apod = xp.array(mask, dtype="float32")
        return apod

    # Preenche atrasos e apodizações de TX para cada ângulo
    for i, tx in enumerate(ang_list):
        # Atraso geométrico da onda plana para cada pixel
        txdel[i] = delay_plane(grid, angles[[tx]])
        # Ajusta com o "time_zero" específico daquele ângulo
        txdel[i] += time_zero[tx] * c
        # Apodização por onda plana para aqueles pixels
        txapo[i] = apod_plane(grid, angles[tx], xlims)

    # ============================================================
    # FUNÇÕES PARA FOCALIZAÇÃO EM RX (ELEMENTOS DO TRANSDUTOR)
    # ============================================================

    def delay_focus(grid, ele_pos):
        """
        Calcula o atraso de recepção (RX) entre cada pixel e cada elemento.

        ENTRADAS:
          - grid    : coordenadas dos pixels (npixels, 3)
          - ele_pos : coordenadas (x,y,z) do(s) elemento(s) (nelems, 3)
                      ou de um elemento específico, dependendo de como é chamado.

        SAÍDA:
          - dist    : distância euclidiana entre cada pixel e elemento,
                      usada depois para obter o atraso em amostras.
        """
        # Usa broadcasting: grid (npixels,3) - ele_pos (nelems,1,3) ou (1,3)
        # aqui, xp.expand_dims(ele_pos, 0) garante shape compatível.
        dist = xp.linalg.norm(grid - xp.expand_dims(ele_pos, 0), axis=-1)
        return dist

    def apod_focus(grid, ele_pos, fnum=1, min_width=1e-3):
        """
        Calcula a apodização (peso) de RX para cada elemento e pixel.

        ENTRADAS:
          - grid     : pixels (npixels, 3)
          - ele_pos  : posições dos elementos (nelems, 3)
          - fnum     : f-number desejado (controla abertura efetiva)
          - min_width: largura mínima (em metros) para manter
                       elementos próximos do pixel ativados

        SAÍDA:
          - apod     : máscara de apodização, shape (nelems, npixels),
                       com valores 0 ou 1 indicando se o par (elemento, pixel)
                       contribui ou não na soma.

        Ideia:
          Aproxima um critério de f-number: para um pixel a certa profundidade z
          e afastamento lateral x, só elementos que mantêm z/|x| maior que fnum
          (ou que estejam muito próximos lateralmente) são usados.
          Também trata bordas da abertura para não eliminar elementos extremos.
        """
        # Coloca o eixo dos elementos na frente, com um eixo extra de pixel:
        ppos = xp.expand_dims(grid, 0)            # (1, npixels, 3)
        epos = xp.reshape(ele_pos, (-1, 1, 3))    # (nelems, 1, 3)

        # Vetor do elemento até o pixel: v = pixel - elemento
        v = ppos - epos                           # (nelems, npixels, 3)

        # Critério de f-number aproximado: |z/x| > fnum
        # Adicionamos um pequeno valor (1e-30) no denominador para evitar divisão por zero.
        mask = xp.abs(v[:, :, 2] / (v[:, :, 0] + 1e-30)) > fnum

        # Garante que elementos muito próximos lateralmente (|x| <= min_width)
        # não sejam descartados pelo critério de f-number.
        mask = mask | (xp.abs(v[:, :, 0]) <= min_width)

        # Ajustes para as bordas da abertura: mantemos elementos extremos
        # quando o pixel está claramente de um lado (esquerda/direita).
        mask = mask | ((v[:, :, 0] >= min_width) & (ppos[:, :, 0] <= epos[0, 0, 0]))
        mask = mask | ((v[:, :, 0] <= -min_width) & (ppos[:, :, 0] >= epos[-1, 0, 0]))

        # Converte máscara booleana em float32 (0.0 ou 1.0).
        # Resultado: apod.shape = (nelems, npixels)
        apod = xp.array(mask, dtype="float32")
        return apod

    # Preenche atrasos e apodizações de RX para cada elemento
    for j, rx in enumerate(ele_list):
        # Atraso de propagação do pixel até o elemento
        rxdel[j] = delay_focus(grid, ele_pos[rx])
        # Apodização de RX para aquele elemento e todos os pixels
        rxapo[j] = apod_focus(grid, ele_pos[rx])

    # ------------------------------------------------------------
    # CONVERSÃO DE DISTÂNCIAS PARA ÍNDICES DE AMOSTRAS
    # ------------------------------------------------------------
    # Até aqui, txdel e rxdel estão em unidades de distância (metros).
    # Para aplicar os atrasos nos sinais amostrados, precisamos
    # convertê-los para amostras de tempo:
    #   tempo = distância / c      →   índice ≈ tempo * fs
    # Logo:
    #   delay_em_amostras = dist * (fs / c)
    # ------------------------------------------------------------
    txdel *= fs / c
    rxdel *= fs / c

    # ============================================================
    # INTERPOLAÇÃO NO TEMPO (APLICAÇÃO DOS ATRASOS)
    # ============================================================

    def apply_delays(iq, d):
        """
        Aplica atrasos fracionários no tempo usando interpolação linear.

        ENTRADAS:
          - iq : tensor com os sinais I/Q, shape (Nbatch, Nsamples, 2)
                 Nbatch  = pode ser ângulo, elemento ou combinação;
                 Nsamples = número de amostras no tempo;
                 2       = componentes I e Q.
          - d  : atrasos (em amostras), shape (Nbatch, Npix) ou (Nbatch, Npix, 1)

        SAÍDA:
          - ifoc : componente I após foco/interpolação, shape (Nbatch, Npix)
          - qfoc : componente Q após foco/interpolação, shape (Nbatch, Npix)

        Funcionamento:
          Para cada valor fracionário d[n,p], pegamos as amostras
          vizinhas d0 = floor(d) e d1 = d0 + 1 e fazemos:
              out = w0 * iq[n, d0] + w1 * iq[n, d1]
          onde os pesos w0 e w1 dependem da distância até cada vizinho.
        """
        # Se d tiver um eixo extra de tamanho 1 no final, removemos esse eixo
        # para trabalhar sempre com shape (Nbatch, Npix)
        if d.ndim == 3 and d.shape[-1] == 1:
            d = d[..., 0]   # (Nbatch, Npix)

        # Índices inteiros inferiores e superiores
        d0 = xp.floor(d).astype(xp.int32)   # (Nbatch, Npix)
        d1 = d0 + 1                         # (Nbatch, Npix)

        # Para replicar o comportamento de tf.gather_nd com batch_dims=1,
        # construímos um índice de batch:
        b = xp.arange(iq.shape[0])[:, None]  # (Nbatch, 1)

        # iq tem shape (Nbatch, Nsamples, 2).
        # Para cada linha do batch 'n' e pixel 'p':
        #   iq0[n, p, :] = iq[n, d0[n,p], :]
        #   iq1[n, p, :] = iq[n, d1[n,p], :]
        iq0 = iq[b, d0]   # (Nbatch, Npix, 2)
        iq1 = iq[b, d1]   # (Nbatch, Npix, 2)

        # Convertemos os índices para float32 para cálculo dos pesos
        d0f = d0.astype(xp.float32)         # (Nbatch, Npix)
        d1f = d1.astype(xp.float32)         # (Nbatch, Npix)
        df  = d.astype(xp.float32)          # (Nbatch, Npix)

        # Pesos da interpolação linear (um eixo extra para casar com o canal 2)
        w0 = (d1f - df)[..., None]          # (Nbatch, Npix, 1)
        w1 = (df - d0f)[..., None]          # (Nbatch, Npix, 1)

        # Interpolação: combinação ponderada das duas amostras vizinhas
        out = w0 * iq0 + w1 * iq1           # (Nbatch, Npix, 2)

        # Separa as componentes I (canal 0) e Q (canal 1)
        ifoc = out[:, :, 0]
        qfoc = out[:, :, 1]

        return ifoc, qfoc

    # Empilha I e Q em um único tensor:
    #   iqdata.shape = (..., 2)
    # onde o último eixo representa (I, Q).
    iqdata = xp.stack((idata, qdata), axis=-1)

    # Inicializa os acumuladores de DAS para I e Q:
    #   - idas e qdas representam a soma de todas as contribuições
    #     (TX, RX) para cada pixel.
    idas = xp.zeros(npixels, dtype="float")
    qdas = xp.zeros(npixels, dtype="float")

    #___________________________________________________________
    # FUNÇÃO DE INTERPOLAÇÃO ESPACIAL TIPO grid_sample
    # ----------------------------------------------------------
    # Aqui implementamos uma versão em NumPy/CuPy que imita o
    # comportamento de uma interpolação bilinear 2D, semelhante
    # ao que funções como grid_sample fazem em frameworks de deep
    # learning. No contexto deste código, usamos isso para aplicar
    # os atrasos de tempo como se fosse uma “amostragem 1D” ao longo
    # do eixo das amostras, mas reutilizando a lógica 2D.
    #___________________________________________________________

    def grid_sample_xp(input, grid, align_corners=False):
        """
        Realiza interpolação bilinear em uma grade de pontos de amostragem.

        ENTRADAS:
            input: tensor de entrada com shape (N, C, H, W)
                N = tamanho do batch (por exemplo, ângulos)
                C = canais (por exemplo, I/Q)
                H = altura (aqui, quantidade de amostras no tempo)
                W = largura (aqui, interpretado como 1, pois usamos efeito 1D)

            grid:  tensor de coordenadas normalizadas, shape (N, H_out, W_out, 2)
                grid[..., 0] = coordenadas x normalizadas em [-1, 1]
                               (mapeadas para o eixo W, colunas)
                grid[..., 1] = coordenadas y normalizadas em [-1, 1]
                               (mapeadas para o eixo H, linhas)

            align_corners: se True, mapeia extremidades (-1,+1) exatamente
                           para os índices das bordas; se False, usa a
                           convenção com deslocamento de -0.5 no cálculo.

        SAÍDA:
            out: tensor interpolado, shape (N, C, H_out, W_out)

        Em resumo:
          Para cada ponto de 'grid', encontramos os 4 vizinhos mais
          próximos na imagem 'input' e fazemos uma combinação bilinear
          dos valores, respeitando padding zero fora dos limites.
        """
        # Desempacota shapes de entrada
        N, C, H, W = input.shape
        Ng, H_out, W_out, two = grid.shape
        assert Ng == N,  "grid e input devem ter o mesmo N (tamanho de batch)"
        assert two == 2, "última dimensão de grid deve ser 2 (x,y)"

        # Separa as coordenadas normalizadas x e y
        x = grid[..., 0]  # (N, H_out, W_out)
        y = grid[..., 1]  # (N, H_out, W_out)

        # Converte de coordenadas normalizadas [-1,1] para índice de pixel:
        #   - eixo x → coluna (0 .. W-1)
        #   - eixo y → linha   (0 .. H-1)
        if align_corners:
            ix = (x + 1) * (W - 1) / 2.0
            iy = (y + 1) * (H - 1) / 2.0
        else:
            ix = (x + 1) * W / 2.0 - 0.5
            iy = (y + 1) * H / 2.0 - 0.5

        # Índices inteiros dos 4 vizinhos em cada direção
        ix0 = xp.floor(ix).astype(xp.int64)
        iy0 = xp.floor(iy).astype(xp.int64)
        ix1 = ix0 + 1
        iy1 = iy0 + 1

        # Máscaras de pontos que caem dentro da área válida da imagem
        # (serão usados para evitar acesso fora dos limites).
        inside_x0 = (ix0 >= 0) & (ix0 < W)
        inside_x1 = (ix1 >= 0) & (ix1 < W)
        inside_y0 = (iy0 >= 0) & (iy0 < H)
        inside_y1 = (iy1 >= 0) & (iy1 < H)

        # "Clamping" dos índices: se algum valor extrapolar os limites,
        # ele é trazido de volta ao range [0, W-1] ou [0, H-1].
        ix0_cl = xp.clip(ix0, 0, W - 1)
        ix1_cl = xp.clip(ix1, 0, W - 1)
        iy0_cl = xp.clip(iy0, 0, H - 1)
        iy1_cl = xp.clip(iy1, 0, H - 1)

        # Conversão dos índices inteiros de volta para float para calcular
        # distâncias fracionárias (usadas nos pesos bilineares).
        ix0_f = ix0.astype(ix.dtype)
        iy0_f = iy0.astype(iy.dtype)

        # Distâncias fracionárias de cada ponto até seus vizinhos
        wx1 = ix - ix0_f          # distância até o vizinho da direita (x1)
        wx0 = 1.0 - wx1           # distância até o vizinho da esquerda (x0)
        wy1 = iy - iy0_f          # distância até o vizinho de baixo (y1)
        wy0 = 1.0 - wy1           # distância até o vizinho de cima (y0)

        # Combinação 2D de pesos (bilinear):
        #   wa → peso do canto (x0, y0)
        #   wb → peso do canto (x1, y0)
        #   wc → peso do canto (x0, y1)
        #   wd → peso do canto (x1, y1)
        wa = wx0 * wy0
        wb = wx1 * wy0
        wc = wx0 * wy1
        wd = wx1 * wy1

        # Zera pesos de pontos cuja coordenada extrapola a imagem:
        # isso implementa o "padding zeros" fora da área válida.
        wa *= inside_x0 & inside_y0
        wb *= inside_x1 & inside_y0
        wc *= inside_x0 & inside_y1
        wd *= inside_x1 & inside_y1

        # Índices de batch organizados para broadcast:
        n_idx = xp.arange(N, dtype=xp.int64)[:, None, None]

        # Coleta os 4 vizinhos em 'input'.
        # A forma final almejada é (N, C, H_out, W_out).
        Ia = input[n_idx, :, iy0_cl, ix0_cl]  # pont o (x0, y0)
        Ib = input[n_idx, :, iy0_cl, ix1_cl]  # ponto (x1, y0)
        Ic = input[n_idx, :, iy1_cl, ix0_cl]  # ponto (x0, y1)
        Id = input[n_idx, :, iy1_cl, ix1_cl]  # ponto (x1, y1)

        # Ajusta eixos para manter convenção (N, C, H_out, W_out)
        Ia = xp.transpose(Ia, (0, 3, 1, 2))
        Ib = xp.transpose(Ib, (0, 3, 1, 2))
        Ic = xp.transpose(Ic, (0, 3, 1, 2))
        Id = xp.transpose(Id, (0, 3, 1, 2))

        # Adiciona eixo de canais nas máscaras de peso para broadcast
        wa = wa[:, None, :, :]  # (N, 1, H_out, W_out)
        wb = wb[:, None, :, :]
        wc = wc[:, None, :, :]
        wd = wd[:, None, :, :]

        # Combinação bilinear final:
        # cada saída é a média ponderada dos 4 vizinhos.
        out = wa * Ia + wb * Ib + wc * Ic + wd * Id  # (N, C, H_out, W_out)
        return out

    #_______________________________________________________________________
    # LOOP PRINCIPAL DE BEAMFORMING DAS (TX x RX)
    # ----------------------------------------------------------------------
    # Para cada ângulo de transmissão (tx) e cada elemento de recepção (rx):
    #   1. Pegamos o sinal I/Q correspondente;
    #   2. Somamos os atrasos de TX e RX → atraso total;
    #   3. Transformamos esse atraso em coordenadas normalizadas de grid;
    #   4. Chamamos grid_sample_xp para obter o sinal atrasado;
    #   5. Multiplicamos pela apodização combinada (TX * RX);
    #   6. Acumulamos nas somas idas e qdas.
    # ----------------------------------------------------------------------

    for t, td, ta in tqdm(
        zip(ang_list, txdel, txapo),
        total=len(ang_list),
        desc=f"Beamforming => IMAGEM {acq}"
    ):
        for r, rd, ra in zip(ele_list, rxdel, rxapo):
            # Extrai o sinal da aquisição correspondente:
            #   - t-ésimo ângulo de transmissão
            #   - r-ésimo elemento de recepção
            # Em seguida, organiza para o formato (N, C, H, W) esperado pela grid_sample_xp,
            # onde:
            #   N = 1 (apenas um "batch" aqui),
            #   C = 2 (I e Q),
            #   H = número de amostras no tempo,
            #   W = 1  (tratamos como um eixo “falso” para usar a lógica 2D).
            iq = xp.stack([idata[t, r], qdata[t, r]], axis=0).reshape(1, 2, 1, -1)

            # Atraso total (TX + RX) para cada pixel
            delays = td + rd

            # Converte atrasos em amostras para coordenadas normalizadas em [-1,1],
            # mapeadas para o eixo horizontal (x) do grid_sample_xp.
            # O reshape (1,1,npix,1) cria:
            #   N = 1, H_out = 1, W_out = npix, última dimensão = 1 (eixo x).
            dgs_time = ((delays.reshape(1, 1, -1, 1) * 2 + 1) / idata.shape[-1]) - 1

            # Como H=1 (apenas um “pixel” na vertical), definimos o segundo eixo
            # do grid (y) como zero. Isso implementa na prática uma interpolação
            # 1D ao longo do tempo, usando a mesma função 2D.
            dgs = xp.concatenate([dgs_time, 0 * dgs_time], axis=-1)  # (1,1,npix,2)

            # Aplica a interpolação (atraso) nos sinais I/Q
            out = grid_sample_xp(iq, dgs, align_corners=False)  # (1,2,1,npix)

            # Reformata a saída para um formato mais simples:
            #   (1,2,1,npix) → (2, npix)
            out = out.reshape(2, -1)
            ifoc = out[0]   # componente I focada
            qfoc = out[1]   # componente Q focada

            # Apodização combinada TX * RX para este par (t,r)
            apods = ta * ra

            # Soma DAS: acumula contribuições de todos os pares (t,r) em cada pixel
            idas += ifoc * apods
            qdas += qfoc * apods

    # ------------------------------------------------------------
    # REFORMATAÇÃO DO RESULTADO PARA GRADE 2D E CONVERSÃO PARA NUMPY
    # ------------------------------------------------------------
    # idas e qdas ainda estão achatados em um vetor (npixels,).
    # Aqui restauramos o shape original da imagem (nz, nx).
    idas = idas.reshape(out_shape)
    qdas = qdas.reshape(out_shape)

    # helper para converter CuPy→NumPy (ou apenas garantir NumPy)
    to_np = (lambda a: a.get()) if xp.__name__ == "cupy" else (lambda a: np.asarray(a))

    # Sinal complexo focalizado: I + jQ
    iq = idas + 1j * qdas
    iq = to_np(iq)   # garante que está em NumPy para cálculo e plot

    # ------------------------------------------------------------
    # COMPRIMINDO EM dB (LOG-COMPRESSÃO)
    # ------------------------------------------------------------
    # Calcula magnitude do sinal complexo em cada pixel,
    # depois converte para escala logarítmica (20*log10).
    bimg_db = 20 * np.log10(np.abs(iq))

    # Normaliza a imagem para que o valor máximo fique em 0 dB.
    # Assim, todos os demais pixels ficam em valores negativos.
    bimg_db -= np.amax(bimg_db)

    # ------------------------------------------------------------
    # PREPARAÇÃO DOS EIXOS PARA PLOT EM ESCALA FÍSICA
    # ------------------------------------------------------------
    # x e z podem estar em GPU (CuPy). Convertemos para NumPy
    # e depois definimos os limites em milímetros.
    x_cpu = to_np(x)
    z_cpu = to_np(z)
    xmin, xmax = float(x_cpu.min()), float(x_cpu.max())
    zmin, zmax = float(z_cpu.min()), float(z_cpu.max())

    # extent em milímetros:
    #   - eixo x: lateral (da esquerda para a direita)
    #   - eixo z: profundidade (da superfície para o fundo)
    # origin='upper' indica que a primeira linha é o topo da imagem,
    # por isso usamos [xmin,xmax,zmax,zmin] invertendo z.
    extent = [xmin*1e3, xmax*1e3, zmax*1e3, zmin*1e3]

    # Cria figura e exibe a imagem B-mode
    plt.figure()
    # vmin=-60 define uma faixa dinâmica de 60 dB abaixo do máximo:
    # pixels com intensidade menor que -60 dB "somem" (ficam saturados no preto).
    plt.imshow(bimg_db, vmin=-60, cmap="gray", origin="upper",
               aspect="auto", extent=extent)
    plt.title("Imagem B-mode")
    plt.xlabel("Lateral (mm)")
    plt.ylabel("Profundidade (mm)")
    plt.colorbar(label='Intensidade (dB)')
    plt.show()

    # ------------------------------------------------------------
    # FUNÇÃO PARA SALVAR A FIGURA
    # ------------------------------------------------------------
    def save_fig(fig=None, nome_base="reconstrucao", pasta="IMAGENS_SALVAS", dpi=200):
        """
        Salva a figura do B-mode em disco.

        PARÂMETROS:
          - fig       : figura do matplotlib. Se None, usa a figura atual.
          - nome_base : prefixo do nome do arquivo (ex.: "reconstrucao").
          - pasta     : pasta onde a imagem será salva. Será criada se não existir.
          - dpi       : resolução da imagem em pontos por polegada (ppi/dpi).

        O nome final do arquivo inclui o número da aquisição (acq),
        por exemplo: "reconstrucao24.png", "reconstrucao25.png", etc.
        """
        fig = fig or plt.gcf()
        Path(pasta).mkdir(parents=True, exist_ok=True)

        sufixo = acq
        path = Path(pasta) / f"{nome_base}{sufixo}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"[OK] Figura salva em: {path}")

    # Salva a imagem reconstruída da aquisição atual
    save_fig()


   