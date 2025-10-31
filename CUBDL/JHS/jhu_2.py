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
input("Aperte ENTER para continuar...")
#------------------------------------------------------------------------------------

# P.angles (nangles,), radianos
angles = xp.array(arquivo["angles"])
print(f"ANGLES => {angles.shape}")
# P.ele_pos (nelems, 3), posições (x,y,z) dos elementos
ele_pos = xp.array(arquivo["element_positions"], dtype="float32")
print(f"ELE_POS => {ele_pos.shape}")
# P.fc, P.fs, P.fdemod, P.c, P.time_zero (arrays/escalares)

fs = xp.array(arquivo["sampling_frequency"]).item()
fdemod = 0
time_zero = -1 * xp.array(arquivo["time_zero"], dtype="float32")
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
tamanho_pixel = xp.array(arquivo["pixel_d"]).item()
print(f"Tamanho pixel => {tamanho_pixel}")

#-------------------------------------------------------
# P = PICMUSData(database_path, acq, target, dtype)
P = xp.array(arquivo["channel_data"], dtype="float32")
#-------------------------------------------------------
# Define pixel grid limits (assume y == 0)
ele_pos = xp.array(arquivo["element_positions"], dtype="float32")
xlims = [ele_pos[0, 0], ele_pos[-1, 0]]
# "Eu quero que a imagem de ultrassom comece a 5 mm de profundidade e termine a 55 mm "
# "de profundidade."

# A imagem não vai mostrar o que está colado no transdutor (de 0 a 5 mm), e também 
# não vai mostrar nada além de 55 mm de profundidade.

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
#-------------------------------------------------------

grid = make_pixel_grid(xlims, zlims, dx, dz)
fnum = 1

