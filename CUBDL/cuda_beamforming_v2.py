# ============================================================
# DAS (plane-wave) + Compounding coerente (CUDA-ready, robusto)
# Com apodização TX/RX, rotação de fase opcional e clipping seguro
# ============================================================
import h5py, numpy as np, matplotlib.pyplot as plt
from tqdm.auto import tqdm

# --------------------- Configurações -------------------------
USE_CUDA         = True     # tenta usar GPU (CuPy)
USE_TX_APOD      = True     # apodização de transmissão (apertura projetada)
USE_RX_APOD      = True     # apodização de recepção (f-number retangular)
USE_HANN_ELEM    = False     # janela Hann por elemento (somada à RX)
USE_PHASE_ROT    = True    # rotação de fase para dados demodulados
FDEMOD_HZ        = 0.0      # frequência de demodulação; >0 se houver demod
RX_FNUMBER       = 1.0      # f-number da recepção
MIN_WIDTH_M      = 1e-3     # tolerância lateral mínima (m) em apod_focus
TX_APERT_MARGIN  = 1.2      # margem na abertura projetada (apod_plane)
CLIP_INDICES     = True     # clip dos índices fracionários para faixa válida

# -------- parâmetros do usuário --------
path = r"/home/users/lpaparella/ULTRASSOM/IMAGENS/1_CUBDL_Task1_Data/INS015.hdf5"


nx, nz = 128, 512            # resolução da imagem
num_angulos_usados = None    # None -> usar todos; senão um int (ex.: 9, 13, ...)
faixa_dB = 60                # faixa dinâmica para plot

# --------- Backend: CUDA (CuPy) se disponível; caso contrário NumPy ----------
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
            name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
            print(f"[CUDA] Usando GPU: {name}")
        else:
            print("[CUDA] Nenhuma GPU detectada. Usando CPU (NumPy).")
    except Exception as e:
        print(f"[CUDA] CuPy indisponível ({e}). Usando CPU (NumPy).")

if hilbert_xp is None:
    # fallback para CPU
    from scipy.signal import hilbert as s_hilbert
    hilbert_xp = s_hilbert

# --------------------- Utilidades --------------------------------
def janela_hann(n, backend=np):
    n0 = backend.arange(n, dtype=backend.float32)
    return (0.5 - 0.5*backend.cos(2*backend.pi*n0/(n-1))).astype(backend.float32)

def interp1_complex_multi(rf_ch, n_float, xp, clip=True):
    """
    Interpolação linear vetorizada por canal (segura nas bordas).
    rf_ch: (Nelem, Nsamples) complexo64
    n_float: (Nelem, Nx) float32 (índices fracionários por canal/coluna)
    Retorna: (Nelem, Nx) complexo64
    """
    n_float = n_float.astype(xp.float32)
    Nelem, Nsamples = rf_ch.shape

    if clip:
        n_float = xp.clip(n_float, 0.0, xp.float32(Nsamples-2))

    n0 = xp.floor(n_float).astype(xp.int64)
    n1 = n0 + 1
    w  = n_float - n0

    # máscara válida (útil mesmo com clipping, garante segurança)
    valid = (n0 >= 0) & (n1 < Nsamples)

    out = xp.zeros_like(n_float, dtype=rf_ch.dtype)
    if bool(valid.any()):
        row = xp.arange(Nelem, dtype=xp.int64)[:, None]     # (Nelem, 1)
        # Coleta real/imag
        re0 = rf_ch.real[row, n0][valid]
        re1 = rf_ch.real[row, n1][valid]
        im0 = rf_ch.imag[row, n0][valid]
        im1 = rf_ch.imag[row, n1][valid]
        # Interpolação linear
        out_real = (1.0 - w[valid]) * re0 + w[valid] * re1
        out_imag = (1.0 - w[valid]) * im0 + w[valid] * im1
        out.real[valid] = out_real
        out.imag[valid] = out_imag
    return out

def complex_rotate(vals, theta, xp):
    """
    Rotação de fase: vals * e^{j*theta} sem exponencial complexo explícito
    vals: (Nelem, Nx) complexo
    theta: (Nelem, Nx) float32
    """
    c = xp.cos(theta)
    s = xp.sin(theta)
    re = vals.real * c - vals.imag * s
    im = vals.imag * c + vals.real * s
    return re + 1j * im

def apod_plane_row(x_row, z_val, theta, xlims, xp, margin=1.2):
    """
    Apodização TX para uma linha (profundidade fixa z):
    Projeta cada pixel (x,z) de volta na abertura ao longo de theta e
    mantém apenas aqueles cuja projeção cai dentro de [xmin, xmax] * margem.
    Retorna máscara shape (Nx,) float32 {0,1}
    """
    # theta no backend atual (NumPy ou CuPy) e tan no próprio backend:
    theta_xp = xp.asarray(theta, dtype=x_row.dtype)     # escalar shape ()
    x_proj   = x_row - z_val * xp.tan(theta_xp)         # tudo em xp

    xmin, xmax = xlims                                  # já estão em xp.float32 no seu código
    mask = (x_proj >= xmin * margin) & (x_proj <= xmax * margin)
    return mask.astype(xp.float32)


def apod_focus_elem_x(x_row, z_val, x_elem_vec, xp, fnum=1.0, min_width=1e-3, xlims=None):
    """
    Apodização RX retangular por f-number:
    Mantém pares (elem, x) cujo |vz / vx| > fnum  OU  |vx| <= min_width.
    Também corta elementos 'para fora' da abertura se x ultrapassa limites (opcional via xlims).
    Retorna máscara shape (Nelem, Nx) float32 {0,1}
    """
    Nelem = x_elem_vec.shape[0]
    vx = x_row[None, :] - x_elem_vec[:, None]  # (Nelem, Nx)
    vz = xp.full_like(vx, z_val, dtype=xp.float32)
    # Evitar divisão por zero
    eps = xp.float32(1e-30)
    cond_fnum = xp.abs(vz / (vx + eps)) > xp.float32(fnum)
    cond_close = xp.abs(vx) <= xp.float32(min_width)
    mask = cond_fnum | cond_close

    if xlims is not None:
        xmin, xmax = xlims
        # Se pixel está à esquerda da borda esquerda e o elemento está à direita do pixel (ou vice-versa),
        # podemos desligar essas combinações (opcional, efeito suave).
        # Aqui, aplicamos um corte simples por limites laterais do pixel:
        in_apert = (x_row >= xmin) & (x_row <= xmax)            # (Nx,)
        mask = mask & in_apert[None, :]

    return mask.astype(xp.float32)

# --------------------- Carregar dados (CPU) -----------------------------
with h5py.File(path, "r") as f:
    rf = f["/channel_data"][()]                  # (Nang,Nelem,Nsamples) ou (Nelem,Nsamples)
    ele_pos = f["/element_positions"][()].T      # tipicamente (Nelem,3) após .T
    # t0 em segundos; alguns datasets usam /time_zero, outros /start_time
    if "/time_zero" in f:
        t0 = float(np.array(f["/time_zero"][()]).ravel()[0])
    else:
        t0 = float(np.array(f["/start_time"][()]).ravel()[0])

    fs = float(np.array(f["/sampling_frequency"][()]).ravel()[0])
    c  = float(np.array(f["/sound_speed"][()]).ravel()[0]) if "/sound_speed" in f else 1540.0

    # ângulos (vários nomes possíveis)
    ang = None
    for k in ("/transmit_direction", "/angles", "/tx_angles", "/angles_deg", "/angles_rad"):
        if k in f:
            ang = np.array(f[k][()]).ravel()
            break

if ang is None or ang.size == 0:
    ang = np.array([0.0], dtype=float)
ang = ang.astype(float)
if np.max(np.abs(ang)) > 2*np.pi:   # provavelmente graus
    ang_rad = np.deg2rad(ang)
else:
    ang_rad = ang
ang_graus = np.rad2deg(ang_rad)

# normalizar RF para (Nang, Nelem, Nsamples)
rf_arr = np.array(rf)  # ainda na CPU
if rf_arr.ndim == 2:
    rf_arr = rf_arr[np.newaxis, ...]  # (1,Nelem,Nsamples)
elif rf_arr.ndim != 3:
    raise RuntimeError(f"RF shape inesperado: {rf_arr.shape}")
Nang_total, Nelem, Nsamples = rf_arr.shape

# posições x dos elementos (m)
ele = np.array(ele_pos, dtype=float)
x_elem = ele[:, 0] if (ele.ndim == 2 and ele.shape[1] >= 1) else ele.squeeze()
# heurística: se escala parecer mm, converte p/ m
if (np.max(x_elem) - np.min(x_elem)) > 0.1:
    x_elem = x_elem / 1000.0

# grade de imagem (CPU primeiro; depois mandamos p/ GPU se houver)
x_min, x_max = float(x_elem.min()), float(x_elem.max())
grade_x_cpu = np.linspace(x_min, x_max, nx, dtype=np.float32)
z_max = c * (t0 + (Nsamples - 1) / fs) / 2.0
grade_z_cpu = np.linspace(0.0, z_max, nz, dtype=np.float32)

# apodização por elemento (Hann)
hann_cpu = janela_hann(Nelem, backend=np) if USE_HANN_ELEM else np.ones(Nelem, np.float32)

# escolher subconjunto de ângulos (uniforme)
idx_ang = np.arange(Nang_total)
if num_angulos_usados is not None and num_angulos_usados < Nang_total:
    passo = max(1, Nang_total // num_angulos_usados)
    idx_ang = idx_ang[::passo][:num_angulos_usados]

print(f"Compounding coerente com {idx_ang.size} de {Nang_total} ângulos.")

# --- mover constantes para GPU se disponível ---
x_elem_xp  = xp.asarray(x_elem)
grade_x    = xp.asarray(grade_x_cpu)
grade_z    = xp.asarray(grade_z_cpu)
hann_w     = xp.asarray(hann_cpu)
ang_rad_xp = xp.asarray(ang_rad)
xmin_xp, xmax_xp = xp.float32(x_min), xp.float32(x_max)
xlims = (xmin_xp, xmax_xp)

# imagem complexa acumulada (compounding coerente)
img_complex_total = xp.zeros((nz, nx), dtype=xp.complex64)

# ---- loop de compounding por ângulo ----
for ia in tqdm(idx_ang, desc="Beamforming (ângulos)"):
    # traços desse ângulo: (Nelem, Nsamples)  -> enviar para GPU aqui
    rf_angle = xp.asarray(rf_arr[ia, :, :])  # (Nelem, Nsamples) na GPU se xp=cp

    # sinal analítico por canal (interp fracionária estável de fase)
    # rf_analytic = hilbert_xp(to_cpu(rf_angle), axis=1).astype(np.complex64)
    # ---- gerar sinal analítico (Hilbert) por canal ----
    if xp.__name__ == "cupy":
        # CuPy: tudo já está na GPU
        rf_analytic = hilbert_xp(rf_angle, axis=1).astype(xp.complex64)
    else:
        # NumPy: faz Hilbert no CPU
        rf_analytic = hilbert_xp(rf_angle, axis=1).astype(np.complex64)



    rf_analytic = xp.asarray(rf_analytic)  # volta ao backend

    theta = float(ang_rad[ia]) if ia < ang_rad.size else float(ang_rad[0])
    sin_t, cos_t = np.sin(theta), np.cos(theta)  # CPU ok
    sin_t = xp.float32(sin_t); cos_t = xp.float32(cos_t)

    # imagem para este ângulo
    img_complex = xp.zeros((nz, nx), dtype=xp.complex64)

    # ---- pré-cálculo de termos fixos por ângulo ----
    # tan(theta) é usado só em apod_plane_row (calculado lá em CPU->xp)
    # Loop de profundidade
    for iz in range(nz):
        z = grade_z[iz]

        # t_tx(x) = (z*cos + x*sin)/c  -> shape (nx,)
        t_tx_x = (z * cos_t + grade_x * sin_t) / xp.float32(c)  # (nx,)

        # dist[ch, x] = sqrt((x - x_elem[ch])^2 + z^2)
        xdiff = grade_x[None, :] - x_elem_xp[:, None]            # (Nelem, nx)
        dist  = xp.sqrt(xdiff*xdiff + (z*z))                      # (Nelem, nx)
        t_rx  = dist / xp.float32(c)                              # (Nelem, nx)

        # índices fracionários n = (t0 + t_tx + t_rx)*fs
        nidx  = (xp.float32(t0) + t_tx_x[None, :] + t_rx) * xp.float32(fs)  # (Nelem, nx)

        # Interpola todos os canais e todas as colunas de uma vez: (Nelem, nx)
        vals = interp1_complex_multi(rf_analytic, nidx, xp, clip=CLIP_INDICES)  # (Nelem, nx)

        # --------- Rotação de fase (opcional, para dados demodulados) ----------
        if USE_PHASE_ROT and FDEMOD_HZ > 0.0:
            tshift  = nidx / xp.float32(fs)               # (Nelem, nx) em segundos
            tdemod  = xp.float32(2.0) * z / xp.float32(c) # escalar (seg) = duas-passagens até z
            theta_r = xp.float32(2.0*np.pi*FDEMOD_HZ) * (tshift - tdemod)  # (Nelem, nx)
            vals = complex_rotate(vals, theta_r, xp)

        # --------- Apodizações -----------------------------------------------
        # RX (f-number retangular)
        if USE_RX_APOD:
            apod_rx = apod_focus_elem_x(
                grade_x, z, x_elem_xp, xp,
                fnum=RX_FNUMBER, min_width=MIN_WIDTH_M, xlims=xlims
            )  # (Nelem, nx)
        else:
            apod_rx = xp.ones((Nelem, nx), dtype=xp.float32)

        # Hann por elemento
        if USE_HANN_ELEM:
            apod_rx = apod_rx * hann_w[:, None].astype(apod_rx.dtype)

        # TX (apertura projetada)
        if USE_TX_APOD:
            apod_tx_row = apod_plane_row(
                grade_x, z, theta, xlims, xp, margin=TX_APERT_MARGIN
            )  # (nx,)
        else:
            apod_tx_row = xp.ones((nx,), dtype=xp.float32)

        # Peso total por (elem, x)
        apod_total = apod_rx * apod_tx_row[None, :]

        # Soma coerente nos canais -> linha (nx,)
        soma_linha = xp.sum(apod_total.astype(vals.dtype) * vals, axis=0)  # (nx,)
        img_complex[iz, :] = soma_linha

    # soma coerente entre ângulos (compounding)
    img_complex_total += img_complex

# --------------------- envelope + log e plot -----------------------------
env = xp.abs(img_complex_total)
env /= (env.max() + 1e-12)
img_db = 20.0 * xp.log10(env + 1e-12)
img_db = xp.clip(img_db, -faixa_dB, 0)

# trazer para CPU para plot
img_db_cpu = to_cpu(img_db)
grade_x_cpu = to_cpu(grade_x)
grade_z_cpu = to_cpu(grade_z)

# plot em mm
extent_mm = [grade_x_cpu[0]*1e3, grade_x_cpu[-1]*1e3,
             grade_z_cpu[-1]*1e3, grade_z_cpu[0]*1e3]
plt.figure(figsize=(6,8))
plt.imshow(img_db_cpu, cmap="gray", extent=extent_mm,
           aspect="auto", vmin=-faixa_dB, vmax=0)
plt.xlabel("Lateral (mm)")
plt.ylabel("Profundidade (mm)")
title_cuda = "ON" if xp is not np else "OFF"
plt.title(f"B-mode DAS + Compounding ({idx_ang.size} ângulos) [CUDA {title_cuda}]"
          f"\nTX_apod={USE_TX_APOD}, RX_apod={USE_RX_APOD}, Hann={USE_HANN_ELEM}, PhaseRot={USE_PHASE_ROT}")
plt.colorbar(label="dB")
plt.tight_layout()
plt.show()
