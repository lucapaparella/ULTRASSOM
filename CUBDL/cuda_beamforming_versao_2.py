# ================================
# DAS (plane-wave) + Compounding coerente (CUDA-ready) — versão otimizada
# ================================
import os, matplotlib

assert os.environ.get("DISPLAY"), "X11 não ativo — reconecte com ssh -X"

matplotlib.use("TkAgg")

import h5py, numpy as np, matplotlib.pyplot as plt
from tqdm.auto import tqdm

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

# -------- parâmetros do usuário --------
path = r"/home/users/lpaparella/ULTRASSOM/IMAGENS/2_Post_CUBDL_JHU_Breast_Data/JHU030.hdf5"

# nx, nz = 128, 512              # resolução da imagem
num_angulos_usados = None      # None -> usar todos; ou um int (ex.: 9, 13, ...)
faixa_dB = 60                  # faixa dinâmica para plot
Bz = 64                        # tamanho do bloco em z (ajuste conforme memória); usa processamento por blocos

# -------- utilidades --------
def janela_hann(n, backend=np):
    n0 = backend.arange(n, dtype=backend.float32)
    return (0.5 - 0.5*backend.cos(2*backend.pi*n0/(n-1))).astype(backend.float32)

def interp1_complex_multi(rf_ch, n_float, xp):
    """
    Interpolação linear vetorizada por canal.
    rf_ch: (Nelem, Nsamples) complexo
    n_float: (Nelem, M) float32 (índices fracionários por canal e coluna(s))
    Retorna: (Nelem, M) complexo
    """
    n_float = n_float.astype(xp.float32)
    n0 = xp.floor(n_float).astype(xp.int64)
    n1 = n0 + 1
    w  = n_float - n0
    Nelem, Nsamples = rf_ch.shape

    valid = (n0 >= 0) & (n1 < Nsamples)
    out = xp.zeros_like(n_float, dtype=rf_ch.dtype)
    # if bool(valid.any()):
    #     row = xp.arange(Nelem, dtype=xp.int64)[:, None]  # (Nelem,1) para broadcast
    #     re0 = rf_ch.real[row, n0][valid]
    #     re1 = rf_ch.real[row, n1][valid]
    #     im0 = rf_ch.imag[row, n0][valid]
    #     im1 = rf_ch.imag[row, n1][valid]
    #     out_real = (1.0 - w[valid]) * re0 + w[valid] * re1
    #     out_imag = (1.0 - w[valid]) * im0 + w[valid] * im1
    #     out.real[valid] = out_real
    #     out.imag[valid] = out_imag
    # return out
    row = xp.arange(Nelem, dtype=xp.int64)[:, None]  # (Nelem,1) para broadcast
    re0 = rf_ch.real[row, n0]
    re1 = rf_ch.real[row, n1]
    im0 = rf_ch.imag[row, n0]
    im1 = rf_ch.imag[row, n1]
    out_real = (1.0 - w) * re0 + w * re1
    out_imag = (1.0 - w) * im0 + w * im1
    out.real = out_real
    out.imag = out_imag
    return out
# -------- carregar dados (CPU) --------
with h5py.File(path, "r") as f:
    rf = f["/channel_data"][()]                  # (Nang,Nelem,Nsamples) ou (Nelem,Nsamples)
    ele_pos = f["/element_positions"][()].T      # tipicamente (Nelem,3) após .T
    # t0 = float(np.array(f["/time_zero"][()]).ravel()[0])
    # t0 = float(np.array(f["/start_time"][()]).ravel()[0])
    fs = float(np.array(f["/sampling_frequency"][()]).ravel()[0])
    c  = float(np.array(f["/sound_speed"][()]).ravel()[0]) if "/sound_speed" in f else 1540.0
   
    if "/start_time" in f:
        t0 = float(np.array(f["/start_time"][()]).ravel()[0])
    else:
        t0 = float(np.array(f["/time_zero"][()]).ravel()[0])

    
    # tenta pixel_positions (presentes em alguns datasets)
    if "/pixel_positions" in f:
        nx, nz = f["/pixel_positions"].shape[1], f["/pixel_positions"].shape[2]
        print(f["/pixel_positions"].shape[0])

    # tenta depth_axis e angles (caso beamformed_data não exista)
    elif "/depth_axis" in f and "/angles" in f:
        nz = f["/depth_axis"].shape[-1]
        nx = f["/angles"].shape[-1]
    
    # tenta usar beamformed_data (caso exista)
    elif "/beamformed_data" in f:
        shape =f["/beamformed_data"].shape
        if len(shape)==2:
            nx, nz = shape
        else:
            _,nx, nz = shape
        

    else:
        # fallback
        nx, nz = 128, 512

    print(f"[INFO] Parâmetros de grade: nx = {nx}, nz = {nz}")
    print(f"b1 => {f["/beamformed_data"].shape}")
    img_beam = np.array(f["/beamformed_data"][()], dtype=np.float32)
    

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
if (np.max(x_elem) - np.min(x_elem)) > 0.1:  # heurística: estava em mm
    x_elem = x_elem / 1000.0  # mm -> m

# grade de imagem (CPU primeiro; depois mandamos p/ GPU se houver)
x_min, x_max = float(x_elem.min()), float(x_elem.max())
grade_x_cpu = np.linspace(x_min, x_max, nx, dtype=np.float32)
z_max = c * (t0 + (Nsamples - 1) / fs) / 2.0
grade_z_cpu = np.linspace(0.0, z_max, nz, dtype=np.float32)

# apodização por elemento
apod_cpu = janela_hann(Nelem, backend=np)

# escolher subconjunto de ângulos (uniforme)
idx_ang = np.arange(Nang_total)
if num_angulos_usados is not None and num_angulos_usados < Nang_total:
    passo = max(1, Nang_total // num_angulos_usados)
    idx_ang = idx_ang[::passo][:num_angulos_usados]

print(f"[INFO] Compounding coerente com {idx_ang.size} de {Nang_total} ângulos.")

# --- mover constantes para GPU se disponível ---
x_elem_xp  = xp.asarray(x_elem)        # (Nelem,)
grade_x    = xp.asarray(grade_x_cpu)   # (nx,)
grade_z    = xp.asarray(grade_z_cpu)   # (nz,)
apod       = xp.asarray(apod_cpu)      # (Nelem,)
ang_rad_xp = xp.asarray(ang_rad)       # (Nang_total,)

# Pré-cálculos que não dependem de z nem do ângulo:
xdiff = grade_x[None, :] - x_elem_xp[:, None]     # (Nelem, nx), fixo durante todo o beamforming
# Grade (z,x) — depende só da malha; `t_tx` vai usar sen/cos por ângulo
zz, xx = xp.meshgrid(grade_z, grade_x, indexing="ij")  # zz:(nz,nx)  xx:(nz,nx)

# imagem complexa acumulada (compounding coerente)
img_complex_total = xp.zeros((nz, nx), dtype=xp.complex64)

# ---- loop de compounding por ângulo ----
for ia in tqdm(idx_ang, desc="Beamforming (ângulos)"):
    # traços desse ângulo: (Nelem, Nsamples)  -> enviar para GPU aqui
    rf_angle = xp.asarray(rf_arr[ia, :, :])  # (Nelem, Nsamples) na GPU se xp=cp

    # sinal analítico por canal (interp fracionária estável de fase)
    rf_analytic = hilbert_xp(rf_angle, axis=1).astype(xp.complex64)

    # seno/cosseno do ângulo (em float32 p/ economizar)
    theta = float(ang_rad[ia]) if ia < ang_rad.size else float(ang_rad[0])
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    sin_t = xp.float32(sin_t); cos_t = xp.float32(cos_t)

    # tempo de TX para toda a malha (nz, nx)
    # t_tx(z,x) = (z*cos_t + x*sin_t) / c
    t_tx = (zz * cos_t + xx * sin_t) / c        # (nz, nx)

    # imagem complexa para este ângulo
    img_complex = xp.zeros((nz, nx), dtype=xp.complex64)

    # -------- processamento por blocos de z --------
    Bz_eff = int(min(Bz, nz)) if Bz is not None else nz
    for z0 in range(0, nz, Bz_eff):
        z1 = min(z0 + Bz_eff, nz)
        B = z1 - z0

        # fatias do bloco
        zz_blk   = zz[z0:z1, :]            # (B, nx)
        t_tx_blk = t_tx[z0:z1, :]          # (B, nx)

        # distâncias para todos os canais × (B,nx): broadcasting
        # xdiff:(Nelem, nx) -> (Nelem,1,nx)
        # zz_blk:(B,nx)     -> (1,B,nx)
        dist_blk = xp.sqrt(xdiff[:, None, :]**2 + zz_blk[None, :, :]**2)  # (Nelem,B,nx)
        # índices fracionários
        nidx_blk = (t0 + t_tx_blk[None, :, :] + dist_blk / c) * fs        # (Nelem,B,nx)

        # achata para (Nelem, B*nx) para reutilizar a interp 2D
        nidx_flat = nidx_blk.reshape(Nelem, -1)                           # (Nelem, B*nx)

        # interpola (continua vetorizado em canais)
        vals_flat = interp1_complex_multi(rf_analytic, nidx_flat, xp)     # (Nelem, B*nx)

        # volta ao shape do bloco e soma coerente por canais
        vals_blk = vals_flat.reshape(Nelem, B, nx)                         # (Nelem,B,nx)
        soma_blk = xp.sum(apod[:, None, None].astype(vals_blk.dtype) * vals_blk, axis=0)  # (B, nx)

        # escreve no volume da imagem
        img_complex[z0:z1, :] = soma_blk

    # soma coerente entre ângulos (compounding)
    img_complex_total += img_complex

# envelope + log (na GPU se disponível)
env = xp.abs(img_complex_total)
env /= (env.max() + 1e-12)
img_db = 20.0 * xp.log10(env + 1e-12)
img_db = xp.clip(img_db, -faixa_dB, 0)

# trazer para CPU para plot
img_db_cpu = to_cpu(img_db)
grade_x_cpu = to_cpu(grade_x)
grade_z_cpu = to_cpu(grade_z)

print(f"[INFO] imagem reconstruída => {img_db_cpu.shape}")


# converte para dB (evita log de zero)
# img_beam = 20 * np.log10(np.abs(img_beam) / (np.max(np.abs(img_beam)) + 1e-12))
# img_beam = np.clip(img_beam, -60, 0)  # faixa dinâmica de 60 dB
# converte para NumPy antes de plotar
img_beam_cpu = img_beam.get() if hasattr(img_beam, "get") else img_beam
print(f"[INFO] imagem original => {img_beam_cpu[0].shape}")
# plot em mm
extent_mm = [grade_x_cpu[0]*1e3, grade_x_cpu[-1]*1e3,
             grade_z_cpu[-1]*1e3, grade_z_cpu[0]*1e3]

# --- Plot ---
plt.figure(figsize=(6, 8))
# origin="lower": mostra profundidade aumentando para baixo
plt.imshow(img_beam.T, cmap="gray", 
           aspect="auto", vmin=-faixa_dB, vmax=0)
plt.title("Imagem original: beamformed_data (em dB)")
plt.xlabel("Lateral (x)")
plt.ylabel("Profundidade (z)")
plt.colorbar(label="dB")
plt.tight_layout()
plt.show()
# for angulo in range(75):
#     plt.imshow(img_beam_cpu[angulo, :, :], cmap='gray', aspect='auto', extent=extent_mm,
#    vmin=-faixa_dB, vmax=0)
#     plt.title(f"Ângulo {angulo}")
#     plt.pause(0.2)  # pausa curta para ver cada um
#     plt.clf()

plt.figure(figsize=(6,8))
plt.imshow(img_db_cpu, cmap="gray", extent=extent_mm,
            vmin=-faixa_dB, vmax=0)
plt.xlabel("Lateral (mm)")
plt.ylabel("Profundidade (mm)")
plt.title(f"B-mode DAS + Compounding ({idx_ang.size} ângulos) [CUDA {'ON' if xp is not np else 'OFF'}]")
plt.colorbar(label="dB")
plt.tight_layout()
plt.show()

