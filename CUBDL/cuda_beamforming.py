# ================================
# DAS (plane-wave) + Compounding coerente (CUDA-ready)
# ================================
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
        # Garante que há device
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
path = r"C:\Users\lucap\Documents\CUBDL_Data\CUBDL_Data\3_Additional_CUBDL_Data\Focused_Data\OSL\OSL011.hdf5"

nx, nz = 128, 512              # resolução da imagem
num_angulos_usados = None     # None -> usar todos; ou um int (ex.: 9, 13, ...)
faixa_dB = 60                 # faixa dinâmica para plot

# -------- utilidades --------
def janela_hann(n, backend=np):
    n0 = backend.arange(n, dtype=backend.float32)
    return (0.5 - 0.5*backend.cos(2*backend.pi*n0/(n-1))).astype(backend.float32)

def interp1_complex_multi(rf_ch, n_float, xp):
    """
    Interpolação linear vetorizada por canal.
    rf_ch: (Nelem, Nsamples) complexo
    n_float: (Nelem, Nx) float32 (índices fracionários por canal e coluna x)
    Retorna: (Nelem, Nx) complexo
    """
    n_float = n_float.astype(xp.float32)
    n0 = xp.floor(n_float).astype(xp.int64)
    n1 = n0 + 1
    w  = n_float - n0
    Nelem, Nsamples = rf_ch.shape

    valid = (n0 >= 0) & (n1 < Nsamples)
    out = xp.zeros_like(n_float, dtype=rf_ch.dtype)
    if bool(valid.any()):
        row = xp.arange(Nelem, dtype=xp.int64)[:, None]  # (Nelem,1) broadcast
        re0 = rf_ch.real[row, n0][valid]
        re1 = rf_ch.real[row, n1][valid]
        im0 = rf_ch.imag[row, n0][valid]
        im1 = rf_ch.imag[row, n1][valid]
        out_real = (1.0 - w[valid]) * re0 + w[valid] * re1
        out_imag = (1.0 - w[valid]) * im0 + w[valid] * im1
        out.real[valid] = out_real
        out.imag[valid] = out_imag
    return out

# -------- carregar dados (CPU) --------
with h5py.File(path, "r") as f:
    rf = f["/channel_data"][()]                  # (Nang,Nelem,Nsamples) ou (Nelem,Nsamples)
    ele_pos = f["/element_positions"][()].T      # tipicamente (Nelem,3) após .T
    # t0 = float(np.array(f["/time_zero"][()]).ravel()[0])
    t0 = float(np.array(f["/start_time"][()]).ravel()[0])
    fs = float(np.array(f["/sampling_frequency"][()]).ravel()[0])
    c = float(np.array(f["/sound_speed"][()]).ravel()[0]) if "/sound_speed" in f else 1540.0
    
    teste = f["/beamformed_data"][()].T
      
    print(f" fs => {teste.shape}")
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

print(f"Compounding coerente com {idx_ang.size} de {Nang_total} ângulos.")

# --- mover constantes para GPU se disponível ---
x_elem_xp  = xp.asarray(x_elem)
grade_x    = xp.asarray(grade_x_cpu)
grade_z    = xp.asarray(grade_z_cpu)
apod       = xp.asarray(apod_cpu)
ang_rad_xp = xp.asarray(ang_rad)

# imagem complexa acumulada (compounding coerente)
img_complex_total = xp.zeros((nz, nx), dtype=xp.complex64)

# ---- loop de compounding por ângulo ----
for ia in tqdm(idx_ang, desc="Beamforming (ângulos)"):
    # traços desse ângulo: (Nelem, Nsamples)  -> enviar para GPU aqui
    rf_angle = xp.asarray(rf_arr[ia, :, :])  # (Nelem, Nsamples) na GPU se xp=cp

    # sinal analítico por canal (interp fracionária estável de fase)
    rf_analytic = hilbert_xp(rf_angle, axis=1).astype(xp.complex64)

    theta = float(ang_rad[ia]) if ia < ang_rad.size else float(ang_rad[0])
    sin_t, cos_t = np.sin(theta), np.cos(theta)  # pequeno custo; pode ficar em CPU
    sin_t = xp.float32(sin_t); cos_t = xp.float32(cos_t)

    # imagem para este ângulo
    img_complex = xp.zeros((nz, nx), dtype=xp.complex64)

    # Pré-calculo por profundidade: vetor t_tx para todas as colunas (nx)
    for iz in range(nz):
        z = grade_z[iz]

        # t_tx(x) = (z*cos + x*sin)/c  -> shape (nx,)
        t_tx_x = (z * cos_t + grade_x * sin_t) / c  # (nx,)

        # Para cada canal, compute t_rx(x) e índices fracionários n = (t0 + t_tx + t_rx)*fs
        # dist[ch, x] = sqrt((x - x_elem[ch])^2 + z^2)
        # Vamos vetorizar: (Nelem, nx)
        xdiff = grade_x[None, :] - x_elem_xp[:, None]            # (Nelem, nx)
        dist  = xp.sqrt(xdiff*xdiff + (z*z))                      # (Nelem, nx)
        t_rx  = dist / c                                          # (Nelem, nx)
        nidx  = (t0 + t_tx_x[None, :] + t_rx) * fs                # (Nelem, nx)

        # Interpola todos os canais e todas as colunas de uma vez: (Nelem, nx)
        vals = interp1_complex_multi(rf_analytic, nidx, xp)       # (Nelem, nx)

        # Aplica apodização por canal e soma coerente nos canais -> linha (nx,)
        soma_linha = xp.sum((apod[:, None].astype(vals.dtype)) * vals, axis=0)  # (nx,)
        img_complex[iz, :] = soma_linha

    # soma coerente entre ângulos (compounding)
    img_complex_total += img_complex

# envelope + log (na GPU se disponível)
env = xp.abs(img_complex_total)
env /= (env.max() + 1e-12)
img_db = 20.0 * xp.log10(env + 1e-12)
img_db = xp.clip(img_db, -faixa_dB, 0)

# trazer para CPU para plot
img_db_cpu = to_cpu(img_db)


print(f"imagem => {img_db_cpu.shape}")


grade_x_cpu = to_cpu(grade_x)
grade_z_cpu = to_cpu(grade_z)

# plot em mm
extent_mm = [grade_x_cpu[0]*1e3, grade_x_cpu[-1]*1e3,
             grade_z_cpu[-1]*1e3, grade_z_cpu[0]*1e3]
plt.figure(figsize=(6,8))
plt.imshow(teste, cmap="gray", extent=extent_mm,
           aspect="auto", vmin=-faixa_dB, vmax=0)
plt.xlabel("Lateral (mm)")
plt.ylabel("Profundidade (mm)")
plt.title(f"B-mode DAS")
plt.colorbar(label="dB")
plt.tight_layout()


plt.figure(figsize=(6,8))
plt.imshow(img_db_cpu, cmap="gray", extent=extent_mm,
           aspect="auto", vmin=-faixa_dB, vmax=0)
plt.xlabel("Lateral (mm)")
plt.ylabel("Profundidade (mm)")
plt.title(f"B-mode DAS + Compounding ({idx_ang.size} ângulos) [CUDA {'ON' if xp is not np else 'OFF'}]")
plt.colorbar(label="dB")
plt.tight_layout()
plt.show()
