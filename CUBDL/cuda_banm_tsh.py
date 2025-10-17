# ================================
# DAS (plane-wave) + Coherent Compounding (CUDA-ready)
# Arquivo HDF5 com:
#   angles:          (Nang,)
#   channel_data:    (Nelem, Nsamples_total)  -> concatenado por ângulos
#   sampling_frequency: (1,)
#   sound_speed:        (1,)
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
        if cp.cuda.runtime.getDeviceCount() > 0:
            xp = cp
            hilbert_xp = c_hilbert
            to_cpu = cp.asnumpy
            dev = cp.cuda.Device(); dev.use()
            print(f"[CUDA] GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
        else:
            print("[CUDA] Nenhuma GPU detectada. Usando CPU (NumPy).")
    except Exception as e:
        print(f"[CUDA] CuPy indisponível ({e}). Usando CPU (NumPy).")

if hilbert_xp is None:
    from scipy.signal import hilbert as s_hilbert
    hilbert_xp = s_hilbert

# -------- parâmetros do usuário --------
path = r"C:\Users\lucap\Documents\CUBDL_Data\CUBDL_Data\1_CUBDL_Task1_Data\TSH002\TSH002.hdf5"

# Se o arquivo não trouxer posições dos elementos, assumimos arranjo linear:
PITCH_M = 0.2e-3   # 0.30 mm típico; ajuste para o seu transdutor
T0 = 0.0           # start_time padrão (ajuste se souber t0 real)

# Resolução/grade (se desejar forçar valores, altere aqui; por padrão é automático)
NX_USER = None     # None => usa Nelem (recomendado como ponto de partida)
NZ_USER = None     # None => calculado a partir de tempo de voo

# Bloco em profundidade para economizar memória (GPU/CPU)
Bz = 64

# -------- utilidades --------
def janela_hann(n, backend=np):
    n0 = backend.arange(n, dtype=backend.float32)
    return (0.5 - 0.5*backend.cos(2*backend.pi*n0/(n-1))).astype(backend.float32)

def interp1_complex_multi(rf_ch, n_float, xp):
    """
    Interpolação linear vetorizada por canal.
    rf_ch:  (Nelem, Nsamples) complexo
    n_float:(Nelem, M) float32 (índices fracionários por canal)
    Retorna:(Nelem, M) complexo
    """
    n_float = n_float.astype(xp.float32)
    n0 = xp.floor(n_float).astype(xp.int64)
    n1 = n0 + 1
    w  = n_float - n0
    Nelem, Nsamples = rf_ch.shape

    valid = (n0 >= 0) & (n1 < Nsamples)
    out = xp.zeros_like(n_float, dtype=rf_ch.dtype)
    if bool(valid.any()):
        row = xp.arange(Nelem, dtype=xp.int64)[:, None]         # (Nelem,1)
        re0 = rf_ch.real[row, n0][valid]; re1 = rf_ch.real[row, n1][valid]
        im0 = rf_ch.imag[row, n0][valid]; im1 = rf_ch.imag[row, n1][valid]
        out_real = (1.0 - w[valid]) * re0 + w[valid] * re1
        out_imag = (1.0 - w[valid]) * im0 + w[valid] * im1
        out.real[valid] = out_real
        out.imag[valid] = out_imag
    return out

# -------- carregar dados (CPU) --------
with h5py.File(path, "r") as f:
    ang = np.array(f["/angles"][()], dtype=float).ravel()              # (Nang,)
    rf2 = np.array(f["/channel_data"][()], dtype=np.float32)           # (Nelem, Nsamples_total)
    fs  = float(np.array(f["/sampling_frequency"][()]).ravel()[0])
    c   = float(np.array(f["/sound_speed"][()]).ravel()[0]) if "/sound_speed" in f else 1540.0
    mod = float(np.array(f["/modulation_frequency"][()]).ravel()[0]) if "/modulation_frequency" in f else None

    #correção pelo arquivo do excel
    print(f"sound_speed => {c}")
# Detectar formatos
Nang = ang.size
Nelem, Nsamples_total = rf2.shape
if Nsamples_total % Nang != 0:
    raise RuntimeError(f"Nsamples_total={Nsamples_total} não é múltiplo de Nang={Nang}.")


Nsamples = Nsamples_total // Nang
print(f"Nsamples/ang => {Nsamples}")

# Reformata: (Nelem, Nsamples_total) -> (Nang, Nelem, Nsamples)
rf_arr = rf2.reshape(Nelem, Nang, Nsamples).transpose(1, 0, 2).astype(np.float32)
print(f"rf_arr => {rf_arr.shape}")
# Ângulos em radianos
ang_rad = ang if np.max(np.abs(ang)) <= 2*np.pi else np.deg2rad(ang)
ang_rad = ang_rad.astype(float)

# ---- grade automática ----
# lateral: usamos NX = Nelem por padrão (boa aproximação inicial)
nx = NX_USER if (NX_USER is not None) else Nelem
# profundidade: do tempo de voo máximo (ida e volta)
z_max = c * (T0 + (Nsamples - 1) / fs) / 2.0
nz = NZ_USER if (NZ_USER is not None) else min(max(256, Nsamples // 2), Nsamples)  # heurística ok p/ começar
print(f"nz => {nz}")
# posições x dos elementos (m) — arranjo linear centrado
#   [-pitch*(Nelem-1)/2, ..., +pitch*(Nelem-1)/2]
x_elem = (np.arange(Nelem, dtype=float) - (Nelem - 1) / 2.0) * PITCH_M

# grade x–z (CPU; depois mandamos p/ GPU)
grade_x_cpu = np.linspace(x_elem.min(), x_elem.max(), nx, dtype=np.float32)
grade_z_cpu = np.linspace(0.0, z_max, nz, dtype=np.float32)

print(f"grade_x_cpu => {grade_x_cpu.shape}")
print(f"grade_z_cpu => {grade_z_cpu.shape}")
# apodização por elemento
apod_cpu = janela_hann(Nelem, backend=np)
print(f"apod_cpu => {apod_cpu}")
print(f"Nelem => {Nelem}")

print(f"[INFO] Detecção automática:")
print(f"       Nang={Nang}, Nelem={Nelem}, Nsamples={Nsamples} (fs={fs/1e6} MHz, c={c:.1f} m/s)")
if mod is not None:
    print(f"       f_c={mod/1e6} MHz (modulation_frequency)")
print(f"       Grade: nx={nx}, nz={nz}, z_max={z_max*1e3:.1f} mm  | pitch={PITCH_M*1e3:.3f} mm")

# --- mover constantes para GPU se disponível ---
x_elem_xp  = xp.asarray(x_elem)            # (Nelem,)
grade_x    = xp.asarray(grade_x_cpu)       # (nx,)
grade_z    = xp.asarray(grade_z_cpu)       # (nz,)
apod       = xp.asarray(apod_cpu)          # (Nelem,)
ang_rad_xp = xp.asarray(ang_rad)           # (Nang,)

# Pré-cálculos independentes de z/ângulo:
xdiff = grade_x[None, :] - x_elem_xp[:, None]                  # (Nelem, nx)
zz, xx = xp.meshgrid(grade_z, grade_x, indexing="ij")          # (nz, nx)

# imagem complexa acumulada (compounding coerente)
img_complex_total = xp.zeros((nz, nx), dtype=xp.complex64)

# ---- loop de compounding por ângulo ----
for ia in tqdm(range(Nang), desc="Beamforming (ângulos)"):
    # traços desse ângulo -> GPU/CPU
    rf_angle = xp.asarray(rf_arr[ia, :, :])                    # (Nelem, Nsamples)
    # sinal analítico (por canal)
    rf_analytic = hilbert_xp(rf_angle, axis=1).astype(xp.complex64)

    theta = float(ang_rad[ia])
    sin_t = xp.float32(np.sin(theta));  cos_t = xp.float32(np.cos(theta))

    # t_tx para toda a malha (nz, nx): plane-wave
    t_tx = (zz * cos_t + xx * sin_t) / c                       # (nz, nx)

    img_complex = xp.zeros((nz, nx), dtype=xp.complex64)

    # blocos em z para limitar memória
    Bz_eff = int(min(Bz, nz)) if Bz is not None else nz
    for z0 in range(0, nz, Bz_eff):
        z1 = min(z0 + Bz_eff, nz)
        B  = z1 - z0

        zz_blk   = zz[z0:z1, :]                                # (B, nx)
        t_tx_blk = t_tx[z0:z1, :]                              # (B, nx)

        # distâncias RX: (Nelem, B, nx)
        dist_blk = xp.sqrt(xdiff[:, None, :]**2 + zz_blk[None, :, :]**2)
        # índices fracionários
        nidx_blk = (T0 + t_tx_blk[None, :, :] + dist_blk / c) * fs     # (Nelem,B,nx)

        # achata para (Nelem, B*nx) para usar a mesma interp
        nidx_flat = nidx_blk.reshape(Nelem, -1)

        vals_flat = interp1_complex_multi(rf_analytic, nidx_flat, xp)   # (Nelem, B*nx)
        vals_blk  = vals_flat.reshape(Nelem, B, nx)

        soma_blk  = xp.sum(apod[:, None, None].astype(vals_blk.dtype) * vals_blk, axis=0)  # (B, nx)
        img_complex[z0:z1, :] = soma_blk

    img_complex_total += img_complex

# envelope + log
faixa_dB = 60
env = xp.abs(img_complex_total)
env /= (env.max() + 1e-12)
img_db = 20.0 * xp.log10(env + 1e-12)
img_db = xp.clip(img_db, -faixa_dB, 0)

# trazer para CPU e exibir
img_db_cpu   = to_cpu(img_db)
grade_x_cpu  = to_cpu(grade_x)
grade_z_cpu  = to_cpu(grade_z)

print(f"[INFO] imagem reconstruída => {img_db_cpu.shape}")
print(f"[INFO] Parâmetros de grade: nx = {nx}, nz = {nz}")
print(f"[INFO] Dimensões: {nz} (profundidade) × {nx} (lateral)")

extent_mm = [grade_x_cpu[0]*1e3, grade_x_cpu[-1]*1e3,
             grade_z_cpu[-1]*1e3, grade_z_cpu[0]*1e3]

plt.figure(figsize=(6,8))
plt.imshow(img_db_cpu, cmap="gray", extent=extent_mm,
           aspect="auto", vmin=-faixa_dB, vmax=0)
plt.xlabel("Lateral (mm)")
plt.ylabel("Profundidade (mm)")
plt.title(f"B-mode DAS + Compounding ({Nang} ângulos) [CUDA {'ON' if xp is not np else 'OFF'}]")
plt.colorbar(label="dB")
plt.tight_layout()
plt.show()
