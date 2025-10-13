# ================================
# DAS (plane-wave) + Compounding coerente
# ================================
import h5py, numpy as np, matplotlib.pyplot as plt
from scipy.signal import hilbert
from tqdm.auto import tqdm

# -------- parâmetros do usuário --------
path = r"C:\Users\lucap\Documents\CUBDL\OSL010.hdf5"
nx, nz = 128, 512                 # resolução da imagem
num_angulos_usados = 1        # None -> usar todos; ou um int (ex.: 9, 13, ...)
faixa_dB = 60                     # faixa dinâmica para plot

# -------- utilidades --------
def janela_hann(n):
    n0 = np.arange(n, dtype=np.float32)
    return (0.5 - 0.5*np.cos(2*np.pi*n0/(n-1))).astype(np.float32)

def interp1_complex(traco_complexo, n_float):
    """
    Interpolação linear em um traço complexo (1D).
    n_float pode ser escalar ou array; retorna array complex64 do mesmo shape.
    """
    n_float = np.asarray(n_float, dtype=np.float32)
    n0 = np.floor(n_float).astype(np.int64)
    n1 = n0 + 1
    w  = n_float - n0
    valid = (n0 >= 0) & (n1 < traco_complexo.shape[0])

    out = np.zeros_like(n_float, dtype=np.complex64)
    if np.any(valid):
        re0 = traco_complexo.real[n0[valid]]
        re1 = traco_complexo.real[n1[valid]]
        im0 = traco_complexo.imag[n0[valid]]
        im1 = traco_complexo.imag[n1[valid]]
        out.real[valid] = (1.0 - w[valid]) * re0 + w[valid] * re1
        out.imag[valid] = (1.0 - w[valid]) * im0 + w[valid] * im1
    return out

# -------- carregar dados --------
with h5py.File(path, "r") as f:
    rf = f["/channel_data"][()]                  # (Nang,Nelem,Nsamples) ou (Nelem,Nsamples)
    ele_pos = f["/element_positions"][()].T      # tipicamente (Nelem,3) após .T
    t0 = float(np.array(f["/start_time"][()]).ravel()[0])
    fs = float(np.array(f["/sampling_frequency"][()]).ravel()[0])
    # opcionais
    c = float(np.array(f["/sound_speed"][()]).ravel()[0]) if "/sound_speed" in f else 1540.0

    # ângulos (vários nomes possíveis)
    ang = None
    for k in ("/transmit_direction", "/angles", "/tx_angles", "/angles_deg", "/angles_rad"):
        if k in f:
            ang = np.array(f[k][()]).ravel()
            break

# fallbacks
if ang is None or ang.size == 0:
    ang = np.array([0.0], dtype=float)  # broadside
# detectar graus/radianos
ang = ang.astype(float)
if np.max(np.abs(ang)) > 2*np.pi:   # provavelmente graus
    ang_rad = np.deg2rad(ang)
else:
    ang_rad = ang
ang_graus = np.rad2deg(ang_rad)

# normalizar RF para (Nang, Nelem, Nsamples)
rf_arr = np.array(rf)
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

# grade de imagem
x_min, x_max = float(x_elem.min()), float(x_elem.max())
grade_x = np.linspace(x_min, x_max, nx, dtype=np.float32)
z_max = c * (t0 + (Nsamples - 1) / fs) / 2.0
grade_z = np.linspace(0.0, z_max, nz, dtype=np.float32)

# apodização por elemento
apod = janela_hann(Nelem)

# escolher subconjunto de ângulos (uniforme)
idx_ang = np.arange(Nang_total)
if num_angulos_usados is not None and num_angulos_usados < Nang_total:
    passo = max(1, Nang_total // num_angulos_usados)
    idx_ang = idx_ang[::passo][:num_angulos_usados]

print(f"Compounding coerente com {idx_ang.size} de {Nang_total} ângulos.")

# imagem complexa acumulada (compounding coerente)
img_complex_total = np.zeros((nz, nx), dtype=np.complex64)

# ---- loop de compounding ----
for ia in tqdm(idx_ang, desc="Beamforming (ângulos)"):
    # traços desse ângulo: (Nelem, Nsamples)
    rf_angle = rf_arr[ia, :, :]

    # sinal analítico por canal (interp fracionária estável de fase)
    rf_analytic = hilbert(rf_angle, axis=1).astype(np.complex64)

    theta = float(ang_rad[ia]) if ia < ang_rad.size else float(ang_rad[0])
    sin_t, cos_t = np.sin(theta), np.cos(theta)

    # imagem para este ângulo
    img_complex = np.zeros((nz, nx), dtype=np.complex64)

    # varremos linhas de profundidade
    for iz, z in enumerate(grade_z):
        # t_tx(x) = (z*cos + x*sin)/c (pré-calcula para todos x da linha)
        t_tx_x = (z * cos_t + grade_x * sin_t) / c  # shape (nx,)
        # para cada coluna (x), somar sobre elementos
        for ix, x in enumerate(grade_x):
            t_tx = t_tx_x[ix]
            # somatório coerente sobre elementos
            # (poderia vetorizar, aqui mantemos claro/didático)
            soma = 0.0 + 0.0j
            for ch in range(Nelem):
                dist = np.sqrt((x - x_elem[ch])**2 + z**2)
                t_rx = dist / c
                n = (t0 + t_tx + t_rx) * fs  # índice fracionário
                soma += apod[ch] * interp1_complex(rf_analytic[ch], np.array([n], dtype=np.float32))[0]
            img_complex[iz, ix] = soma

    # soma coerente entre ângulos (compounding)
    img_complex_total += img_complex

# envelope + log
env = np.abs(img_complex_total)
env /= (env.max() + 1e-12)
img_db = 20.0 * np.log10(env + 1e-12)
img_db = np.clip(img_db, -faixa_dB, 0)

# plot em mm
extent_mm = [grade_x[0]*1e3, grade_x[-1]*1e3, grade_z[-1]*1e3, grade_z[0]*1e3]
plt.figure(figsize=(6,8))
plt.imshow(img_db, cmap="gray", extent=extent_mm, aspect="auto", vmin=-faixa_dB, vmax=0)
plt.xlabel("Lateral (mm)")
plt.ylabel("Profundidade (mm)")
plt.title(f"B-mode DAS + Compounding ({idx_ang.size} ângulos)")
plt.colorbar(label="dB")
plt.tight_layout()
plt.show()
