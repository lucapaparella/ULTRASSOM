# ================================
# DAS (plane-wave) + Compounding coerente (CUDA-ready) — versão didática
# ================================
import h5py, numpy as np, matplotlib.pyplot as plt
from tqdm.auto import tqdm

# ---------- Backend: CUDA (CuPy) se disponível; senão NumPy ----------
USE_CUDA = True
xp = np
to_cpu = lambda a: a
hilbert_xp = None
_is_cupy = False  # flag útil para pequenos "ifs"

if USE_CUDA:
    try:
        import cupy as cp
        from cupyx.scipy.signal import hilbert as c_hilbert
        if cp.cuda.runtime.getDeviceCount() > 0:
            xp = cp
            hilbert_xp = c_hilbert
            to_cpu = cp.asnumpy
            _is_cupy = True
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
path = r"C:\Users\lucap\Documents\CUBDL_Data\CUBDL_Data\1_CUBDL_Task1_Data\OSL010\OSL010.hdf5"

nx_default, nz_default = 128, 512    # usado só como fallback
num_angulos_usados = None            # None -> usar todos; ou um int (ex.: 9, 13, ...)
faixa_dB = 60                        # faixa dinâmica para plot
Bz = 64                              # tamanho do bloco em z (ajuste conforme memória)

# -------- utilidades --------
def janela_hann(n, backend=np):
    """Apodização Hann por elemento (reduz lóbulos laterais)."""
    n0 = backend.arange(n, dtype=backend.float32)
    return (0.5 - 0.5*backend.cos(2*backend.pi*n0/(n-1))).astype(backend.float32)

def _any(x):
    """any() que funciona para NumPy e CuPy (retorna bool Python)."""
    try:
        return bool(x.any())
    except Exception:
        # Cupy 0-dim para Python
        return bool(to_cpu(x).any())

def interp1_complex_multi(rf_ch, n_float, xp):
    """
    Interpolação linear vetorizada por canal, preservando fase.
    rf_ch:   (Nelem, Nsamples) complexo - traço analítico por elemento
    n_float: (Nelem, M) float32  - índices fracionários (por elemento e por coluna/posição)
    Retorna: (Nelem, M) complexo
    """
    n_float = n_float.astype(xp.float32)
    n0 = xp.floor(n_float).astype(xp.int64)   # vizinho inferior
    n1 = n0 + 1                               # vizinho superior
    w  = n_float - n0                         # peso fracionário
    Nelem, Nsamples = rf_ch.shape

    valid = (n0 >= 0) & (n1 < Nsamples)       # máscara para não extrapolar
    out = xp.zeros_like(n_float, dtype=rf_ch.dtype)

    if _any(valid):  # robusto para numpy/cupy
        row = xp.arange(Nelem, dtype=xp.int64)[:, None]  # (Nelem,1) broadcast nas colunas
        # amostras válidas real/imag
        re0 = rf_ch.real[row, n0][valid]; re1 = rf_ch.real[row, n1][valid]
        im0 = rf_ch.imag[row, n0][valid]; im1 = rf_ch.imag[row, n1][valid]
        # interp. linear separada em real/imag evita rotação de fase espúria
        out_real = (1.0 - w[valid]) * re0 + w[valid] * re1
        out_imag = (1.0 - w[valid]) * im0 + w[valid] * im1
        out.real[valid] = out_real
        out.imag[valid] = out_imag
    return out

# ======== carregar dados (CPU) ========
with h5py.File(path, "r") as f:
    # ----- Sinais RF por ângulo e elemento -----
    # Esperado: (Nang, Nelem, Nsamples) ou (Nelem, Nsamples)
    rf = f["/channel_data"][()]
    rf_arr = np.array(rf)  # CPU

    if rf_arr.ndim == 2:
        # Se vier como (Nelem, Nsamples), adiciona eixo de ângulo
        rf_arr = rf_arr[np.newaxis, ...]  # (1, Nelem, Nsamples)
    elif rf_arr.ndim != 3:
        raise RuntimeError(f"RF shape inesperado: {rf_arr.shape}")
    Nang_total, Nelem, Nsamples = rf_arr.shape

    # ----- Parâmetros físicos -----
    fs = float(np.array(f["/sampling_frequency"][()]).ravel()[0])       # Hz
    c  = float(np.array(f["/sound_speed"][()]).ravel()[0]) if "/sound_speed" in f else 1540.0 # m/s

    # `start_time` por ângulo (shape típico (1, 75)). Se não houver, tenta time_zero.
    if "/start_time" in f:
        t0_vec = np.array(f["/start_time"][()]).reshape(-1)  # flatten
        # se for (1, Nang) vira (Nang,)
        if t0_vec.size == Nang_total:
            pass
        elif t0_vec.size == 1:
            t0_vec = np.full((Nang_total,), float(t0_vec[0]), dtype=float)
        else:
            raise RuntimeError(f"start_time shape inesperado para {Nang_total} ângulos: {t0_vec.shape}")
    elif "/time_zero" in f:
        t0_scalar = float(np.array(f["/time_zero"][()]).ravel()[0])
        t0_vec = np.full((Nang_total,), t0_scalar, dtype=float)
    else:
        # fallback conservador
        t0_vec = np.zeros((Nang_total,), dtype=float)

    # frequencia central (para correção de fase TX)
    f0 = None
    if "/modulation_frequency" in f:
        f0 = float(np.array(f["/modulation_frequency"][()]).ravel()[0])

    # ----- Geometria do array -----
    ele_pos = f["/element_positions"][()].T  # (Nelem,3) após .T
    ele = np.array(ele_pos, dtype=float)     # CPU
    x_elem = ele[:, 0] if (ele.ndim == 2 and ele.shape[1] >= 1) else ele.squeeze()
    # Heurística de unidade: se range muito grande, pode estar em mm
    if (np.max(x_elem) - np.min(x_elem)) > 1.0:  # > 1 m? provavelmente estava em mm
        x_elem = x_elem / 1000.0

    # ----- Ângulos / direção de transmissão -----
    # Preferir transmit_direction (2,Nang): [sinθ; cosθ]
    sincos = None
    if "/transmit_direction" in f:
        td = np.array(f["/transmit_direction"][()])  # (2, Nang)
        if td.shape[0] == 2 and td.shape[1] == Nang_total:
            sincos = td.astype(float)
        else:
            # Alguns datasets podem vir transpostos
            if td.shape[1] == 2 and td.shape[0] == Nang_total:
                sincos = td.T.astype(float)
            else:
                raise RuntimeError(f"transmit_direction shape inesperado: {td.shape}")

    # Se não houver transmit_direction, tenta ângulos em rad/graus
    ang = None
    if sincos is None:
        for k in ("/angles_rad", "/angles", "/tx_angles", "/angles_deg"):
            if k in f:
                ang = np.array(f[k][()]).ravel().astype(float)
                break
        if ang is None or ang.size == 0:
            ang = np.zeros((Nang_total,), dtype=float)  # fallback: todos 0
        # graus -> rad
        if np.max(np.abs(ang)) > 2*np.pi:
            ang_rad = np.deg2rad(ang)
        else:
            ang_rad = ang
        sincos = np.vstack([np.sin(ang_rad), np.cos(ang_rad)])  # (2,Nang)

    # ----- Grade de pixels (x,z) -----
    # Se existir pixel_positions (3, nx, nz), use como verdade de escala
    have_pix = "/pixel_positions" in f
    if have_pix:
        pix = np.array(f["/pixel_positions"][()])  # (3, nx, nz)
        # Convenção típica: eixo 0 -> x, eixo 2 -> z
        X = pix[0, :, :]  # (nx, nz)
        Z = pix[2, :, :]  # (nx, nz)
        x_line = X[:, 0].astype(np.float32)        # (nx,)
        z_line = Z[0, :].astype(np.float32)        # (nz,)
        # checagem de unidade rudimentar (se valores parecem mm, converte p/ m)
        if (x_line.max() - x_line.min()) > 1.0: x_line = x_line / 1000.0
        if (z_line.max() - z_line.min()) > 1.0: z_line = z_line / 1000.0
        nx, nz = x_line.size, z_line.size
    else:
        # Fallback: lateral cobre a largura do array; profundidade pelo tempo máximo
        nx, nz = nx_default, nz_default
        x_min, x_max = float(x_elem.min()), float(x_elem.max())
        x_line = np.linspace(x_min, x_max, nx, dtype=np.float32)
        # limite de profundidade aproximado pelo tempo máximo de recepção
        tmax = t0_vec.max() + (Nsamples - 1) / fs
        z_max = c * tmax / 2.0
        z_line = np.linspace(0.0, z_max, nz, dtype=np.float32)

# --------- preparar constantes (GPU se disponível) ----------
# apodização por elemento
apod_cpu = janela_hann(Nelem, backend=np)

# seleção de ângulos (uniforme)
idx_ang = np.arange(Nang_total)
if num_angulos_usados is not None and num_angulos_usados < Nang_total:
    passo = max(1, Nang_total // num_angulos_usados)
    idx_ang = idx_ang[::passo][:num_angulos_usados]
print(f"[INFO] Compounding coerente com {idx_ang.size} de {Nang_total} ângulos.")

# mover p/ GPU se houver
x_elem_xp  = xp.asarray(x_elem)          # (Nelem,)
grade_x    = xp.asarray(x_line)          # (nx,)
grade_z    = xp.asarray(z_line)          # (nz,)
apod       = xp.asarray(apod_cpu)        # (Nelem,)
sint_all   = xp.asarray(sincos[0, :])    # (Nang,)
cost_all   = xp.asarray(sincos[1, :])    # (Nang,)
t0_all     = xp.asarray(t0_vec)          # (Nang,)
f0_val     = None if f0 is None else float(f0)

# Pré-cálculos que não dependem de z/ângulo
xdiff = grade_x[None, :] - x_elem_xp[:, None]  # (Nelem, nx)

# Grade (z,x)
zz, xx = xp.meshgrid(grade_z, grade_x, indexing="ij")  # (nz, nx) cada

# imagem complexa acumulada (compounding coerente)
img_complex_total = xp.zeros((grade_z.size, grade_x.size), dtype=xp.complex64)

# ========= LOOP DE ÂNGULOS =========
for ia in tqdm(idx_ang, desc="Beamforming (ângulos)"):
    # --- dados do ângulo na GPU ---
    rf_angle = xp.asarray(rf_arr[ia, :, :])                 # (Nelem, Nsamples) (real)
    rf_analytic = hilbert_xp(rf_angle, axis=1).astype(xp.complex64)  # analítico no eixo das amostras

    # seno/cosseno e t0 deste ângulo (float32 p/ economia)
    
    sin_t = sint_all[ia].astype(xp.float32)   # 0-D array CuPy/NumPy ok
    cos_t = cost_all[ia].astype(xp.float32)
    t0_a  = t0_all[ia].astype(xp.float32)


    # tempo de TX em toda a malha: t_tx(z,x) = (z*cosθ + x*sinθ) / c
    t_tx = (zz * cos_t + xx * sin_t) / c  # (nz, nx)

    # imagem para este ângulo
    img_complex = xp.zeros_like(img_complex_total)

    # -------- processamento por blocos em z (economiza memória) --------
    Bz_eff = int(min(Bz, grade_z.size)) if Bz is not None else grade_z.size
    for z0 in range(0, grade_z.size, Bz_eff):
        z1 = min(z0 + Bz_eff, grade_z.size)
        B = z1 - z0

        # fatias do bloco
        zz_blk   = zz[z0:z1, :]            # (B, nx)   (somente para clareza)
        t_tx_blk = t_tx[z0:z1, :]          # (B, nx)

        # distâncias RX por canal (broadcasting):
        # xdiff:(Nelem,nx) -> (Nelem,1,nx); zz_blk:(B,nx) -> (1,B,nx)
        dist_blk = xp.sqrt(xdiff[:, None, :]**2 + zz_blk[None, :, :]**2)      # (Nelem,B,nx)

        # índices fracionários n = (t0 + t_tx + t_rx)*fs
        nidx_blk = (t0_a + t_tx_blk[None, :, :] + dist_blk / c) * fs          # (Nelem,B,nx)

        # achatamos para (Nelem, B*nx) para reaproveitar a função de interp.
        nidx_flat = nidx_blk.reshape(Nelem, -1)

        # interpola por canal, preservando fase (continua vetorizado em canais)
        vals_flat = interp1_complex_multi(rf_analytic, nidx_flat, xp)          # (Nelem, B*nx)

        # volta ao shape do bloco
        vals_blk = vals_flat.reshape(Nelem, B, grade_x.size)                   # (Nelem,B,nx)

        # (Opcional, mas MUITO recomendado) Correção de fase de TX para compounding coerente:
        # multiplicar por exp(-j*2π f0 * t_tx) remove a fase de transmissão,
        # alinhando os ângulos antes da soma.
        if f0_val is not None:
            phase_tx = xp.exp(-1j * (2.0 * np.pi * f0_val) * t_tx_blk)[None, :, :]  # (1,B,nx)
            vals_blk = vals_blk * phase_tx

        # soma coerente por elementos (com apodização)
        soma_blk = xp.sum(apod[:, None, None].astype(vals_blk.dtype) * vals_blk, axis=0)  # (B,nx)

        # grava no volume da imagem deste ângulo
        img_complex[z0:z1, :] = soma_blk

    # soma coerente entre ângulos
    img_complex_total += img_complex

# ====== envelope + log-compression ======
env = xp.abs(img_complex_total)
env /= (env.max() + 1e-12)
img_db = 20.0 * xp.log10(env + 1e-12)
img_db = xp.clip(img_db, -faixa_dB, 0)

# ====== trazer para CPU e plotar ======
img_db_cpu = to_cpu(img_db)
x_cpu = to_cpu(grade_x) * 1e3  # mm
z_cpu = to_cpu(grade_z) * 1e3  # mm

print(f"[INFO] imagem reconstruída => {img_db_cpu.shape} (nz,nx)")

plt.figure(figsize=(6,8))
# origin="lower": profundidade aumenta para baixo no gráfico, coerente com z>=0
plt.imshow(img_db_cpu, cmap="gray",
           extent=[x_cpu[0], x_cpu[-1], z_cpu[0], z_cpu[-1]],
           origin="lower", aspect="auto", vmin=-faixa_dB, vmax=0)
plt.xlabel("Lateral x (mm)")
plt.ylabel("Profundidade z (mm)")
plt.title(f"B-mode DAS + Compounding ({idx_ang.size} ângulos) [CUDA {'ON' if _is_cupy else 'OFF'}]")
plt.colorbar(label="dB")
plt.tight_layout()
plt.show()
