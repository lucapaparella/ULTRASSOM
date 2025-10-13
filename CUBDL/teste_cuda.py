import cupy as cp, time

print("=== Teste GPU sem cuRAND ===")
print("Config CuPy:"); cp.show_config()

# Info de memória
free, total = cp.cuda.runtime.memGetInfo()
print(f"Memória livre/total: {free/1e9:.2f} GB / {total/1e9:.2f} GB")

N = 10_000_000

# Em vez de cp.random, use inicialização determinística (não chama cuRAND)
a = cp.linspace(0, 1, N, dtype=cp.float32)
b = cp.linspace(1, 2, N, dtype=cp.float32)

cp.cuda.Stream.null.synchronize()

t0 = time.time()
c = a + b
cp.cuda.Stream.null.synchronize()
t1 = time.time()

print(f"Soma (GPU) ok em {t1 - t0:.4f} s. c[0]={float(c[0]):.4f}, c[-1]={float(c[-1]):.4f}")

# Um GEMM para exercitar cuBLAS (também não usa cuRAND)
m = 2048
A = cp.arange(m*m, dtype=cp.float32).reshape(m, m) / m
B = cp.arange(m*m, dtype=cp.float32).reshape(m, m) / (m*2)

cp.cuda.Stream.null.synchronize()
t0 = time.time()
C = A @ B
cp.cuda.Stream.null.synchronize()
t1 = time.time()
print(f"MatMul 2048x2048 (cuBLAS) ok em {t1 - t0:.3f} s. C[0,0]={float(C[0,0]):.3f}")
print("=== Concluído ===")
