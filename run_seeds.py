import subprocess, os, sys, pathlib

REPO = "/content/Replication_DCCA"
RNA  = f"{REPO}/data/PBMC/RNA.h5ad"
ATAC = f"{REPO}/data/PBMC/ATAC.h5ad"
LOGS = pathlib.Path("/content/logs"); LOGS.mkdir(exist_ok=True)

SEEDS, N_EPOCHS = [0, 1, 2, 3, 4], 400
# DCCA khong co --n_warmup: anneal_epoch bi gan cung = 10 trong fit()
# (DCCA/MVAE_cycleVAE.py:162), truyen vao cung khong co tac dung.

assert os.path.exists(RNA) and os.path.exists(ATAC), "Sai duong dan .h5ad"

# OMP/MKL=1: ep reduction da luong ve thu tu co dinh (sklearn KMeans + torch CPU).
# CUBLAS_WORKSPACE_CONFIG: bat buoc de torch.use_deterministic_algorithms co hieu luc tren GPU.
env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
           CUBLAS_WORKSPACE_CONFIG=":4096:8")

def run(seed, tag=None, epochs=N_EPOCHS):
    tag = tag or f"seed{seed}"
    log = LOGS / f"{tag}.log"
    print(f"\n{'='*64}\n{tag}  (seed={seed})  ->  {log}\n{'='*64}", flush=True)
    cmd = [sys.executable, "-u", f"{REPO}/Main_SNARE_seq.py",
           "--File1", RNA, "--File2", ATAC,
           "--seed", str(seed),
           "--max_epoch", str(epochs),
           "--outdir", f"/content/output/{tag}"]
    with open(log, "w") as fh:
        p = subprocess.Popen(cmd, cwd=REPO, env=env, text=True, bufsize=1,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL)   # DEVNULL = giau thanh tqdm
        for line in p.stdout:
            fh.write(line)
            if any(k in line for k in ("ARI", "NMI", "[seed]", "converged", "Finish training")):
                print(line.rstrip(), flush=True)
        p.wait()
    print(f"[{tag}] exit code = {p.returncode}", flush=True)
    if p.returncode != 0:
        print(f"  -> xem log day du: !tail -40 {log}", flush=True)
    return p.returncode

for s in SEEDS:
    run(s)
