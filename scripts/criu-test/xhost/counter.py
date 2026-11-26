import hashlib, os, sys, time
total = int(sys.argv[1]); sleep = float(sys.argv[2])
state = hashlib.sha256(b"hibernus").digest()      # lives only in this process's memory
big = bytearray(os.urandom(1) * (64 * 1024 * 1024))  # 64 MB of memory to carry across
for step in range(1, total + 1):
    state = hashlib.sha256(state).digest()
    print(f"step {step} pid {os.getpid()} state {state.hex()[:16]} host {os.uname().nodename}", flush=True)
    time.sleep(sleep)
