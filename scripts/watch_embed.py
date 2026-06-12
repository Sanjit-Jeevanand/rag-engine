import sqlite3
import time

DB = "data/docs.db"

conn = sqlite3.connect(DB)
total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
start_count = conn.execute(
    "SELECT COUNT(*) FROM documents WHERE status='embedded'"
).fetchone()[0]
start_time = time.time()

print(f"Total chunks: {total:,}")
print(f"{'Embedded':>12}  {'Remaining':>12}  {'Speed':>10}  {'ETA':>12}")
print("-" * 56)

while True:
    count = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE status='embedded'"
    ).fetchone()[0]
    elapsed = time.time() - start_time
    done = count - start_count
    speed = done / elapsed if elapsed > 1 else 0
    remaining = total - count
    if speed > 0:
        s = int(remaining / speed)
        eta_str = f"{s // 3600}h {s % 3600 // 60:02d}m"
    else:
        eta_str = "—"

    print(
        f"{count:>12,}  {remaining:>12,}  {speed:>8.0f}/s  {eta_str:>12}",
        end="\r",
        flush=True,
    )
    time.sleep(5)
