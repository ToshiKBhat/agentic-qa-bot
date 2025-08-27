from pathlib import Path
import sys
import pandas as pd

BASE = Path("data/parquet").resolve()
BASE.mkdir(parents=True, exist_ok=True)

TRIPS_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2021-01.parquet"
ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi+_zone_lookup.csv"

try:
    import urllib.request as U
except Exception:
    print("Python stdlib urllib not available", file=sys.stderr)
    sys.exit(1)

trips_pq = BASE / "yellow_tripdata_2021-01.parquet"
zones_csv = BASE / "taxi_zone_lookup.csv"
zones_pq = BASE / "taxi_zone_lookup.parquet"

if not trips_pq.exists():
    print(f"Downloading trips → {trips_pq}")
    U.urlretrieve(TRIPS_URL, trips_pq)
else:
    print("Trips parquet already present.")

if not zones_csv.exists():
    print(f"Downloading zones CSV → {zones_csv}")
    U.urlretrieve(ZONES_URL, zones_csv)
else:
    print("Zones CSV already present.")

print(f"Converting zones CSV → Parquet: {zones_pq}")
df = pd.read_csv(zones_csv)
df.to_parquet(zones_pq, index=False)
print("Done.")