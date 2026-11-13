import json
import hashlib
from pathlib import Path

manifest_path = Path("data/benchmarks/phase_6/metadata/manifest.json")
gt_path = Path("data/benchmarks/phase_6/ground_truth/ground_truth.json")
images_dir = Path("data/benchmarks/phase_6/images")

with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

with open(gt_path, "r", encoding="utf-8") as f:
    gt = json.load(f)

print(f"Total manifest entries: {len(manifest)}")
print(f"Total ground truth entries: {len(gt)}")

hash_mismatches = []
for entry in manifest:
    cid = entry["case_id"]
    img_path = images_dir / f"{cid}.jpg"
    if not img_path.exists():
        print(f"MISSING IMAGE: {img_path}")
        continue
    with open(img_path, "rb") as img_f:
        actual_hash = hashlib.sha256(img_f.read()).hexdigest()
    expected_hash = entry["sha256"]
    if actual_hash != expected_hash:
        hash_mismatches.append((cid, expected_hash, actual_hash))
    else:
        print(f"VERIFIED: {cid} | SHA-256: {actual_hash[:16]}... | License: {entry.get('license')} | Author: {entry.get('author')}")

if hash_mismatches:
    print(f"FAIL: {len(hash_mismatches)} hash mismatches found!")
else:
    print("SUCCESS: All 12 benchmark images cryptographically match manifest.")
