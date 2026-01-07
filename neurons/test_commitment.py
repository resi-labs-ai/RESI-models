# The commitment data hex from the chain (the long hex string from your extrinsic)
# Remove any leading prefix bytes (like the 'q' / 0x71 at the start)
raw_hex = "37623232363832323361323233323338333133323631333933393336323232633232373232323361323234643635363537303433366636343639366536373332333032663734363537333734333132323263323237343232336133313337333633373337333133373332333733323764"

# Step 1: Decode outer hex → get inner hex string
inner_hex = bytes.fromhex(raw_hex).decode('utf-8')
print(f"Inner hex: {inner_hex}")

# Step 2: Decode inner hex → get JSON
import json
commitment_json = bytes.fromhex(inner_hex).decode('utf-8')
commitment = json.loads(commitment_json)

print(f"\n✅ Decoded commitment:")
print(f"   Hash (h):      {commitment['h']}")
print(f"   Repo (r):      {commitment['r']}")
print(f"   Timestamp (t): {commitment['t']}")

# Verify repo matches
expected_repo = "MeepCoding20/test1"
if commitment['r'] == expected_repo:
    print(f"\n🎉 Repo matches: {expected_repo}")
else:
    print(f"\n❌ Repo mismatch! Expected {expected_repo}, got {commitment['r']}")