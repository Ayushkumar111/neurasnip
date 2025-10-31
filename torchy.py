"""
Find where torch is actually being imported from
"""
import sys

print("🔍 Python Search Path (in order):")
print("=" * 60)
for i, path in enumerate(sys.path, 1):
    print(f"{i}. {path}")

print("\n" + "=" * 60)

try:
    import torch
    print(f"\n✅ Torch found at:")
    print(f"   {torch.__file__}")
    
    if "AppData\\Roaming" in torch.__file__:
        print("\n❌ ERROR: Importing from user directory!")
        print("   This is WRONG - should be from venv!")
    elif "venv" in torch.__file__:
        print("\n✅ CORRECT: Importing from venv!")
    
except ImportError as e:
    print(f"\n❌ Torch not found: {e}")