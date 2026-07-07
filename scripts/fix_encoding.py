import os
import glob

# Run from project root
root_dir = os.getcwd()

for filepath in glob.glob(f"{root_dir}/**/__init__.py", recursive=True):
    try:
        with open(filepath, 'rb') as f:
            raw = f.read()
        
        # Check for UTF-16 BOM (0xFF 0xFE)
        if raw.startswith(b'\xff\xfe'):
            # Decode as UTF-16, then re-encode as UTF-8
            content = raw.decode('utf-16')
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Fixed: {filepath}")
        else:
            print(f"⏭️ Skipped (already UTF-8): {filepath}")
    except Exception as e:
        print(f"❌ Error with {filepath}: {e}")