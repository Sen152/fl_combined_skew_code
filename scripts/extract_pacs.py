"""从parquet文件提取PACS图片到目录结构"""
import pyarrow.parquet as pq
import os
from PIL import Image
import io

PACS_CLASSES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']

import argparse

_ap = argparse.ArgumentParser(description='Extract the PACS image files from the distributed parquet shard.')
_ap.add_argument('--parquet', required=True, help='path to train-00000-of-00001.parquet')
_ap.add_argument('--out', default='./data/PACS', help='output root (default: ./data/PACS)')
_args = _ap.parse_args()

parquet_path = _args.parquet
output_root = _args.out

print(f"Reading parquet: {parquet_path}")
table = pq.read_table(parquet_path)
print(f"Columns: {table.column_names}")
print(f"Rows: {len(table)}")

# 查看第一行
for col in table.column_names:
    val = table[col][0].as_py()
    if col == 'image':
        print(f"  {col}: type={type(val)}")
        if isinstance(val, dict):
            print(f"    keys: {list(val.keys())}")
            if 'bytes' in val:
                print(f"    bytes length: {len(val['bytes'])}")
    else:
        print(f"  {col}: {val}")

# 创建目录结构
for domain in ['art_painting', 'cartoon', 'photo', 'sketch']:
    for cls in PACS_CLASSES:
        os.makedirs(os.path.join(output_root, domain, cls), exist_ok=True)

# 提取图片
count = 0
for i in range(len(table)):
    img_data = table['image'][i].as_py()
    label = table['label'][i].as_py()
    domain = table['domain'][i].as_py()

    class_name = PACS_CLASSES[label]

    # 解码图片
    if isinstance(img_data, dict) and 'bytes' in img_data:
        img_bytes = img_data['bytes']
    elif isinstance(img_data, bytes):
        img_bytes = img_data
    else:
        print(f"  Skipping row {i}: unexpected image format {type(img_data)}")
        continue

    img = Image.open(io.BytesIO(img_bytes))
    save_path = os.path.join(output_root, domain, class_name, f"{i:05d}.jpg")
    img.save(save_path)
    count += 1

    if count % 1000 == 0:
        print(f"  Saved {count} images...")

print(f"\nDone! Saved {count} images to {output_root}")
