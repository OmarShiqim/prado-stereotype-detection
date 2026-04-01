import torch
import sqlite3
import pandas as pd
from pathlib import Path
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import time
from tqdm import tqdm

DEVICE = "cuda"
DTYPE = torch.float16
BASE_DIR = Path("/home/agrupa-lab/agrupa")
DB_PATH = BASE_DIR / "agrupa.sqlite"
OUTPUT_DIR = Path("/home/agrupa-lab/agrupa/IE_capstones/Omar/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "llava_descriptions.csv"

print(f"Device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.1f} GB")

conn = sqlite3.connect(DB_PATH)
query = """
    SELECT a.cat_no, a.titulo, a.autor, a.is_fauna, a.is_religious, a.century,
           i.file_path
    FROM artwork a
    INNER JOIN artwork_image i ON a.cat_no = i.cat_no
    WHERE substr(a.cat_no, 1, 1) = 'P'
"""
df = pd.read_sql(query, conn)
conn.close()

df['image_exists'] = df['file_path'].apply(lambda x: Path(x).exists())
df_ready = df[df['image_exists']].reset_index(drop=True)
print(f"Total artworks ready: {len(df_ready)}")

if OUTPUT_FILE.exists():
    existing = pd.read_csv(OUTPUT_FILE)
    processed_ids = set(existing['cat_no'].tolist())
    df_ready = df_ready[~df_ready['cat_no'].isin(processed_ids)].reset_index(drop=True)
    results = existing.to_dict('records')
    print(f"Resuming: {len(processed_ids)} already done")
else:
    results = []

print(f"Remaining: {len(df_ready)}")

print("Loading LLaVA model onto GPU...")
MODEL_NAME = "llava-hf/llava-v1.6-mistral-7b-hf"
processor = LlavaNextProcessor.from_pretrained(MODEL_NAME)
model = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True
).to(DEVICE)
model.eval()

print(f"Model loaded! VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

PROMPT = "Describe this artwork in detail, including the people or animals depicted, their appearance, actions, social roles, and the overall mood of the scene."

def generate_description(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        formatted_prompt = f"[INST] <image>\n{PROMPT} [/INST]"
        inputs = processor(
            text=formatted_prompt,
            images=image,
            return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False
            )
        full_text = processor.decode(output[0], skip_special_tokens=True)
        description = full_text.split("[/INST]")[-1].strip()
        return description
    except Exception as e:
        return f"ERROR: {str(e)}"

print("Starting pipeline...")
start_time = time.time()

for idx, row in tqdm(df_ready.iterrows(), total=len(df_ready)):
    desc = generate_description(row['file_path'])
    results.append({
        'cat_no': row['cat_no'],
        'titulo': row['titulo'],
        'autor': row['autor'],
        'is_fauna': row['is_fauna'],
        'is_religious': row['is_religious'],
        'century': row['century'],
        'llava_description': desc,
        'timestamp': pd.Timestamp.now().isoformat()
    })

    if len(results) % 50 == 0:
        pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
        elapsed = time.time() - start_time
        print(f"Checkpoint: {len(results)} done in {elapsed/3600:.1f}h")

pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
elapsed = time.time() - start_time
print(f"Done! {len(results)} saved in {elapsed/3600:.1f} hours")
