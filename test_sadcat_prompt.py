import torch
import sqlite3
import pandas as pd
from pathlib import Path
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

DEVICE = "cuda"
DTYPE = torch.float16
BASE_DIR = Path("/home/agrupa-lab/agrupa")
DB_PATH = BASE_DIR / "agrupa.sqlite"
OUTPUT_DIR = Path("/home/agrupa-lab/agrupa/IE_capstones/Omar/outputs")

# Load 20 sample artworks
conn = sqlite3.connect(DB_PATH)
query = """
    SELECT a.cat_no, a.titulo, a.autor, a.is_fauna, a.is_religious, a.century,
           i.file_path
    FROM artwork a
    INNER JOIN artwork_image i ON a.cat_no = i.cat_no
    WHERE substr(a.cat_no, 1, 1) = 'P'
    ORDER BY RANDOM()
    LIMIT 20
"""
df = pd.read_sql(query, conn)
conn.close()

df['image_exists'] = df['file_path'].apply(lambda x: Path(x).exists())
df = df[df['image_exists']].reset_index(drop=True)
print(f"Sample artworks: {len(df)}")

# Load LLaVA
print("Loading LLaVA...")
MODEL_NAME = "llava-hf/llava-v1.6-mistral-7b-hf"
processor = LlavaNextProcessor.from_pretrained(MODEL_NAME)
model = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True
).to(DEVICE)
model.eval()
print("Model loaded!")

PROMPT = """Describe the social characteristics of the people or animals depicted in this artwork. 
Focus on their personality traits, emotional warmth, friendliness, trustworthiness, 
competence, capability, status, and dominance. What social roles do they occupy? 
How do they relate to others around them?"""

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

# Generate descriptions
results = []
for _, row in df.iterrows():
    print(f"Processing: {row['titulo']}")
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
    print(f"Done: {desc[:100]}...")

output_file = OUTPUT_DIR / "llava_sadcat_test.csv"
pd.DataFrame(results).to_csv(output_file, index=False)
print(f"\nSaved to {output_file}")