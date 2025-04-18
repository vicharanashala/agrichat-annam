import torch
from sentence_transformers import SentenceTransformer, util

df = data.copy()

df['qa_combined'] = df['questions'].str.lower().str.strip() + ' ' + df['answers'].str.lower().str.strip()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SentenceTransformer('all-MiniLM-L6-v2').to(device)

print(f"Using device: {device}")
print(f"Model device: {next(model.parameters()).device}")



embeddings = model.encode(
    df['qa_combined'],
    convert_to_tensor=True,
    device=device,
    show_progress_bar=True,
    batch_size=512,
)
cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)
print(f"Embeddings device: {embeddings.device}")
to_drop = set()
for i in range(len(df)):
    if i in to_drop:
        continue
    for j in range(i+1, len(df)):
        if cosine_scores[i][j] > 0.95:
            to_drop.add(j)

df_dedup = df.drop(index=list(to_drop)).reset_index(drop=True)

print(f"Original rows: {len(df)}")
print(f"Rows after deduplication: {len(df_dedup)}")