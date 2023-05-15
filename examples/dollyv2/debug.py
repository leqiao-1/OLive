import torch
from transformers import pipeline

generate_text = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.float16, trust_remote_code=True, device_map="auto")
res = generate_text("Explain to me the difference between nuclear fission and fusion.")
print(res[0]["generated_text"])
