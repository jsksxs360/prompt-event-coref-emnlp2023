import re

from transformers import AutoConfig, AutoTokenizer
from tqdm.auto import tqdm
import json

a = "Thought Leader \u00bb Michael Trapido \u00bb McCain versus Putin:"
# a = sample['document']
char_start, char_end = 14068, 14070
print(a[char_start:char_end+1])

tokenizer = AutoTokenizer.from_pretrained('../../PT_MODELS/allenai/longformer-large-4096/')

encoding = tokenizer(a)

print(encoding)

token_start = encoding.char_to_token(char_start)
if not token_start:
    token_start = encoding.char_to_token(char_start + 1)
token_end = encoding.char_to_token(char_end)

print(token_start, token_end)
print(encoding.tokens())
