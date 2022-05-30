from transformers import RobertaTokenizer


def encode_batch(batch):
    global tokenizer
    tokenizer: RobertaTokenizer
    return tokenizer(batch["text"], padding="max_length", truncation=True)
