from preference_datasets import get_dataset, get_batch_iterator
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2')
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

train_iterator = get_batch_iterator(names=['kub','sshf'],
                                    tokenizer=tokenizer,
                                    split='train', 
                                    n_epochs=1, 
                                    n_examples=100, 
                                    batch_size=32, 
                                    silent=False, 
                                    cache_dir='./dataset')

for batch in train_iterator:
    breakpoint()