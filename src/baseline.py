from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
import torchtext
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchtext.data.metrics import bleu_score
import os
from datasets import load_dataset, load_from_disk

from tqdm import tqdm
import argparse

from main import test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', type = str, default = 'cuda')
    parser.add_argument('-batch_size', type = int, default= 8)
    parser.add_argument('-max_len', type = int, default = 128)
    parser.add_argument('-lang', type=str, help="pt, gl, ba", default='pt')
    args = parser.parse_args()
    parser.add_argument('-results', type=str, default=f"../data/baseline_predictions_{args.lang}.csv")
    args = parser.parse_args()

    torch.manual_seed(42)  # pytorch random seed
    np.random.seed(42)  # numpy random seed
    torch.backends.cudnn.deterministic = True
    
    tokenizer = AutoTokenizer.from_pretrained(os.environ["TOKENIZER_13B_PATH"],
                                          use_auth_token=True,)
    tokenizer.pad_token = tokenizer.eos_token

    model = LlamaForCausalLM.from_pretrained(os.environ['LLAMA_MODEL_13B_PATH'],
                                                use_safetensors=False,
                                                # device_map='auto',
                                                torch_dtype=torch.float16,
                                                use_auth_token=True,
                                                #  load_in_8bit=True,
                                                load_in_4bit=True
                                                )
    model.to(args.device)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    por_dataset = load_from_disk("../data/por.hf")
    eng_dataset = load_from_disk("../data/eng.hf")

    val_set = AdapterTranslatorDataset(
        eng_dataset['dev'],
        por_dataset['dev'],
        tokenizer,
        args.max_len
    )

    val_params = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": 0,
    }

    val_loader = DataLoader(val_set, **val_params)

    predictions, actuals, ave_bleu = test(tokenizer, model, args.device, val_loader)
    final_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals})
    final_df.to_csv(args.results)
    print(f"Validation BLEU score: {ave_bleu}")
