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

from peft import get_peft_model, LoraConfig, TaskType

from datasets import load_dataset, load_from_disk

from tqdm import tqdm
import argparse

import os

from Dataloader import AdapterTranslatorDataset

def train(epoch, tokenizer, model, device, loader, optimizer, criterion):

    model.train()
    for _, data in enumerate(loader, 0):
        y = data["target_ids"].to(args.device, dtype=torch.long)
        labels = y.clone()
        labels[labels == tokenizer.pad_token_id] = -100 
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            labels=labels,
        )

        loss = criterion(outputs.logits, y)

        # if _ % 10 == 0:
        #     training_logger.add_row(str(epoch), str(_), str(loss))
        #     console.print(training_logger)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def validate(tokenizer, model, device, loader, criterion):

    model.eval()
    losses = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data["target_ids"].to(args.device, dtype=torch.long)
            labels = y.clone()
            labels[labels == tokenizer.pad_token_id] = -100 
            ids = data["source_ids"].to(device, dtype=torch.long)
            mask = data["source_mask"].to(device, dtype=torch.long)

            # print(ids, mask, lm_labels)

            outputs = model(
                input_ids=ids,
                attention_mask=mask,
                labels=labels,
            )

            loss = criterion(outputs.logits, y)
            losses.append(loss)

    return np.mean(losses)

def test(tokenizer, model, device, loader):

    """
    Function to evaluate model for predictions

    """
    model.eval()
    predictions = []
    actuals = []
    bleu_scores = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=150, 
                num_beams=2,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            if _%10==0:
                pass
                # console.print(f'Completed {_}')

            predictions.extend(preds)
            actuals.extend(target)
            bleu_scores.append(bleu_score(preds,target))

    ave_bleu = np.mean(bleu_scores)
    return predictions, actuals, ave_bleu 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', type = str, default = 'cuda')
    parser.add_argument('-batch_size', type = int, default= 8)
    parser.add_argument('-lr', type = float, default = 1e-3)
    parser.add_argument('-max_len', type = int, default = 128)
    parser.add_argument('-max_eps', type = int, default= 5)
    parser.add_argument('-dataset', type=str, default='../data/nus_dataset_triples.csv')
    parser.add_argument('-results', type=str, default='../data/triples_results_cameo.csv')
    parser.add_argument('-lang', type=str, help="pt, gl, ba", default='pt')
    parser.add_argument('-t5_path', type=str, default=None)
    args = parser.parse_args()
    parser.add_argument('-save_model_as', type=str, default=f'../models/adapter_translation_{args.lang}.pt')
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

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.to(args.device)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    por_dataset = load_from_disk("../data/por.hf")
    eng_dataset = load_from_disk("../data/eng.hf")

    training_set = AdapterTranslatorDataset(
        eng_dataset['devtest'],
        por_dataset['devtest'],
        tokenizer,
        args.max_len
    )
    val_set = AdapterTranslatorDataset(
        eng_dataset['dev'],
        por_dataset['dev'],
        tokenizer,
        args.max_len
    )

    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": 0,
    }

    val_params = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": 0,
    }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=args.lr
    )

    best_loss = 1e6

    for epoch in tqdm(range(args.max_eps)):
        train(epoch, tokenizer, model, args.device, training_loader, optimizer, criterion)
        val_loss = validate(epoch, tokenizer, model, args.device, val_loader, optimizer, criterion)
        print("Epoch {} complete. , Validation Loss : {}".format(epoch, val_loss))
        # wandb.log(
        #     {"train": {"loss": loss.item()},"val": {"loss": val_loss.item(), "wer": val_wer.item()}})
        if val_loss < best_loss:
            print("Best validation loss improved from {} to {}. Saving model...".format(best_loss, val_loss))
            best_loss = val_loss
            torch.save(model, args.save_model_as)

    predictions, actuals, ave_bleu = test(epoch, tokenizer, model, args.device, val_loader)
    final_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals})
    final_df.to_csv(f"../data/final_predictions_{args.lang}.csv")
    print(f"Validation BLEU score: {ave_bleu}")
