from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
import torchtext
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os

from peft import get_peft_model, LoraConfig, TaskType

from datasets import load_dataset 

def train(epoch, tokenizer, model, device, loader, optimizer):

    """
    Function to be called for training with the parameters passed from main function

    """

    model.train()
    for _, data in enumerate(loader, 0):
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)

        print(ids, mask, lm_labels)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            labels=lm_labels,
        )
        loss = outputs[0]

        # if _ % 10 == 0:
        #     training_logger.add_row(str(epoch), str(_), str(loss))
        #     console.print(training_logger)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def validate(epoch, tokenizer, model, device, loader):

    """
    Function to evaluate model for predictions

    """
    model.eval()
    predictions = []
    actuals = []
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
    return predictions, actuals

def T5Trainer(
    model, tokenizer, model_params, output_dir="./outputs/"
):

    """
    T5 trainer

    """

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # logging
    # console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    # tokenzier for encoding the text
    # tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    # model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    model = model.to('cuda')

    # logging
    # console.log(f"[Data]: Reading data...\n")

    # Importing the raw dataset
    # dataframe = dataframe[[source_text, target_text]]
    # display_df(dataframe.head(2))

    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training and the rest for validation.
    # train_size = 0.8
    # train_dataset = dataframe.sample(frac=train_size, random_state=model_params["SEED"])
    # val_dataset = dataframe.drop(train_dataset.index).reset_index(drop=True)
    # train_dataset = train_dataset.reset_index(drop=True)

    # console.print(f"FULL Dataset: {dataframe.shape}")
    # console.print(f"TRAIN Dataset: {train_dataset.shape}")
    # console.print(f"TEST Dataset: {val_dataset.shape}\n")

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = YourDataSetClass(
        eng_dataset['devtest'],
        por_dataset['devtest'],
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
    )
    val_set = YourDataSetClass(
        eng_dataset['dev'],
        por_dataset['dev'],
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
    )

    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": model_params["TRAIN_BATCH_SIZE"],
        "shuffle": True,
        "num_workers": 0,
    }

    val_params = {
        "batch_size": model_params["VALID_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=model_params["LEARNING_RATE"]
    )

    # Training loop
    # console.log(f"[Initiating Fine Tuning]...\n")

    for epoch in range(model_params["TRAIN_EPOCHS"]):
        train(epoch, tokenizer, model, 'cuda', training_loader, optimizer)

    # console.log(f"[Saving Model]...\n")
    # Saving the model after training
    path = os.path.join(output_dir, "model_files")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

    # evaluating test dataset
    # console.log(f"[Initiating Validation]...\n")
    for epoch in range(model_params["VAL_EPOCHS"]):
        predictions, actuals = validate(epoch, tokenizer, model, 'cuda', val_loader)
        final_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals})
        final_df.to_csv(os.path.join(output_dir, "predictions.csv"))

    # console.save_text(os.path.join(output_dir, "logs.txt"))

    # console.log(f"[Validation Completed.]\n")
    # console.print(
    #     f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n"""
    # )
    # console.print(
    #     f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir,'predictions.csv')}\n"""
    # )
    # console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir,'logs.txt')}\n""")    

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                          use_auth_token=True,)
    tokenizer.pad_token = tokenizer.eos_token

    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                                device_map='auto',
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


    por_dataset = load_dataset('facebook/flores', 'por_Latn')
    eng_dataset = load_dataset('facebook/flores', 'eng_Latn')

    por_dataset.save_to_disk("data/por.hf")
    eng_dataset.save_to_disk("data/eng.hf")

    model_params = {
        "MODEL": "t5-base",  # model_type: t5-base/t5-large
        "TRAIN_BATCH_SIZE": 8,  # training batch size
        "VALID_BATCH_SIZE": 8,  # validation batch size
        "TRAIN_EPOCHS": 3,  # number of training epochs
        "VAL_EPOCHS": 1,  # number of validation epochs
        "LEARNING_RATE": 1e-4,  # learning rate
        "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text
        "MAX_TARGET_TEXT_LENGTH": 50,  # max length of target text
        "SEED": 42,  # set seed for reproducibility
    }

    T5Trainer(
        model=model,
        tokenizer=tokenizer,
        model_params=model_params,
        output_dir="outputs",
    )