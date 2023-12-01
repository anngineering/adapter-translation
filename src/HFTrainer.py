import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchtext.data.metrics import bleu_score

from datasets import load_dataset, load_from_disk, concatenate_datasets

from tqdm import tqdm

import os

from dataloader import AdapterTranslatorDataset

from utils import utils

from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
    Trainer
)
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
# from trl import SFTTrainer

from sklearn.model_selection import train_test_split


logging.set_verbosity(logging.CRITICAL)

class Llama2ForTranslation:
    def __init__(self):
        self.cfg = utils.load_config()
        # self.device_map = {"": 0}
        compute_dtype = getattr(torch, self.cfg['bnb_8bit_compute_dtype'])

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.cfg["use_8bit"],
            bnb_4bit_quant_type=self.cfg["bnb_8bit_quant_type"],
            bnb_4bit_compute_dtype=compute_dtype
        )

        # if compute_dtype == torch.float16 and self.cfg['use_8bit']:
        #     major, _ = torch.cuda.get_device_capability()
        #     if major >= 8:
        #         print("=" * 80)
        #         print("Your GPU supports bfloat16: accelerate training with bf16=True")
        #         print("=" * 80)

        self.other_dataset = load_from_disk(self.cfg["dataset_path"])
        self.eng_dataset = load_from_disk("../data/eng.hf")

        self.eng_dataset = concatenate_datasets([self.eng_dataset['devtest'], self.eng_dataset['dev']])
        self.other_dataset = concatenate_datasets([self.other_dataset['devtest'], self.other_dataset['dev']])

        self.eng_train_dataset, self.eng_test_dataset = train_test_split(self.eng_dataset, test_size=0.2, random_state=42)
        self.other_train_dataset, self.other_test_dataset = train_test_split(self.other_dataset, test_size=0.2, random_state=42)

        self.tokenizer = AutoTokenizer.from_pretrained(os.environ["TOKENIZER_7B_PATH"],
                                          use_auth_token=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"


        self.model = LlamaForCausalLM.from_pretrained(
            os.environ['LLAMA_MODEL_7B_PATH'],
            quantization_config=bnb_config,
            device_map='auto'
        )
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1

        self.peft_config = LoraConfig(
            lora_alpha=self.cfg['lora_alpha'],
            lora_dropout=self.cfg['lora_dropout'],
            r=self.cfg['lora_r'],
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model.add_adapter(self.peft_config)
        # self.model.print_trainable_parameters()

        self.training_arguments = TrainingArguments(
            output_dir=self.cfg["output_dir"],
            num_train_epochs=self.cfg["max_eps"],
            per_device_train_batch_size=self.cfg['per_device_train_batch_size'],
            per_device_eval_batch_size=self.cfg['per_device_eval_batch_size'],
            gradient_accumulation_steps=self.cfg['gradient_accumulation_steps'],
            # max_seq_length=self.cfg['max_seq_length'],
            optim=self.cfg["optim"],
            save_steps=self.cfg['save_steps'],
            logging_steps=self.cfg['logging_steps'],
            learning_rate=self.cfg['learning_rate'],
            weight_decay=self.cfg['weight_decay'],
            fp16=self.cfg['fp16'],
            bf16=self.cfg['bf16'],
            max_grad_norm=self.cfg['max_grad_norm'],
            max_steps=self.cfg['max_steps'],
            warmup_ratio=self.cfg['warmup_ratio'],
            # group_by_length=self.cfg['group_by_length'],
            lr_scheduler_type=self.cfg['lr_scheduler_type'],
            eval_accumulation_steps=4,
            report_to="tensorboard"
        )

        self.saved_model = '_'.join([os.environ['LLAMA_MODEL_7B_PATH'], self.cfg["lang"]])
        # self.saved_model = f'/models/test_translate/{self.cfg["lang"]}'

    def get_dataloader(self):
        training_set = AdapterTranslatorDataset(
        self.eng_train_dataset,
        self.other_train_dataset,
        self.tokenizer,
        self.cfg['max_seq_length'],
        self.cfg["lang"]
        )
        val_set = AdapterTranslatorDataset(
            self.eng_test_dataset,
            self.other_test_dataset,
            self.tokenizer,
            self.cfg['max_seq_length'],
            self.cfg["lang"]
        )

        # # Defining the parameters for creation of dataloaders
        # train_params = {
        #     "batch_size": self.cfg['per_device_train_batch_size'],
        #     "shuffle": True,
        #     "num_workers": 5,
        # }

        # val_params = {
        #     "batch_size": self.cfg['per_device_eval_batch_size'],
        #     "shuffle": False,
        #     "num_workers": 5,
        # }

        # # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
        # training_loader = DataLoader(training_set, **train_params)
        # val_loader = DataLoader(val_set, **val_params)

        return training_set, val_set
    
    def train_and_eval(self):
        training_set, val_set = self.get_dataloader()

        trainer = Trainer(
            model=self.model,
            train_dataset=training_set,
            eval_dataset=val_set,
            # peft_config=self.peft_config,
            # dataset_text_field="text",
            tokenizer=self.tokenizer,
            args=self.training_arguments,
            # packing=self.cfg["packing"],
            # compute_metrics=lambda p: {"bleu_score": self.compute_metrics(p)},
            compute_metrics= self.compute_metrics,
        )
        trainer.train()

        trainer.model.save_pretrained(self.saved_model)

        results = trainer.evaluate()


    def compute_metrics(self, pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions.argmax(-1)

        preds = [self.tokenizer.batch_decode(p, skip_special_tokens=True, clean_up_tokenization_spaces=True) for p in pred_ids]
        labels_ids[labels_ids == -100] = self.tokenizer.pad_token_id
        labels = [self.tokenizer.batch_decode(l, skip_special_tokens=True, clean_up_tokenization_spaces=True) for l in labels_ids]
        bleu = bleu_score(preds, labels)

        print(preds[0])
        print(labels[0])

        print(f"BLEU Score: {bleu}")

        return {'bleu': bleu}


    def reload_saved_model(self):
        base_model = self.model = LlamaForCausalLM.from_pretrained(
        os.environ['LLAMA_MODEL_7B_PATH'],
        # low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        load_in_4bit=True
        # device_map=self.device_map,
        )
        model = PeftModel.from_pretrained(base_model, self.saved_model)
        return model.merge_and_unload()

    def test(self, input_text):
        model = self.reload_saved_model()
        tokenizer = AutoTokenizer.from_pretrained(os.environ['LLAMA_MODEL_7B_PATH'], trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        prompt = input_text
        pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=128)
        result = pipe(f"<s>[INST] {prompt} [/INST]")
        print(result[0]['generated_text'])


if __name__ == "__main__":
    llama_translator = Llama2ForTranslation()

