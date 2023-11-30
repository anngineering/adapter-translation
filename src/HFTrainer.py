import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchtext.data.metrics import bleu_score

from datasets import load_dataset, load_from_disk

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
)
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from trl import SFTTrainer


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
        self.model = get_peft_model(self.model, self.peft_config)

        # self.model.print_trainable_parameters()

        self.training_arguments = TrainingArguments(
            output_dir=self.cfg["output_dir"],
            num_train_epochs=self.cfg["max_eps"],
            per_device_train_batch_size=self.cfg['per_device_train_batch_size'],
            per_device_eval_batch_size=self.cfg['per_device_eval_batch_size'],
            gradient_accumulation_steps=self.cfg['gradient_accumulation_steps'],
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
            report_to="tensorboard"
        )

        self.saved_model = '_'.join([os.environ['LLAMA_MODEL_7B_PATH'], self.cfg["lang"]])
        # self.saved_model = f'/models/test_translate/{self.cfg["lang"]}'

    def get_dataloader(self):
        training_set = AdapterTranslatorDataset(
        self.eng_dataset['devtest'],
        self.other_dataset['devtest'],
        self.tokenizer,
        self.cfg['max_seq_length'],
        self.cfg["lang"]
        )
        val_set = AdapterTranslatorDataset(
            self.eng_dataset['dev'],
            self.other_dataset['dev'],
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

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=training_set,
            eval_dataset=val_set,
            peft_config=self.peft_config,
            dataset_text_field="text",
            max_seq_length=self.cfg['max_seq_length'],
            tokenizer=self.tokenizer,
            args=self.training_arguments,
            packing=self.cfg["packing"],
            compute_metrics=lambda p: {"bleu_score": bleu_score(p.predictions, p.label_ids)},
        )
        trainer.train()

        trainer.model.save_pretrained(self.saved_model)

        results = trainer.evaluate()

        print(f"BLEU Score: {results['bleu_score']}")

    def reload_saved_model(self):
        base_model = self.model = LlamaForCausalLM.from_pretrained(
        os.environ['LLAMA_MODEL_7B_PATH'],
        # low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=self.device_map,
        )
        model = PeftModel.from_pretrained(base_model, self.saved_model)
        return model.merge_and_unload()

    def test(self, input_text):
        model = self.reload_saved_model()
        tokenizer = AutoTokenizer.from_pretrained(os.environ['LLAMA_MODEL_7B_PATH'], trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        prompt = input_text
        pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
        result = pipe(f"<s>[INST] {prompt} [/INST]")
        print(result[0]['generated_text'])


if __name__ == "__main__":
    llama_translator = Llama2ForTranslation()

