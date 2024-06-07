from datasets import load_dataset,load_from_disk, concatenate_datasets
import transformers
from transformers import Trainer, DataCollatorForLanguageModeling
from dataclasses import dataclass, field
from typing import Optional
import warnings
import wandb, os
wandb.login()
from huggingface_hub import login
import re
login(token="WANDB_TOKEN")

warnings.filterwarnings("ignore", category=UserWarning, module="deepspeed")
# import deepspeed
# deepspeed.ops.op_builder.CPUAdamBuilder().load() # needed for now, will be fixed in the future

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="codellama/CodeLlama-7b-hf")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={
                           "help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

def train():
    wandb_project = "codellama_x86_4tasks"
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
        os.environ["WANDB_MODE"] = "online"
        os.environ["WANDB_DISABLED"] = "false"


    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    dataset = load_from_disk(data_args.data_path)
    eval_dataset = dataset['validation']
    eval_dataset = eval_dataset.remove_columns("label")
    train_dataset = dataset['optimized']
    train_dataset = train_dataset.remove_columns("label")
    dataset = train_dataset
    
    def tokenize_function(example_batch):
        return tokenizer(example_batch['prompt'], truncation=True, max_length=512, padding='max_length')

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        use_cache=False
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False
    )

    tokenizer.add_special_tokens({'pad_token': DEFAULT_PAD_TOKEN})
    tokenizer.add_special_tokens({'eos_token': DEFAULT_EOS_TOKEN})
    tokenizer.add_special_tokens({'bos_token': DEFAULT_BOS_TOKEN})
    tokenizer.add_special_tokens({'unk_token': DEFAULT_UNK_TOKEN})
    #print(tokenizer.eos_token)
    ##Added special end token only for llama 3
    # Apply the function to the dataset
    # print("Adding Llama3 EoS Tokens...\n")
    # def process_text(sample):
    #     text = sample["prompt"]
    #     # Remove <s> at the beginning and </s> at the end
    #     if text.startswith("<s>") and text.endswith("</s>"):
    #         #text = text[3:-4] + tokenizer.eos_token
    #         text = format_string(text) + tokenizer.eos_token
    #     sample["prompt"] = text
    #     #print('text: ' + text)
    #     return sample
    
    # def format_string(input_string):
    #     # Find the indices of relevant substrings
    #     inst_start = input_string.find("[INST]")
    #     code_start = input_string.find("Code:") + len("Code:")
    #     inst_end = input_string.find("[/INST]")
    #     code_end = input_string.find("[/INST]") + len("[/INST]")
        
    #     # Extract instruction, code, and response
    #     instruction = input_string[inst_start+len("[INST]"):code_start - len("Code:")].strip()
    #     code = input_string[code_start:inst_end].strip()
    #     response = input_string[code_end:-4].strip()

    #     # Format the string
    #     formatted_string = f"Below is an instruction that describes a task, paired with an input code that provides further context. Write a response that appropiately completes the request.\n\n"
    #     formatted_string += f"### Instruction\n{instruction}\n\n"
    #     formatted_string += f"### Code\n{code}\n\n"
    #     formatted_string += f"### Response\n{response}"
        
    #     return formatted_string

    # dataset = dataset.map(process_text)
    # eval_dataset = eval_dataset.map(process_text)

    model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    tokenized_eval_dataset = eval_dataset.map(tokenize_function,batched=True)

    trainer = Trainer(model=model, tokenizer=tokenizer,
                      args=training_args, train_dataset=tokenized_dataset,
                      eval_dataset=tokenized_eval_dataset,
                      data_collator=data_collator)

    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
