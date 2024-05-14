import logging
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import datasets
import torch
import yaml
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

logger = logging.getLogger(__name__)


def get_target_module(target_modules: str | list[str]) -> str | list[str]:
    if isinstance(target_modules, list):
        return target_modules
    elif target_modules == "all-linear":
        return target_modules
    elif target_modules == "deepseek":
        # https://github.com/deepseek-ai/DeepSeek-MoE/blob/66edeee5a4f75cbd76e0316229ad101805a90e01/finetune/finetune.py#L33
        return [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
        ]
    elif target_modules == "standard":
        # https://colab.research.google.com/drive/1g1cxccQz3ki01XMH9ut-cISQ3enmbvSA?usp=sharing
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ]


def load_dataset(data_files: list[str]) -> datasets.Dataset:
    ds_list: list[datasets.Dataset] = []
    for data_file in data_files:
        _ds = datasets.load_dataset("json", data_files=data_file)
        _ds = _ds["train"]
        _ds = _ds.select_columns("text")
        ds_list.append(_ds)
    return datasets.concatenate_datasets(ds_list)


@dataclass
class SFTTrainingArguments:
    model_name_or_path: str
    tokenizer_name_or_path: str | None
    additional_special_tokens: list[str] | None
    use_fast: bool
    load_in_8bit: bool
    load_in_4bit: bool
    use_flash_attention_2: bool
    max_seq_length: int
    data_files: list[str]
    eval_data_files: list[str] | None

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("load_in_8bit and load_in_4bit are mutually exclusive")

    def from_pretrained_kwargs(self, training_args: TrainingArguments) -> dict:
        if self.load_in_8bit:
            kwargs = {"load_in_8bit": True}
        elif self.load_in_4bit:
            kwargs = {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            }
        elif training_args.bf16:
            kwargs = {"torch_dtype": torch.bfloat16}
        else:
            kwargs = {"torch_dtype": torch.float16}
        kwargs["use_flash_attention_2"] = self.use_flash_attention_2
        return kwargs


def load_config(
    config_path: str,
) -> tuple[TrainingArguments, SFTTrainingArguments, LoraConfig | None]:
    config = yaml.safe_load(Path(config_path).open())

    training_args = TrainingArguments(**config["training_args"])
    sft_args = SFTTrainingArguments(**config["sft_args"])

    lora_config: LoraConfig | None = None
    if config["lora"] is not None:
        target_modules = get_target_module(config["lora"].pop("target_modules"))
        lora_config = LoraConfig(
            fan_in_fan_out=True,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
            **config["lora"],
        )

    return training_args, sft_args, lora_config


def main(config_path: str) -> None:
    logger.info("Loading config")
    training_args, sft_args, lora_config = load_config(config_path)

    # setup tokenizer
    tokenizer_name_or_path = (
        sft_args.tokenizer_name_or_path or sft_args.model_name_or_path
    )
    logger.info(f"Loading tokenizer from {tokenizer_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_fast=sft_args.use_fast,
        additional_special_tokens=sft_args.additional_special_tokens,
        trust_remote_code=True,
    )

    logger.info("Loading data")
    train_dataset = load_dataset(sft_args.data_files)
    if sft_args.eval_data_files:
        eval_dataset = load_dataset(sft_args.eval_data_files)
    else:
        eval_dataset = None

    # setup prompts
    logger.info("Formatting prompts")
    instruction_ids = tokenizer.encode("\n\n### 指示:\n", add_special_tokens=False)[1:]
    response_ids = tokenizer.encode("\n\n### 応答:\n", add_special_tokens=False)[1:]
    collator = DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_ids,
        response_template=response_ids,
        tokenizer=tokenizer,
    )

    logger.info(f"Loading model from {sft_args.model_name_or_path}")
    kwargs = sft_args.from_pretrained_kwargs(training_args)
    model = AutoModelForCausalLM.from_pretrained(
        sft_args.model_name_or_path,
        trust_remote_code=True,
        **kwargs,
    )

    trainer = SFTTrainer(
        args=training_args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        data_collator=collator,
        peft_config=lora_config,
        max_seq_length=sft_args.max_seq_length,
    )

    logger.info("Training")
    trainer.train()

    logger.info("Saving model")
    trainer.save_model()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )

    parser = ArgumentParser()
    parser.add_argument("--config-path", "-c", type=str)
    args = parser.parse_args()
    main(config_path=args.config_path)
