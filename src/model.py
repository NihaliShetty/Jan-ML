import os
os.environ['COCO_DIR'] = '/usr1/data/mingqia2/datasets/coco/'
os.environ['AOKVQA_DIR'] = '/usr1/data/mingqia2/aokvqa/'
os.environ['HF_HOME'] = '/usr1/data/models_cache'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import json 
from collections import Counter
from tqdm import tqdm
import gc
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datasets import load_dataset
from PIL import Image
import wandb

from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

from load_aokvqa import load_aokvqa, get_coco_path
import random 
from accelerate import Accelerator
import torch.distributed as dist
from torch import nn
from torch.nn import DataParallel

lora_config = LoraConfig(
    r=8,  # Rank of the low-rank matrices
    lora_alpha=16,  # Scaling factor
    target_modules=["query", "value", "key"],  # Apply LoRA to specific layers
    lora_dropout=0.1  # Dropout rate for LoRA layers
)

def print_tensor_info(tensor, name):
    print(f"{name} - Shape: {tensor.shape}, "
          f"Dtype: {tensor.dtype}, "
          f"Device: {tensor.device}, "
          f"Memory: {tensor.element_size() * tensor.nelement() / 1024 / 1024 / 1024:.2f} GB")

def print_batch_details(batch):
    print("\nBatch Details:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: Shape={value.shape}, "
                  f"Dtype={value.dtype}, "
                  f"Device={value.device}, "
                  f"Memory={value.element_size() * value.nelement() / 1024 / 1024 / 1024:.2f} GB")
            
def print_memory_stats():
    print("\nCUDA Memory Stats:")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024:.2f} GB")
    print(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024 / 1024 / 1024:.2f} GB")
    print(f"Max Cached: {torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024:.2f} GB")

# Call this at various points in your training loop
class VQADataset(Dataset):
    def __init__(self, dataset, processor, coco_dir, max_length=256):
        """
        Args:
            dataset: List of samples with original question, answer, and visual clues.
            processor: BLIP processor for text and image preprocessing.
            image_dir: Path to the directory containing images.
        """
        self.dataset = dataset
        self.processor = processor
        self.coco_dir = coco_dir
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image_path = get_coco_path(sample['split'], sample['image_id'], self.coco_dir)
        image = Image.open(image_path).convert("RGB")
        
        question = sample['question']
        # question_id = sample['question_id']
        visual_clues = sample.get('viper_gpt', {}).get('viper_question', '') + ' ' + sample.get('viper_gpt', {}).get('viper_response', '')
        rationales = sample.get('rationales', [])
        
        # Format choices into A:xx, B:xx, ... 
        formatted_choices = ', '.join([f"{chr(65 + i)}: {choice}" for i, choice in enumerate(sample['choices'])])
        correct_choice = chr(65 + sample["correct_choice_idx"]) # Convert index to letter
        direct_answers = sample['direct_answers']
        most_frequent_answer = Counter(direct_answers).most_common()[0][0]

        # Prompts for MC and DA tasks
        mc_question = (
            f"Question: {question} \n Visual Clues: {visual_clues} \n Choices: {formatted_choices}. "
            "Please provide a rationale and then return the letter of the correct answer in the format: "
            "'Rationale: [your explanation] \\n Answer: [your answer]'."
        )
        da_question = (
            f"Question: {question} \n Visual Clues: {visual_clues}. "
            "Please provide a rationale and then return the direct answer in the format: "
            "'Rationale: [your explanation] \\n Answer: [your answer]'."
        )
        
        rationale_text = " ".join(rationales)
        mc_output = f"Rationale: {rationale_text} \n Answer: {correct_choice}"
        da_output = f"Rationale: {rationale_text} \n Answer: {most_frequent_answer}"
        
        # Process image and augmented text
        mc_encoding = self.processor(image, mc_question, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        da_encoding = self.processor(image, da_question, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")  
        
        mc_labels = self.processor.tokenizer(
        mc_output, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
    ).input_ids.squeeze(0)
        
        da_labels = self.processor.tokenizer(
        da_output, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
    ).input_ids.squeeze(0)
        
        # Ensure no unexpected dimensions in labels
        assert mc_labels.dim() == 1, f"Unexpected MC Labels Shape: {mc_labels.shape}"
        assert da_labels.dim() == 1, f"Unexpected DA Labels Shape: {da_labels.shape}"
           
        return {
            "pixel_values": mc_encoding["pixel_values"].squeeze(0),
            "mc_input_ids": mc_encoding["input_ids"].squeeze(0),
            "mc_attention_mask": mc_encoding["attention_mask"].squeeze(0),
            "mc_labels": mc_labels,
            "da_input_ids": da_encoding["input_ids"].squeeze(0),
            "da_attention_mask": da_encoding["attention_mask"].squeeze(0),
            "da_labels": da_labels,
            # "question_id": question_id
        }
            
def vqa_collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    mc_input_ids = torch.stack([item["mc_input_ids"] for item in batch])
    mc_attention_mask = torch.stack([item["mc_attention_mask"] for item in batch])
    mc_labels = torch.stack([item["mc_labels"] for item in batch])
    da_input_ids = torch.stack([item["da_input_ids"] for item in batch])
    da_attention_mask = torch.stack([item["da_attention_mask"] for item in batch])
    da_labels = torch.stack([item["da_labels"] for item in batch])
    # question_ids = torch.stack([item["question_id"] for item in batch])
    
    return {
        "pixel_values": pixel_values,
        "mc_input_ids": mc_input_ids,
        "mc_attention_mask": mc_attention_mask,
        "mc_labels": mc_labels,
        "da_input_ids": da_input_ids,
        "da_attention_mask": da_attention_mask,
        "da_labels": da_labels,
        # "question_ids": question_ids
    }
    
class CustomLoss(nn.Module):
    def __init__(self, da_weight=1.0, mc_weight=1.0):
        super(CustomLoss, self).__init__()
        self.da_weight = da_weight
        self.mc_weight = mc_weight
        self.loss_fn = nn.CrossEntropyLoss() 
    
    def forward(self, mc_logits, da_logits, mc_labels, da_labels):
        mc_logits = mc_logits.contiguous().view(-1, mc_logits.size(-1))  # Shape: [batch_size * seq_len, vocab_size]
        mc_labels = mc_labels.contiguous().view(-1)                   # Shape: [batch_size * seq_len]

        da_logits = da_logits.contiguous().view(-1, da_logits.size(-1))  # Shape: [batch_size * seq_len, vocab_size]
        da_labels = da_labels.contiguous().view(-1)                    # Shape: [batch_size * seq_len]
        
        mc_loss = self.loss_fn(mc_logits, mc_labels)
        da_loss = self.loss_fn(da_logits, da_labels)
        total_loss = self.mc_weight * mc_loss + self.da_weight * da_loss
        return total_loss, mc_loss, da_loss 

def train_model(model, train_dataloader, processor, val_dataloader, epochs, learning_rate, accelerator, da_weight=1.0, mc_weight=1.0):
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        loss_fn = CustomLoss(da_weight=da_weight, mc_weight=mc_weight)
        device = accelerator.device
        # prepare model, optimizer, and dataloaders with Accelerator
        model, train_dataloader, val_dataloader, optimizer = accelerator.prepare(
            model, train_dataloader, val_dataloader, optimizer
        )

        for epoch in range(epochs):
            model.train()
            total_loss, total_mc_loss, total_da_loss = 0, 0, 0
            
            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")):
                optimizer.zero_grad()
                # print(f"\nBatch {batch_idx}")
                # print_memory_stats()
                # print_batch_details(batch)
                pixel_values = batch["pixel_values"].to(device)

                # MC task
                mc_input_ids = batch["mc_input_ids"].to(device)

                # print_tensor_info(pixel_values, "Pixel Values")
                # print_tensor_info(mc_input_ids, "MC Input IDs")

                mc_attention_mask = batch["mc_attention_mask"].to(device)
                mc_labels = batch["mc_labels"].to(device)

                mc_outputs = model(
                    pixel_values=pixel_values, 
                    input_ids=mc_input_ids, 
                    attention_mask=mc_attention_mask, 
                    labels=mc_labels
                )

                # DA task
                da_input_ids = batch["da_input_ids"].to(device)
                da_attention_mask = batch["da_attention_mask"].to(device)
                da_labels = batch["da_labels"].to(device)
                
                da_outputs = model(
                    pixel_values=pixel_values, 
                    input_ids=da_input_ids, 
                    attention_mask=da_attention_mask, 
                    labels=da_labels
                )
                mc_logits = mc_outputs.logits[:, :mc_labels.size(1), :]  # Shape: [batch_size, label_seq_len, vocab_size]
                da_logits = da_outputs.logits[:, :da_labels.size(1), :]  # Shape: [batch_size, label_seq_len, vocab_size]
                # Compute weighted losses
                loss, mc_loss, da_loss = loss_fn(mc_logits, da_logits, mc_labels, da_labels)

                del mc_outputs, da_outputs, mc_logits, da_logits
                gc.collect()
                torch.cuda.empty_cache()

                # Backpropagation
                optimizer.zero_grad()
                # loss.backward()
                accelerator.backward(loss)
                # print_memory_stats()
                optimizer.step()
                # print_memory_stats()
                gc.collect()
                torch.cuda.empty_cache()

                # Log training metrics
                wandb.log({
                    "train_loss": loss.item(),
                    "train_mc_loss": mc_loss.item(),
                    "train_da_loss": da_loss.item(),
                    "epoch": epoch + 1,
                    "batch_idx": batch_idx,
                })

                total_loss += loss.item()
                total_mc_loss += mc_loss.item()
                total_da_loss += da_loss.item()
            
            wandb.log({
                "avg_epoch_loss": total_loss / len(train_dataloader),
                "avg_epoch_mc_loss": total_mc_loss / len(train_dataloader),
                "avg_epoch_da_loss": total_da_loss / len(train_dataloader),
            })

            print(f"Epoch {epoch + 1}/{epochs}: Total Loss = {total_loss / len(train_dataloader):.4f}, "
                f"MC Loss = {total_mc_loss / len(train_dataloader):.4f}, "
                f"DA Loss = {total_da_loss / len(train_dataloader):.4f}")
            
            # Evaluate after each epoch
            evaluate_model(model, val_dataloader, loss_fn, device, epoch)

def evaluate_model(model, dataloader, loss_fn, device, epoch):
    model.eval()
    total_loss, total_mc_loss, total_da_loss = 0, 0, 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch["pixel_values"].to(device)

            # MC task
            mc_input_ids = batch["mc_input_ids"].to(device)
            mc_attention_mask = batch["mc_attention_mask"].to(device)
            mc_labels = batch["mc_labels"].to(device)

            mc_outputs = model(
                pixel_values=pixel_values, 
                input_ids=mc_input_ids, 
                attention_mask=mc_attention_mask, 
                labels=mc_labels
            )

            # DA task
            da_input_ids = batch["da_input_ids"].to(device)
            da_attention_mask = batch["da_attention_mask"].to(device)
            da_labels = batch["da_labels"].to(device)

            da_outputs = model(
                pixel_values=pixel_values, 
                input_ids=da_input_ids, 
                attention_mask=da_attention_mask, 
                labels=da_labels
            )

            # Compute weighted losses
            mc_logits = mc_outputs.logits[:, :mc_labels.size(1), :]
            da_logits = da_outputs.logits[:, :da_labels.size(1), :]
            
            loss, mc_loss, da_loss = loss_fn(mc_logits, da_logits, mc_labels, da_labels)

            wandb.log({
                "eval_loss": loss.item(),
                "eval_mc_loss": mc_loss.item(),
                "eval_da_loss": da_loss.item(),
                "epoch": epoch
            })

            total_loss += loss.item()
            total_mc_loss += mc_loss.item()
            total_da_loss += da_loss.item()

    avg_loss = total_loss / len(dataloader)
    avg_mc_loss = total_mc_loss / len(dataloader)
    avg_da_loss = total_da_loss / len(dataloader)

    print(f"Validation: Total Loss = {avg_loss:.4f}, MC Loss = {avg_mc_loss:.4f}, DA Loss = {avg_da_loss:.4f}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="../results/viper_augmentations/aokvqa_plus_viper_train.json")
    parser.add_argument("--val_file", type=str, default="../results/viper_augmentations/aokvqa_plus_viper_val.json")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--da_weight", type=float, default=1.0)
    parser.add_argument("--mc_weight", type=float, default=1.0)
    parser.add_argument("--save_model_path", type=str, default="./trained_model")
    args = parser.parse_args()

    wandb.init(
        project="your_project_name",  # Replace with your project name
        config={
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "da_weight": args.da_weight,
            "mc_weight": args.mc_weight,
        },
    )

    coco_dir = os.getenv('COCO_DIR')
    aokvqa_dir = os.getenv('AOKVQA_DIR')
    
    # accelerator = Accelerator(gradient_accumulation_steps=4)
    accelerator = Accelerator(
        mixed_precision='fp16',  # or 'bf16'
        # gradient_accumulation_steps=1  # optional, if you want to accumulate gradients
    )
    
    print(f"Using device: {accelerator.device}")
    print(f"Number of GPUs available: {accelerator.num_processes}")
    print(accelerator.state.device)
    print(torch.cuda.device_count())

    device = accelerator.device
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
    model._set_gradient_checkpointing(True)
    print(model)
    model = get_peft_model(model, lora_config)
    model.to(device)

    print(f"Model is on device: {model.device}")
    
    training_dataset = load_dataset("json", data_files={"train": args.train_file}, split="train")
    validation_dataset = load_dataset("json", data_files={"val": args.val_file}, split="val")

    train_dataset = VQADataset(dataset=training_dataset, processor=processor, coco_dir=coco_dir)
    val_dataset = VQADataset(dataset=validation_dataset, processor=processor, coco_dir=coco_dir)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=vqa_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=vqa_collate_fn)

    train_model(model, train_dataloader, processor, val_dataloader, args.epochs, args.learning_rate, accelerator)

    model.save_pretrained(args.save_model_path)
    processor.save_pretrained(args.save_model_path)
    print(f"Model and processor saved to {args.save_model_path}")