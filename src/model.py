import os
# os.environ['COCO_DIR'] = 'coco2017/'
# os.environ['AOKVQA_DIR'] = '/usr1/data/mingqia2/aokvqa/'
# os.environ['HF_HOME'] = '/usr1/data/models_cache'

os.environ['COCO_DIR'] = '/home/lifanyu/Documents/GitHub/aokvqa/datasets/coco/'
os.environ['AOKVQA_DIR'] = '/home/lifanyu/Documents/GitHub/aokvqa/datasets/aokvqa/'
os.environ['HF_HOME'] = '/home/lifanyu/Documents/models_cache'


import json 
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# from datasets import load_dataset
from PIL import Image

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering

from load_aokvqa import load_aokvqa, get_coco_path


#################### Params ####################

NUM_EPOCHS = 3
LR = 5e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# device = "cpu"
# breakpoint()

# Load BLIP-2 model and processor
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="auto")


#################### Loss and Dataset ####################

def generate_visual_clues(vqa_input):
    pass

def preprocess_vqa(sample, processor):
    inputs = processor(
        text=sample["question"], images=sample["image_path"], return_tensors="pt", padding=True
    ).to(device)
    
    # Process answer
    labels = processor.tokenizer(sample["answer"], return_tensors="pt", padding=True).input_ids.to(device)
    
    # Process visual clues (textual metadata)
    visual_clues = generate_visual_clues(sample)
    metadata_inputs = processor.tokenizer(
        visual_clues, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    
    return inputs, labels, metadata_inputs

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mc_loss = nn.CrossEntropyLoss()  # Loss for MC task
        self.da_loss = nn.CrossEntropyLoss()  # Loss for DA task
        # self.ce_loss = nn.CrossEntropyLoss() # overall loss  
        self.aux_loss = nn.MSELoss()  # Auxiliary loss for alignment
    
    def forward(self, mc_logits, da_logits, mc_labels, da_labels):
        # Primary loss for VQA prediction
        mc_loss = self.mc_loss(mc_logits, mc_labels)
        da_loss = self.da_loss(da_logits.view(-1, da_logits.size(-1)), da_labels.view(-1))
        total_loss = mc_loss + da_loss
        return total_loss 
    
        # Auxiliary loss to align metadata and question embeddings
        # aux = self.aux_loss(metadata_embeddings, question_embeddings)
        # return ce + 0.1 * aux  # Combine with weighting

class VQADataset(Dataset):
    def __init__(self, dataset, processor, coco_dir):
        """
        Args:
            dataset: List of samples with original question, answer, and visual clues.
            processor: BLIP processor for text and image preprocessing.
            image_dir: Path to the directory containing images.
        """
        self.dataset = dataset
        self.processor = processor
        self.coco_dir = coco_dir
        # self.split = split # train or val 
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
         # Get image path and open image
        image_path = get_coco_path(sample['split'], sample['image_id'], self.coco_dir)
        image = Image.open(image_path).convert("RGB")
        
        # Combine question with MC options
        question = sample['question']
        choices = sample['choices']
        visual_clues = sample.get('visual_clues', '')  # List of question-answer pairs
        
        mc_question = f"Question: {question} Visual Clues: {visual_clues} Choices: {', '.join(choices)}"
        da_question = f"Question: {question} Visual Clues: {visual_clues}"
        
        # Process image and augmented text
        # mc_encoding = self.processor(image, mc_question, padding="max_length", truncation=True, return_tensors="pt")
        # da_encoding = self.processor(image, da_question, padding="max_length", truncation=True, return_tensors="pt")
        max_length = 128 
        mc_encoding = self.processor(image, mc_question, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        da_encoding = self.processor(image, da_question, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        print(mc_encoding['pixel_values'].shape)
        

        # Prepare answers 
        direct_answers = sample['direct_answers']
        correct_choice_idx = sample['correct_choice_idx'] # index of the correct MC choice
        
        da_text = " | ".join(direct_answers)
        da_labels = self.processor.tokenizer.encode(
            da_text, max_length=128, padding="max_length", truncation=True, return_tensors="pt"
        )
        mc_label = torch.tensor(correct_choice_idx, dtype=torch.long)
        
        encoding = {
            "pixel_values": mc_encoding["pixel_values"],  # Same image for both versions
            "mc_input_ids": mc_encoding["input_ids"],
            "mc_attention_mask": mc_encoding["attention_mask"],
            "da_input_ids": da_encoding["input_ids"],
            "da_attention_mask": da_encoding["attention_mask"],
            "direct_answer_labels": da_labels.squeeze(),
            "multiple_choice_label": mc_label
        }
        
        return encoding
    
    def getItem(self, idx):
        return self.__getitem__(idx)
    
#################### Main ####################

coco_dir = os.getenv('COCO_DIR')
aokvqa_dir = os.getenv('AOKVQA_DIR')
original_val_dataset = load_aokvqa(aokvqa_dir, 'val')
original_train_dataset = load_aokvqa(aokvqa_dir, 'train')

# training_dataset = load_dataset("json", 
#                                 data_files="/home/lifanyu/Documents/GitHub/Jan-ML/results/viper_augmentations/train_questions_filtered.json", 
#                                 # split="train"
#                                 )
# valid_dataset = load_dataset("json", 
#                             #  data_files="/home/lifanyu/Documents/GitHub/Jan-ML/results/viper_augmentations/validation_questions_filtered.json", 
#                             #  split="test" # this is not valid, this function aloows only "train" argument here
#                             )
# print("Training sets: {} - Validating set: {}".format(len(training_dataset), len(valid_dataset)))

augmented_train_dataset = json.load(open(
        # "/home/lifanyu/Documents/GitHub/Jan-ML/results/viper_augmentations/train_questions_filtered.json"
        "/home/lifanyu/Documents/GitHub/Jan-ML/datasets/aokvqa/aokvqa_v1p0_val.json"
    ))
augmented_valid_dataset = json.load(open(
        # "/home/lifanyu/Documents/GitHub/Jan-ML/results/viper_augmentations/validation_questions_filtered.json"
        "/home/lifanyu/Documents/GitHub/Jan-ML/datasets/aokvqa/aokvqa_v1p0_train.json"
    ))

train_dataset = VQADataset(dataset=augmented_train_dataset,
                          processor=processor,
                          coco_dir=coco_dir)
valid_dataset = VQADataset(dataset=augmented_valid_dataset,
                          processor=processor,
                          coco_dir=coco_dir)
# print(train_dataset[0], "\n \nExample Above")
print("Training sets: {} - Validating set: {}".format(len(train_dataset), len(valid_dataset)))

BATCH_SIZE = 8
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

# for k, v in train_dataset[0].items():
#     print(k)
#     try:
#         print(v.shape)
#     except:
#         print(len(v))
#     print("\n--------")

for i in range(100):
    item = train_dataset[i]
    print(item["da_attention_mask"].shape)

#################### Training Loop ####################

loss_fn = CustomLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
        # Move data to device
        print("here")
        pixel_values = batch["pixel_values"].to(device)
        pixel_values = pixel_values.squeeze(1)
        print("pixel_values shape (batch):", pixel_values.shape)

        mc_input_ids = batch["mc_input_ids"].to(device)
        mc_attention_mask = batch["mc_attention_mask"].to(device)
        da_input_ids = batch["da_input_ids"].to(device)
        da_attention_mask = batch["da_attention_mask"].to(device)
        mc_labels = batch["multiple_choice_label"].to(device)
        da_labels = batch["direct_answer_labels"].to(device)

        # Forward pass for MC
        mc_outputs = model(pixel_values=pixel_values, input_ids=mc_input_ids, attention_mask=mc_attention_mask)
        mc_logits = mc_outputs.logits
        

        # Forward pass for DA
        da_outputs = model(pixel_values=pixel_values, input_ids=da_input_ids, attention_mask=da_attention_mask)
        da_logits = da_outputs.logits
        
        loss = loss_fn(mc_logits=mc_logits, da_logits=da_logits, mc_labels=mc_labels, da_labels=da_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch}: Loss = {total_loss / len(train_dataset)}")