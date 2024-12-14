import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import typer
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import json
import os
import ast
import re

def load_viper_responses(file_path):
    question_id_to_viper_response = {}
    with open(file_path, 'r') as file:
        data = json.load(file)
        for entry in data:
            question_id = entry.get("question_id")
            viper_response = entry.get("viper_gpt", {}).get("viper_response")
            if question_id and viper_response:
                question_id_to_viper_response[question_id] = viper_response
    return question_id_to_viper_response

def format_string(s):
    # Convert to lowercase
    s = s.lower()
    
    # Remove punctuation
    s = re.sub(r'[^\w\s]', '', s)
    
    # Remove articles (a, an, the)
    s = re.sub(r'\b(a|an|the)\b', '', s)
    
    # Convert words to digits (if they are numbers)
    s = re.sub(r'\bzero\b', '0', s)
    s = re.sub(r'\bone\b', '1', s)
    s = re.sub(r'\btwo\b', '2', s)
    s = re.sub(r'\bthree\b', '3', s)
    s = re.sub(r'\bfour\b', '4', s)
    s = re.sub(r'\bfive\b', '5', s)
    s = re.sub(r'\bsix\b', '6', s)
    s = re.sub(r'\bseven\b', '7', s)
    s = re.sub(r'\beight\b', '8', s)
    s = re.sub(r'\bnine\b', '9', s)
    
    # Remove extra spaces
    s = re.sub(r'\s+', ' ', s).strip()
    
    return s

def _da_accuracy(preds: list[str], refs: list[list[str]]):
    """
    Compute Direct Answer Accuracy
    Preprocesses both predictions and references before comparison
    """
    # Preprocess predictions and references
    formatted_preds = [format_string(pred) for pred in preds]
    formatted_refs = [[format_string(ref) for ref in ref_list] for ref_list in refs]
    
    # Compute accuracy
    return sum([1 if any(ref.count(pred) >= 1 for ref in ref_list) else 0 
                for pred, ref_list in zip(formatted_preds, formatted_refs)]) / len(preds)

def compute_accuracy(task, preds, refs, correct_choices=None):
    """
    Compute accuracy based on task type
    """
    if task == "mc":
        # Multiple choice: exact match with correct choice
        return sum([1 if format_string(pred) == format_string(choice) else 0 
                    for pred, choice in zip(preds, correct_choices)]) / len(preds)
    elif task == "da":
        # Direct answer: check against direct answer candidates
        return _da_accuracy(preds, refs)
    else:
        raise ValueError(f"Unknown task type: {task}")

def process_and_save_results(updated_data, filename, task, accuracy, viper, rationale):
    # Save the updated dataset to disk
    with open(filename, 'w') as outfile:
        json.dump(updated_data, outfile, indent=4)

    print(f"Accuracy [task:{task}, viper:{viper}, rationale:{rationale}]: {accuracy:.4f}")

app = typer.Typer()

@app.command()
def evaluate(
    model_name: str = typer.Option(
        "Salesforce/instructblip-vicuna-7b", help="Name of the Hugging Face model"
    ),
    dataset_name: str = typer.Option(
        "HuggingFaceM4/A-OKVQA", help="Name of the dataset"
    ),
    split: str = typer.Option(
        "validation", help="Which split of the dataset to use"
    ),
    task: str = typer.Option(
        "da", help="task - direct answer (da) or multiple choice (mc)"
    ),
    viper: bool = typer.Option(
        False, help="Whether to include viper clues"
    ),
    rationale: bool = typer.Option(
        False, help="Whether to generate rationales"
    )
):
    if 'llava-v1.6' in model_name:
        filename = f"aokvqa_task_{task}_{split}_viper_{viper}_rationale_{rationale}.json"
    else:
        filename = f"aokvqa_task_{model_name.split('/')[-1]}_{task}_{split}_viper_{viper}_rationale_{rationale}.json"
        
    # Check if output file already exists
    if os.path.exists(filename):
        print(f"Existing file found: {filename}. Loading existing results...")
        
        # Load existing data
        with open(filename, 'r') as file:
            existing_data = json.load(file)
        
        # Reprocess existing data (parse responses, calculate accuracy)
        preds = [item.get('llava_zero_shot_answer', '') for item in existing_data]
        
        if task == "mc":
            correct_choices = [item['choices'][item['correct_choice_idx']] for item in existing_data]
            refs = None
        else:  # da task
            refs = [ast.literal_eval(item['direct_answers']) for item in existing_data]
            correct_choices = None
        
        accuracy = compute_accuracy(task, preds, refs, correct_choices)
        
        process_and_save_results(existing_data, filename, task, accuracy, viper, rationale)
        return

    # If file doesn't exist, run full inference
    device = "cuda"
    viper_file_path = '/home/jmire/Jan-ML/results/viper_augmentations/aokvqa_plus_viper_val.json'
    question_id_to_viper_response = load_viper_responses(viper_file_path)

    model = InstructBlipForConditionalGeneration.from_pretrained(model_name)
    processor = InstructBlipProcessor.from_pretrained(model_name)
    model.to(device)

    # Load the dataset
    print(f"Loading {dataset_name} dataset...")
    dataset = load_dataset(dataset_name, split=split)
    
    # Evaluation loop
    updated_data = []  # To store the updated dataset with full responses
    preds = []
    refs = []
    correct_choices = []
    
    for item in tqdm(dataset, desc="Evaluating"):
        # dataset properties
        question_id = item["question_id"]
        if question_id not in question_id_to_viper_response:
            continue
        viper_clues = question_id_to_viper_response[question_id]
        image = item["image"]
        question = item["question"]
        choices = item["choices"]
        correct_choice_idx = item["correct_choice_idx"]
        correct_choice = choices[correct_choice_idx]
        
        # Construct prompt
        if task == "da":
            question_part = f"Question: {question}\n"
            rationale_part = ("First, explain your thought process and then answer the question in 1-2 words in the following format:\n"
                            "Thought Process: [brief description of your thought process]\n")
            answer_part = "Answer (1-2 words): "
        else:
            choices_str = ", ".join(choices)
            question_part = (f"Question: {question}\n"
                             f"Answer Options: {choices_str}\n")
            rationale_part = ("First, explain your thought process and then answer the question by outputting exactly one of the given options. Don't include any additional output besides one of the multiple choice answers:\n"
                    "Thought Process: [brief description of your thought process]\n")
            answer_part = f"Answer (Choose one from this list - {choices_str}): "

        # construct prompt
        prompt = question_part
        if viper:
            prompt += f"Visual Clues: {viper_clues}\n"
        if rationale:
            prompt += rationale_part
        prompt += answer_part

        # Inference
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        inputs = processor(images=image, text=prompt, return_tensors="pt", padding=True, truncation=True).to(device)

        output = model.generate(
                **inputs,
                do_sample=False,
                num_beams=5,
                max_length=1024,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1,
        )

        response = processor.batch_decode(output, skip_special_tokens=True)[0].strip()

        # Answer extraction and processing
        if task == "da":
            answer = response.split("Answer (1-2 words):")[-1].strip().lower()
        else:
            answer = response.split("):")[-1].strip().lower()
        answer = answer.replace('[/inst]', '')
        answer = answer.replace('answer:', '')
        answer = answer.strip()

        # Prepare item for saving
        processed_item = {
            **item,
            'llava_zero_shot_full_response': response,
            'llava_zero_shot_answer': answer,
        }
        processed_item.pop("image", None)  # Remove non-serializable image
        updated_data.append(processed_item)
        
        # Collect data for accuracy computation
        preds.append(answer)
        correct_choices.append(correct_choice)
        
        # For DA task, collect direct answers
        if task == "da":
            # Convert string representation of list to actual list
            da_list = ast.literal_eval(item['direct_answers'])
            refs.append(da_list)

    # Compute accuracy based on task
    if task == "mc":
        accuracy = compute_accuracy(task, preds, None, correct_choices)
    else:  # da task
        accuracy = compute_accuracy(task, preds, refs)
    
    # Save results
    process_and_save_results(updated_data, filename, task, accuracy)

if __name__ == "__main__":
    app()