import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import typer
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import json
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests


file_path = '/home/jmire/Jan-ML/results/viper_augmentations/aokvqa_plus_viper_val.json'
question_id_to_viper_response = {}
with open(file_path, 'r') as file:
    data = json.load(file)
    for entry in data:
        question_id = entry.get("question_id")
        viper_response = entry.get("viper_gpt", {}).get("viper_response")
        if question_id and viper_response:
            question_id_to_viper_response[question_id] = viper_response

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
    device = "cuda"
    # processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    # model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
    # model.to(device)

    model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
    model.to(device)

    # Load the dataset
    print(f"Loading {dataset_name} dataset...")
    dataset = load_dataset(dataset_name, split=split)
    # dataset = dataset.select(range(20))
    
    # Evaluation loop
    correct = 0
    total = 0
    updated_data = []  # To store the updated dataset with full responses
    
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
        direct_answers = item["direct_answers"]
        rationales = item["rationales"]

        # prompt parts
        viper_part = f"Visual Clues: {viper_clues}\n" if viper_clues != "" else ""
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
            prompt += viper_part
        if rationale:
            prompt += rationale_part
        prompt += answer_part

        # conversation = [
        #     {
        #       "role": "user",
        #       "content": [
        #           {"type": "text", "text": prompt},
        #           {"type": "image"},
        #         ],
        #     },
        # ]

        # prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        # inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        inputs = processor(images=image, text=prompt, return_tensors="pt", padding=True, truncation=True).to(device)

        # output = model.generate(**inputs, max_new_tokens=200, pad_token_id=model.config.eos_token_id)

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

        # response = processor.decode(output[0], skip_special_tokens=True)

        response = processor.batch_decode(output, skip_special_tokens=True)[0].strip()

        if task == "da":
            answer = response.split("Answer (1-2 words):")[-1].strip().lower()
        else:
            answer = response.split("):")[-1].strip().lower()
        answer = answer.replace('[/inst]', '')
        answer = answer.replace('answer:', '')
        answer = answer.strip()
    
        if answer == correct_choice.lower():
            correct += 1
        total += 1

        # Store the full response in the dataset
        item["llava_zero_shot_full_response"] = response  # Add the full response as a new property
        item["llava_zero_shot_answer"] = answer  # Add the full response as a new property
        item.pop("image", None) # Remove the image property which cannot be serialized
        updated_data.append(item)  # Append the updated item to the new list
    
    # Save the updated dataset to disk
    filename = f"aokvqa_task_{model_name.split("/")[1]}_{task}_{split}_viper_{viper}_rationale_{rationale}.json"
    with open(filename, 'w') as outfile:
        json.dump(updated_data, outfile, indent=4)

    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy [model:{model_name}, task:{task}, viper:{viper}, rationale:{rationale}]: {accuracy:.4f}")
    
if __name__ == "__main__":
    app()
