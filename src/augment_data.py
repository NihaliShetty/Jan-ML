# to augment aokvqa data with viper-gpt question-answer pairs
import json

# Change Filename!
with open('../datasets/aokvqa/aokvqa_v1p0_val.json', 'r') as file:
    aokvqa_train = json.load(file)
# Change Filename!
with open('../results/viper_augmentations/validation_questions_filtered.json', 'r') as file:
    viper_train = json.load(file)

final_train = []
count = 0
other_count = 0
for k, v in viper_train.items():
    # filter out the empty responses
    if len(v["response"]) == 0:
        count += 1
        continue
    question_id = k
    new_dict = {"viper_question": v["question"], "viper_response": v["response"]}
    for item in aokvqa_train:
        if item["question_id"] == question_id:
            item["viper_gpt"] = new_dict
            final_train.append(item)
            # print(item)

print("Length of original qokvpa data: ", len(aokvqa_train))
print("Empty responses from viper gpt: ", count)
print("Final data length: ", len(final_train))
print("Filtered out all the empty ones: ", len(final_train) == len(aokvqa_train) - count)

    
# print(final_train[-2])

# Change Filename!
file_path = '../results/viper_augmentations/aokvqa_plus_viper_val.json'

with open(file_path, 'w') as file:
    json.dump(final_train, file, indent=4)