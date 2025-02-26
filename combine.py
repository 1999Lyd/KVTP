import json
import ast
import os

def load_json(json_path):
    if os.path.exists(json_path):
        with open(json_path, 'r') as file:
            return json.load(file)
    return {}

# Load input JSONs
query_dict = load_json('description_score_long_debiased.json')
index_dict = load_json('description_score_long.json')

# Load existing output JSON once (if it exists)
output_json_path = 'skvqa.json'
data = load_json(output_json_path)  

# Process and update data
for key, value in index_dict.items():
    video_path = ast.literal_eval(key)[0]
    question = ast.literal_eval(key)[1]
    debiased_q = ast.literal_eval(query_dict[key])[2]
    caption = ast.literal_eval(index_dict[key])[0]
    index = ast.literal_eval(index_dict[key])[1]

    # Convert tuple to string key (JSON does not support tuples as keys)
    key_str = str((video_path, question))
    item = (caption, index, debiased_q)

    # Update dictionary in memory
    data[key_str] = item  # No need for str(item)

# Write JSON back to file once
with open(output_json_path, "w") as f:
    json.dump(data, f, indent=4)
