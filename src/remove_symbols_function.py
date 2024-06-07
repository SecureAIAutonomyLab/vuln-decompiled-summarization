import json

# Function to replace underscores with spaces in a list of strings
# used for evaluating function names Average similarity score
def replace_underscores(data):
    return [item.replace('_', ' ') for item in data]

# Read the JSON file
input_file = '../../Experiments_results/results_x86codellama_functionname.json'
output_file = '../../Experiments_results/results_x86codellama_functionname_modified.json'

with open(input_file, 'r') as file:
    data = json.load(file)

if 'pred' in data:
    data['pred'] = replace_underscores(data['pred'])
if 'gt' in data:
    data['gt'] = replace_underscores(data['gt'])

with open(output_file, 'w') as file:
    json.dump(data, file, indent=4)

print(f"Underscores replaced with spaces in pred and gt arrays. New file created: {output_file}")
