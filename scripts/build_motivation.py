import itertools
import pandas as pd
import random

from openai import OpenAI
from pathlib import Path
from pprint import pprint
from rich.progress import track
from typing import Final

# Load investigation data and setup ChatGPT client
df = pd.read_csv(Path(__file__).parent.parent /
                 'data' / 'investigation_dataset.csv')
client = OpenAI()

# Create 15-20 targeted quantities of each CWE
CWE_QUANTITIES: Final[dict[str, int]] = {cwe: random.randint(15, 20) for cwe in
                                         ['CWE-787', 'CWE-416', 'CWE-20', 'CWE-125', 'CWE-476',
                                          'CWE-190', 'CWE-119', 'CWE-798']}
print('Targeted CWE Distribution:')
pprint(CWE_QUANTITIES)


def create_prompt(cwe: str, decompiled_code: str) -> str:
    return f'Inject {cwe} in the following decompiled function, but ensure that the generated ' + \
        'vulnerable code looks like a decompiled code similar to the code provided:' + \
        f'\n\n```c\n{decompiled_code}\n```'


def extract_code(response: str) -> str:
    try:
        return response.split('```')[1].removeprefix('c').removeprefix('pp')
    except IndexError as e:
        raise ValueError('Could not extract code from ChatGPT response') from e


# Use GPT-4 to inject vulnerabilities in random samples
new_data = {'repo': [], 'benign_src_code': [], 'benign_decompiled': [], 'src_file': [],
            'gpt-4_vuln_code': [], 'gpt-4_response': [], 'cwe': []}
rows = []
for cwe in track(list(itertools.chain.from_iterable([t[0]] * t[1] for t in CWE_QUANTITIES.items())),
                 description='Generating vulnerable samples...'):
    if not rows:
        # Reset samples
        rows = [t[1] for t in df.iterrows()]
    sample = rows.pop(random.randint(0, len(rows) - 1))
    benign_decompiled_code = sample['benign_decompiled_code']
    try:
        # Extract code from GPT-4 response
        response = client.chat.completions.create(
            messages=[
                {
                    'role': 'user',
                    'content': create_prompt(cwe, benign_decompiled_code),
                }
            ],
            model='gpt-4-turbo',
        ).choices[0].message.content
        if not response:
            raise ValueError('GPT-4 did not return a response')
        vulnerable_code = extract_code(response)
    except Exception as e:
        print(e)
    else:
        # Add data to dataset
        new_data['repo'].append(sample['repo'])
        new_data['benign_src_code'].append(sample['benign_src_code'])
        new_data['benign_decompiled'].append(benign_decompiled_code)
        new_data['src_file'].append(sample['vuln_src_file'])
        new_data['gpt-4_vuln_code'].append(vulnerable_code)
        new_data['gpt-4_response'].append(response)
        new_data['cwe'].append(cwe)

# Save generated data
pd.DataFrame(new_data).to_csv(Path(__file__).parent.parent /
                              'data' / 'gpt_4_motivation_data.csv')
