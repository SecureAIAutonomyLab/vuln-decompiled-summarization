import pandas as pd

from openai import OpenAI
from pathlib import Path
from rich.progress import track

# Load investigation data and setup ChatGPT client
df = pd.read_csv(Path(__file__).parent.parent /
                 'data' / 'gpt_4_motivation_data.csv')
client = OpenAI()

df_dict = df.to_dict(orient='list')
df_dict['gpt-4_summary'] = [''] * len(df_dict['repo'])


def create_prompt(decompiled_code: str) -> str:
    return f'Summarize the semantics of the following decompiled function:' + \
        f'\n\n```c\n{decompiled_code}\n```'


for idx, decompiled_code in track(enumerate(df_dict['benign_decompiled']),
                                  total=len(df_dict['benign_decompiled']),
                                  description='Generating decompiled summaries...'):
    try:
        # Extract code from GPT-4 response
        response = client.chat.completions.create(
            messages=[
                {
                    'role': 'user',
                    'content': create_prompt(decompiled_code),
                }
            ],
            model='gpt-4-turbo',
        ).choices[0].message.content
        if not response:
            raise ValueError('GPT-4 did not return a response')
    except Exception as e:
        print(e)
    else:
        # Add data to dataset
        df_dict['gpt-4_summary'][idx] = response
        # Save generated data
        pd.DataFrame(df_dict).to_csv(Path(__file__).parent.parent /
                                     'data' / 'gpt_4_motivation_data_with_summaries.csv')
