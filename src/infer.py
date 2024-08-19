from transformers import AutoTokenizer,AutoModelForCausalLM,AutoModelForSeq2SeqLM
#from peft import AutoPeftModelForCausalLM,PeftModel
from datasets import load_dataset,load_from_disk
import torchinfo
import evaluate
import os
import torch
import pandas as pd
import json
import math
import random
import re
import sys

identification = [
    "As a specialist in code security, assess this decompiled code. Is there any vulnerability present? Respond with 'YES' or 'NO' only.",
    "Examine the provided decompiled code for security flaws. If you detect any vulnerabilities, reply with 'YES'. If no vulnerabilities are found, reply with 'NO'.",
    "Check this code for potential vulnerabilities. If vulnerabilities exist, simply state 'YES'. If not, state 'NO'. Avoid additional content.",
    "You are tasked with identifying vulnerabilities in decompiled code. Do vulnerabilities exist here? Answer 'YES' or 'NO' without elaboration.",
    "Analyze the decompiled code below. Should you find any security vulnerabilities, answer only with 'YES'. If none, answer 'NO'.",
    "Please review the following code. Are there any vulnerabilities? Respond 'YES' for vulnerabilities present, 'NO' if absent.",
    "As a code analysis expert, inspect the code for any vulnerabilities. Reply exclusively with 'YES' if any are found, otherwise, 'NO'.",
    "Evaluate the provided code for vulnerabilities. If you identify any, respond with a single 'YES'. If no vulnerabilities, respond with 'NO'.",
    "Determine if the following decompiled code contains vulnerabilities. A simple 'YES' or 'NO' will suffice as your response.",
    "Assess this code snippet for security risks. If you find vulnerabilities, answer 'YES'. If it is secure, answer 'NO'.",
    "Your task is to check for vulnerabilities in this code. If any are present, respond 'YES'. If there are none, respond 'NO'.",
    "Look through the given decompiled code. Are there vulnerabilities? Please answer only with 'YES' or 'NO'.",
    "As a reverse engineer, scrutinize this code for vulnerabilities. Should any be present, simply say 'YES'. If not, say 'NO'.",
    "You are evaluating code security. Do you find any vulnerabilities in the code below? Just answer 'YES' or 'NO'.",
    "Analyze the following decompiled code for security issues. Respond with 'YES' if vulnerabilities exist, otherwise 'NO'.",
    "Survey this decompiled code snippet. Identify any vulnerabilities by responding with only 'YES' or 'NO'.",
    "As a code security analyst, inspect the provided code. Is it vulnerable? Reply solely with 'YES' or 'NO'.",
    "Your expertise is needed to identify vulnerabilities in this code. Do you find any? Answer 'YES' or 'NO' without additional comments.",
    "Review this decompiled code for potential security flaws. Should any be detected, respond only with 'YES'. If none, respond with 'NO'.",
    "Examine the provided decompiled code. Is there any vulnerability? Please answer with either 'YES' or 'NO' only."
]

classification = [
    "Assess the code snippet below and classify any vulnerability using the format 'CWE-XXX'. Provide no additional content.",
    "Review this decompiled code and identify any vulnerability by its CWE number, replying only with 'CWE-XXX'.",
    "Analyze the provided code for vulnerabilities and categorize them strictly in 'CWE-XXX' format. Avoid extra outputs.",
    "Examine the code and provide the CWE classification of any vulnerabilities found, using only 'CWE-XXX'.",
    "Evaluate the decompiled code and indicate any vulnerabilities by responding with 'CWE-XXX'. Exclude additional text.",
    "Your task is to identify and classify vulnerabilities in the following code. Use the format 'CWE-XXX' only.",
    "Scan this decompiled code for vulnerabilities and specify them in the format 'CWE-XXX'. No further details required.",
    "Inspect the code provided below and determine the vulnerability using the designation 'CWE-XXX'. Provide no other outputs.",
    "Your expertise in code security is required to classify vulnerabilities in this code as 'CWE-XXX'. No extra content allowed.",
    "Analyze this snippet of decompiled code and classify any vulnerabilities, using the format 'CWE-XXX' only.",
    "Detect vulnerabilities in the following decompiled code and categorize each strictly as 'CWE-XXX'. Do not add extra text.",
    "Please review the decompiled code and provide the corresponding CWE number for any vulnerabilities in the format 'CWE-XXX'.",
    "Examine this code for security weaknesses and respond with the appropriate CWE classification in 'CWE-XXX' format.",
    "Identify any vulnerabilities in the provided decompiled code and classify them using only 'CWE-XXX'. Exclude any additional commentary.",
    "As a reverse engineer, classify vulnerabilities in this code snippet. Respond with 'CWE-XXX', and avoid extraneous outputs.",
    "Survey this decompiled code. For any vulnerabilities found, classify strictly using 'CWE-XXX'. No other information needed.",
    "Conduct a vulnerability assessment on the decompiled code below and classify any findings as 'CWE-XXX'. Limit your response to this format.",
    "Check the decompiled code for vulnerabilities and provide their CWE identifiers in the format 'CWE-XXX' only.",
    "Your role as a security expert is to classify vulnerabilities in the following code. Please use 'CWE-XXX' format for your response.",
    "As a code analyst, identify and classify any vulnerabilities in this code. Respond in 'CWE-XXX' format only.",
]

name_prediction = [
    "Analyze the following decompiled code and predict the function name. Ensure the name is in a single word using camelCasing or snake_casing. No additional outputs.",
    "Review this decompiled code snippet and determine the function name. Provide your answer as a single word, following C/C++ naming conventions.",
    "Examine the provided decompiled code and predict the function's name. The response should be one word, using either camelCasing or snake_casing.",
    "Assess this decompiled code to predict the name of the function it represents. Respond with just one word, adhering to C/C++ language standards.",
    "Your task is to identify the function name from this decompiled code. Provide the name in one word, using appropriate C/C++ naming conventions.",
    "Inspect the code below and predict the function name, presenting it in a single word following camelCasing or snake_casing rules.",
    "Analyze the decompiled code snippet and provide the function's name in a single word format, following the conventions of camelCasing or snake_casing.",
    "Survey the provided decompiled code and identify the function name, which should be formatted in one word using standard C/C++ naming practices.",
    "Decode the function name from this decompiled code. The answer must be a single word in camelCasing or snake_casing. No extra content.",
    "Evaluate the following decompiled code and predict the function name using a single-word format, following standard naming conventions.",
    "As a code security expert, determine the function name from the decompiled code, adhering to the single-word, camelCase or snake_case format.",
    "Your expertise is needed to predict the function name in this decompiled code. Respond with only one word, using accepted C/C++ naming formats.",
    "Scrutinize this decompiled code to identify the function name. The name should be a single word in either camelCasing or snake_casing.",
    "Your role is to determine the function name from the provided decompiled code. Only one word is needed, conforming to C/C++ naming conventions.",
    "Decipher the function name from this decompiled code. Present your answer in a single word, using camelCasing or snake_casing only.",
    "Analyze the given decompiled code and provide the function name. Ensure the name is delivered in one word, using appropriate coding conventions.",
    "Investigate this decompiled code and predict the function name in a single-word format, following camelCasing or snake_casing.",
    "From the decompiled code below, predict the function name. Use only one word for the name, adhering to standard C/C++ naming rules.",
    "Examine the decompiled code provided and name the function. The response should be formatted in one word, using either camelCasing or snake_casing.",
    "Your task is to extract the function name from the following decompiled code. Deliver your prediction in one word, using snake_casing or camelCasing.",
]

description = [
    "Examine this decompiled code and describe its objective along with any vulnerabilities present. Your description should be generic and applicable broadly to decompiled code, avoiding specific names.",
    "Review the provided decompiled code. Provide a general description of its purpose and any potential security issues, without using specific function or variable names.",
    "Analyze the decompiled code below and explain its intended function and any vulnerabilities it may contain. Ensure the description is universal and does not mention specific names.",
    "Your task is to describe the general objective and identify any vulnerabilities in this decompiled code. Keep the description applicable to similar code by not specifying names.",
    "Assess the following decompiled code. Offer a description of what the code aims to achieve and any security weaknesses, ensuring the explanation remains generic.",
    "Investigate the provided decompiled code. Describe the core functionality and any vulnerabilities without referencing specific variables or functions.",
    "Explain the purpose and highlight any security issues in the decompiled code shown. Ensure the description is generic enough for use with other decompiled codes.",
    "Provide an overview of the objective and vulnerabilities of this decompiled code. Avoid specific names to maintain the general applicability of your description.",
    "Describe the primary objective and any vulnerabilities of the given decompiled code, ensuring your summary avoids specific function and variable names.",
    "Elaborate on the intent and security flaws of this decompiled code. Make sure your description is generic, suitable for use with similar codes.",
    "Interpret the following decompiled code by describing its general objective and any inherent vulnerabilities, omitting any specific identifiers.",
    "Your analysis should explain the main goal and potential security risks in this decompiled code, without using specific function or variable names.",
    "Examine this decompiled code and summarize its purpose and any vulnerabilities in a way that is applicable to other decompiled codes.",
    "Provide a generic description of this decompiled code's objective and vulnerabilities, ensuring it is useful for other codes without specific identifiers.",
    "Analyze and describe the general objective and vulnerabilities of the decompiled code provided, while ensuring your description is universally applicable.",
    "Offer a broad overview of what this decompiled code is meant to achieve and any vulnerabilities it may have, avoiding mentioning any specific names.",
    "Your task is to describe the goals and identify vulnerabilities of this decompiled code in a general manner suitable for decompiled environments.",
    "Conduct an analysis of the decompiled code, focusing on its purpose and vulnerabilities, and provide a description that could be applied generically.",
    "Interpret and describe the objectives and potential vulnerabilities of the decompiled code without referring to specific variable or function names.",
    "Describe the general objective and any vulnerabilities of this decompiled code, ensuring the description can be universally applied without specific names.",
]

def generate_response(model, tokenizer, query):
    # Ensure that the model and input tensors are on the same device
    device = next(model.parameters()).device
    
    # Tokenize inputs
    inputs = tokenizer(query, return_tensors="pt", padding=True, add_special_tokens=True)
    input_ids = inputs["input_ids"].to("cuda")

    # Generate output
    output = model.generate(input_ids=input_ids,max_new_tokens=300)
    
    # Decode and return response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def getPromptIdentification(row):
    
    instruction = random.choice(identification)
    code_field = 'stripped_decompiled_code'
    if pd.isnull(row['label']) or row['label'] == '[]':
        answer = 'NO'
    else:
        answer = 'YES'

    #prompt for codeLlama
    return "[INST] " + instruction + " Code: " + row[code_field].strip() + "[/INST]", answer
    
    #prompt for starcoder and codegen
#     return f'''Below is an instruction that describes a task, paired with an input code that provides further context. Write a response that appropiately completes the request
# ### Instruction\n 
# {instruction}

# ### Code\n 
# {row[code_field].strip()}

# ### Response\n ''', answer

def getPromptClassification(row):
    instruction = random.choice(classification)
    answer = row['label']
    code_field = 'stripped_decompiled_code'
    formatted_result = re.sub(r"\['CWE(\d+)'\]", r"CWE-\1", answer)
    #prompt depending on model
    return "<s>[INST] " + instruction + " Code: " + row[code_field].strip() + " [/INST]", formatted_result
    #return "[INST] " + instruction + " Code: " + row[code_field].strip() + " [/INST]", formatted_result
    # return f'''Below is an instruction that describes a task, paired with an input code that provides further context. Write a response that appropiately completes the request
    # ### Instruction:\n
    # {instruction}

    # ### Code:\n
    # {row[code_field].strip()}

    # ### Response:\n ''', formatted_result

def getPromptFunctionName(row):
    instruction = random.choice(name_prediction)
    answer = row['function']
    code_field = 'stripped_decompiled_code'
    
    #prompt format depending on model
    return "<s>[INST] " + instruction + " Code: " + row[code_field].strip() + " [/INST]", answer
    #return "<s>[INST] " + instruction + " Code: " + row[code_field].strip() + " [/INST]", answer
    # return f'''Below is an instruction that describes a task, paired with an input code that provides further context. Write a response that appropiately completes the request
    # ### Instruction:\n
    # {instruction}

    # ### Code:\n
    # {row[code_field].strip()}

    # ### Response:\n ''', answer
 
def getPromptDescription(row):
    instruction = random.choice(description)
    answer = row['description']
    code_field = 'stripped_decompiled_code'
    
    #prompt format depending on model
    return "<s>[INST] " + instruction + " Code: " + row[code_field].strip() + " [/INST]", answer   
    # return f'''Below is an instruction that describes a task, paired with an input code that provides further context. Write a response that appropiately completes the request
    # ### Instruction:\n
    # {instruction}

    # ### Code:\n
    # {row[code_field].strip()}

    # ### Response:\n ''', answer
        
def main():
    
    modelname = sys.argv[1]
    
    model = AutoModelForCausalLM.from_pretrained(
            modelname,
            #device_map="auto",
            #load_in_8bit=True,
             torch_dtype=torch.float16,
            trust_remote_code=True
            )
    
    model.bfloat16().to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(modelname, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    dataset = load_from_disk(os.environ['50k_x86_4_TASKS_SPLIT'])
    dataset_test = dataset['test']
    pred = []
    gt =[]
    iter = 0
    decomp_code = []
    # Iterate through each code
    for code in dataset_test:
        if iter >= 200:
            break 
        
        #filter based on architecture
        #if not (code['compiler_options'] == "['gcc', '-O0']" or code['compiler_options'] == "['gcc', '-O3']" or code['compiler_options'].startswith("['clang")):
        #if not code['compiler_options'].startswith("['aarch64"):
        #if not code['compiler_options'].startswith("['mips"):
        #if not (code['compiler_options'] == "['gcc', '-O0', '-m64']" or code['compiler_options'] == "['gcc', '-O3', '-m64']"):
           # continue
        
        input = code["stripped_decompiled_code"]
    
        if not input:
            continue
        
        split_token="[/INST]"
        #split_token="### Response:"
        
        #Identification prompt
        #prompt,ground = getPromptIdentification(code)
            
        #Identification prompt for codellama base model
        #prompt,ground = getPromptIdentificationCodeLlamaBase(code)
        #split_token="Answer="
        
        #Classification prompt
        #if pd.isnull(code['label']) or code['label'] == '[]':
           #continue
        #prompt,ground = getPromptClassification(code)
        
        #function name prompt
        #prompt,ground = getPromptFunctionName(code)
        
        #description prompt
        if pd.isnull(code['description']):
          continue
        prompt,ground = getPromptDescription(code)

        #Only for codegen context window
        #if len(prompt) > 1800:
            #continue
        
        gt.append(ground)
        decomp_code.append(input)
        response = generate_response(model, tokenizer, prompt)
        response_split = response.split(split_token)[-1].strip()
        
        if response is None or response == "":
            print("")
        pred.append(response_split)
        
        
        # Print the generated text
        print(f"Input:{input}")
        print(f"Ground truth output:\n{ground}")    
        print(f"Generated Answer:\n{response}")
        print(f"Generated Answer Split:\n{response_split}")
        print(f"sample:{iter}")
        iter+=1
    
    result = {
        "decompcode": decomp_code,
        "pred": pred,
        "gt": gt
    }

    resultFile = os.environ['50k_x86_4_TASKS_SPLIT_RESULTS']
    with open(resultFile, "w") as f:
        json.dump(result, f, indent=4)
        
    print("result has been saved in file")
        
    
if __name__ == "__main__":
    main()