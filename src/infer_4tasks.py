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

identification = [
    "As a software security auditor, evaluate this decompiled code. Are there any vulnerabilities present? Reply with 'YES' or 'NO' only.",
    "Inspect the given decompiled code for security flaws. If you find any vulnerabilities, respond with 'YES'. If none are detected, reply with 'NO'.",
    "Check this code for security risks. If vulnerabilities are present, simply respond 'YES'. If there are none, respond 'NO'.",
    "You are tasked with assessing this decompiled code for security vulnerabilities. Are there any? Answer 'YES' or 'NO' without elaboration.",
    "Analyze the code snippet below. If you find any security vulnerabilities, respond with 'YES'. If none, answer 'NO'.",
    "Review the following decompiled code. Are there any security flaws? Reply 'YES' for vulnerabilities present, 'NO' if absent.",
    "As a code analysis professional, inspect this code for security issues. Reply only with 'YES' if any are found, otherwise, 'NO'.",
    "Evaluate the provided decompiled code for potential vulnerabilities. If any are identified, respond with 'YES'. If none, respond with 'NO'.",
    "Determine if the following code contains security vulnerabilities. A simple 'YES' or 'NO' will suffice as your response.",
    "Assess this decompiled code for any security risks. If you find vulnerabilities, reply with 'YES'. If it is secure, reply 'NO'.",
    "Your task is to verify if there are any vulnerabilities in this code. If any are present, respond with 'YES'. If there are none, respond with 'NO'.",
    "Look through the provided code snippet. Are there security vulnerabilities? Please answer only with 'YES' or 'NO'.",
    "As a reverse engineering expert, scrutinize this decompiled code for vulnerabilities. If any are present, respond 'YES'. If not, respond 'NO'.",
    "You are evaluating the security of this code. Do you detect any vulnerabilities? Just answer 'YES' or 'NO'.",
    "Analyze the decompiled code below for any security issues. If vulnerabilities exist, respond with 'YES'. If none, respond with 'NO'.",
    "Survey this code snippet for security vulnerabilities. Identify any by responding with 'YES' or 'NO' only.",
    "As a security analyst, inspect the provided decompiled code. Are there vulnerabilities? Reply solely with 'YES' or 'NO'.",
    "Your expertise is required to identify security flaws in this code. Do you find any? Answer 'YES' or 'NO' without further comments.",
    "Review the decompiled code for potential vulnerabilities. Should any be found, respond only with 'YES'. If none, respond with 'NO'.",
    "Examine this decompiled code snippet. Is there any security vulnerability? Please answer with either 'YES' or 'NO' only."
]

classification = [
    "Evaluate the code snippet provided and classify any vulnerability using the format 'CWE-XXX'. Do not include any additional text.",
    "Inspect this decompiled code and identify any vulnerabilities by replying with their CWE numbers in the format 'CWE-XXX'.",
    "Analyze the following code for security vulnerabilities and categorize them strictly in 'CWE-XXX' format. Avoid any extra information.",
    "Examine the decompiled code snippet and provide the CWE classification of any vulnerabilities found, using only 'CWE-XXX'.",
    "Review the given code and indicate any vulnerabilities by responding solely with their 'CWE-XXX' classification. No additional details needed.",
    "Your task is to identify and classify any vulnerabilities in the code below using the format 'CWE-XXX' only.",
    "Scan the decompiled code for any vulnerabilities and specify them in the format 'CWE-XXX'. No further commentary required.",
    "Inspect the provided code snippet and classify any vulnerabilities found using the designation 'CWE-XXX'. Provide no extra outputs.",
    "As a security expert, classify any vulnerabilities in this code as 'CWE-XXX'. Avoid any additional text.",
    "Analyze this decompiled code and classify any vulnerabilities found using the format 'CWE-XXX' only.",
    "Identify vulnerabilities in the provided code snippet and categorize each strictly as 'CWE-XXX'. No extra information is necessary.",
    "Please review the given decompiled code and respond with the corresponding CWE number for any vulnerabilities in 'CWE-XXX' format.",
    "Examine the code for security issues and respond with their appropriate CWE classifications in 'CWE-XXX' format.",
    "Identify any vulnerabilities in this decompiled code and classify them using only 'CWE-XXX'. Exclude any additional content.",
    "As a code security specialist, classify any vulnerabilities in this snippet using 'CWE-XXX'. Do not provide extra details.",
    "Survey the provided decompiled code. For any vulnerabilities found, classify strictly using 'CWE-XXX'. No other information is needed.",
    "Conduct an assessment of the decompiled code below and classify any vulnerabilities as 'CWE-XXX'. Limit your response to this format.",
    "Check the given decompiled code for vulnerabilities and provide their CWE identifiers in the format 'CWE-XXX' only.",
    "Your role as a code analyst is to classify any vulnerabilities found in the code. Please respond in 'CWE-XXX' format only.",
    "As a security auditor, identify and classify any vulnerabilities in this code. Respond solely in 'CWE-XXX' format."
]

name_prediction = [
    "Review the decompiled code below and predict the function name. Provide it as a single word using camelCasing or snake_casing. No additional content.",
    "Analyze the given decompiled code and determine the function name. Your answer should be one word, following standard naming conventions.",
    "Examine this decompiled code snippet and predict the function's name. Use a single word, formatted in camelCasing or snake_casing.",
    "Assess the provided decompiled code and provide the function name. Ensure the name is one word, using C/C++ naming standards.",
    "Your task is to identify the function name from this decompiled code. Provide the name in one word, using camelCasing or snake_casing.",
    "Inspect the following code and predict the function name. Present it as a single word, following camelCasing or snake_casing rules.",
    "Analyze the decompiled code and provide the function name in one word, using either camelCasing or snake_casing conventions.",
    "Survey the decompiled code provided and identify the function name. Format it as one word, following standard naming practices.",
    "Decode the function name from the given decompiled code. The answer should be a single word in camelCasing or snake_casing.",
    "Evaluate this decompiled code snippet and predict the function name using a single-word format, following naming conventions.",
    "As an expert, determine the function name from the decompiled code. Respond with one word in camelCasing or snake_casing.",
    "Your expertise is needed to predict the function name from this decompiled code. Use only one word, following naming conventions.",
    "Scrutinize the decompiled code to identify the function name. Provide it in a single word, using camelCasing or snake_casing.",
    "Your task is to determine the function name from the provided decompiled code. Use one word, conforming to standard naming conventions.",
    "Decipher the function name from the decompiled code below. Present it in a single word, using camelCasing or snake_casing.",
    "Analyze this decompiled code and provide the function name. Ensure it is in one word, using camelCasing or snake_casing.",
    "Investigate the given decompiled code and predict the function name. Use a single-word format, following camelCasing or snake_casing.",
    "From the decompiled code provided, predict the function name. The name should be one word, adhering to naming conventions.",
    "Examine the provided decompiled code and name the function. Respond with one word, using camelCasing or snake_casing.",
    "Your task is to extract the function name from the decompiled code below. Provide it in one word, using camelCasing or snake_casing.",
]

description = [
    "Evaluate this decompiled code and describe its purpose along with any potential vulnerabilities. Ensure your description is generic and avoids specific names.",
    "Review the given decompiled code. Provide a broad description of its objective and any security issues, avoiding the use of specific function or variable names.",
    "Analyze the following decompiled code and explain its intended functionality and any vulnerabilities it might have. Keep the description universal without mentioning specific names.",
    "Your task is to describe the general purpose and identify any vulnerabilities in this decompiled code. Ensure the description remains generic and avoids specific identifiers.",
    "Assess the decompiled code provided. Offer a general description of its objective and any security weaknesses, ensuring the explanation is applicable to similar code.",
    "Investigate this decompiled code. Describe its core functionality and any vulnerabilities without referencing specific functions or variables.",
    "Explain the intended purpose and highlight any security issues in the decompiled code shown. Ensure the description is generic enough for use with other decompiled codes.",
    "Provide an overview of the objective and vulnerabilities of this decompiled code. Avoid specific names to maintain the general applicability of your description.",
    "Describe the main objective and any vulnerabilities of the given decompiled code, ensuring your summary avoids specific function and variable names.",
    "Elaborate on the intent and security flaws of this decompiled code. Make sure your description is generic and suitable for similar codes.",
    "Interpret the provided decompiled code by describing its general objective and any inherent vulnerabilities, omitting specific identifiers.",
    "Your analysis should explain the main goal and potential security risks in this decompiled code without using specific function or variable names.",
    "Examine this decompiled code and summarize its purpose and any vulnerabilities in a way that is applicable to other decompiled codes.",
    "Provide a generic description of the decompiled code's objective and vulnerabilities, ensuring it is useful for other codes without specific identifiers.",
    "Analyze and describe the general objective and vulnerabilities of the provided decompiled code, ensuring your description is universally applicable.",
    "Offer a broad overview of what this decompiled code aims to achieve and any vulnerabilities it may have, avoiding mentioning specific names.",
    "Your task is to describe the goals and identify vulnerabilities of this decompiled code in a manner suitable for decompiled environments.",
    "Conduct an analysis of the decompiled code, focusing on its purpose and vulnerabilities, and provide a description that could be applied generically.",
    "Interpret and describe the objectives and potential vulnerabilities of the decompiled code without referring to specific variable or function names.",
    "Describe the general objective and any vulnerabilities of this decompiled code, ensuring the description can be universally applied without specific names."
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
  
    modelname = "../../x86_codeLlama"
    
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
    
    dataset = load_from_disk('../../data/mvd_rebuilt_50k_x86_4_tasks_split')
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
        prompt,ground = getPromptIdentification(code)
            
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
        #if pd.isnull(code['description']):
          #continue
        #prompt,ground = getPromptDescription(code)

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

    resultFile = "../../Experiments_results/result_X86_50k_codeLlama_200_samples_x86_identification.json"
    with open(resultFile, "w") as f:
        json.dump(result, f, indent=4)
        
    print("result has been saved in file")
        
    
if __name__ == "__main__":
    main()