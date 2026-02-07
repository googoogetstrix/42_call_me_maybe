from llm_sdk import Small_LLM_Model
from .cmm_llm import CallMeMaybe_LLM_Model
import json
import argparse
from pathlib import Path
import numpy as np
import math


def softmax(logits: list[float]) -> list[float]:
    max_logit = max(logits)              # stability trick
    exps = [math.exp(x - max_logit) for x in logits]
    s = sum(exps)
    return [x / s for x in exps]


def argmax(logits):
    return int(np.argmax(logits))


def push_json_chunk(src: list[str], chunk: str) -> bool:
    chunk = chunk.replace('Ċ', '\n')
    src.append(chunk)
    temp = ''.join(src)
    print(temp)

    try:
        # extra check to remove special token
        
        json.loads(temp)
    except json.JSONDecodeError:
        return False
    return True


def cleanup_json(src: list[str], prompt:str , reset_as_well=True) -> object:
    obj = json.loads(''.join(src))
    return_obj = {}
    return_obj['prompt'] = prompt
    for valid_prop in ["fn_name", "args"]:
        try:
            if obj[valid_prop]:
                return_obj[valid_prop] = obj[valid_prop]
        except KeyError:
            continue
    if reset_as_well:
        src = []
    return return_obj


# def TEST_main():

#     # model = Small_LLM_Model()
#     # path = model.get_path_to_vocabulary_json()
#     # print(path)
#     model = CallMeMaybe_LLM_Model()
#     print(model._id_to_token)
#     # print(CallMeMaybe_LLM_Model.model_config.get('slots'))
#     # print(model.__class__.model_config.get('slots') )
#     source = 'Bark thrice and roll over!'
#     source = 'Get the hell out of here, mail me someone@somesite.com'
#     # out = model.encode(source)
#     # print(out)
#     # text = model.decode(out)
#     # print(text)

#     # out = model.encode('Bark')
#     # print(out)
#     # text = model.decode(out)
#     # print("_",text,"_", sep='')

#     # ids = out

#     # ids = model._llm._tokenizer.encode(source)
#     USE_CUSTOM = False

#     if USE_CUSTOM:
#         ids = model.encode(source)
#         print("CUSTOM Full encode:", ids)
#         decoded = model._llm._tokenizer.decode(ids)
#         print(f"Full Decode: '{decoded}'")
#     else:
#         ids = model._llm._tokenizer.encode(source)
#         print("LLM Full encode:", ids)
#         decoded = model._llm._tokenizer.decode(ids)
#         print(f"Full Decode: '{decoded}'")



    # Check what ID 0 actually is in your vocab
    # print(f"Token ID 60024 is: _'{model._llm._tokenizer.convert_ids_to_tokens(558)}'_", sep='')




def main():
    """
    """

    try:
        parser = argparse.ArgumentParser(
            description="Parser for call me maybe"
        )
        parser.add_argument("--input",
                            type=str,
                            default="data/input/function_calling_tests.json",
                            help="Input file")
        parser.add_argument("--output",
                            type=str,
                            default="data/output/function_calling_results.json",
                            help="Output file")
        args = parser.parse_args()

        input_path = Path(args.input)
        output_path = Path(args.output)
        definition_path = 'data/input/functions_definition.json'

        model = CallMeMaybe_LLM_Model()

        with open(definition_path, "r") as f:
            f_def = f.read()
        with open(input_path, "r") as f:
            prompts = json.load(f)

        model.set_functions_definition(f_def)
        items = []
        for k in prompts:
            out = model.prompt_selection(k['prompt'])
            items.append(out)
            print(out)
        with open(output_path, "w") as f:
            print(items)
            json.dump(items, f, indent=4, ensure_ascii=False)

    except Exception as e:
        print("Error:", e)
        raise e



def MY_main():
    """
    Yahoo
    """

    MAX_TOKENS_LIMIT = 256
    EOT_TOKEN_ID = 16141
    DEBUG_SCRIPT = True
    try:
        parser = argparse.ArgumentParser(
            description="Parser for call me maybe"
        )
        parser.add_argument("--input",
                            type=str,
                            default="data/input/function_calling_tests.json",
                            help="Input file")
        parser.add_argument("--output",
                            type=str,
                            default="data/output/function_calling_results.json",
                            help="Output file")
        args = parser.parse_args()

        input_path = Path(args.input)
        output_path = Path(args.output)
        definition_path = 'data/input/functions_definition.json'

        print(f"Reading from {input_path}")
        print(f"Writing to {output_path}")
        args = parser.parse_args()

        with open(input_path, "r") as f:
            prompts = json.load(f)
        with open(definition_path, "r") as f:
            f_def = json.load(f)
        
        model = CallMeMaybe_LLM_Model()
        items = []

        # each_prompt = "Reverse the string 'world'"
        # each_prompt = "Greet shrek"
        # each_prompt = "Is 4 an even number?"

        for k in prompts:
            json_result = ["{", '"fn_name"', ":"]
            # each_prompt = "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'"
            each_prompt = k['prompt']
            prompt = "You are helpful JSON generator, you will response only a valid JSON, no explanation.\n"
            prompt += 'You always return JSON in this format { "fn_name": "fn_xxx", "args": {"xxx": "yyy"} }.\n'
            prompt += f"From provided JSON: {f_def}, choose the best function and arguments to solve this question.\n"
            prompt += f"Question: {each_prompt}\n"
            prompt += 'Answer: { "fn_name" : '

            USE_LLM_TOKENIZER = False
            print("===========================================")
            print(prompt)
            print("===========================================")
            print(f"USE_LLM_TOKENIZER: {USE_LLM_TOKENIZER}")
            
            if USE_LLM_TOKENIZER:
                torch_tensors = model._llm._encode(prompt)
            else:
                torch_tensors = model.encode(prompt)

            ids = torch_tensors[0].tolist()

            for i in range(0, MAX_TOKENS_LIMIT):
                out = model._llm.get_logits_from_input_ids(ids)
                next_token_id = argmax(out)
                if DEBUG_SCRIPT: 
                    print(f"{next_token_id:>10}", end=": ")
                if USE_LLM_TOKENIZER:
                    decoded = model._llm._decode([next_token_id])
                else:
                    decoded = model.decode([next_token_id])

                if push_json_chunk(json_result, decoded):
                    print("INSIDE CHUNK")
                    obj = cleanup_json(json_result, each_prompt)
                    items.append(obj)
                    break

                if next_token_id == EOT_TOKEN_ID:
                    print(" NEXT_TOKEN_ID BREAK! ")
                    break

                ids.append(next_token_id)

    except Exception as e:
        print(e)
        raise e

    with open(output_path, "w") as f:
        print("FINALLY:")
        print(items)
        json.dump(items, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
