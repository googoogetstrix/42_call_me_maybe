This project has been created as part of the 42 curriculum by bworrawa


Instructions 
evaluation process 
    check directory structure
    > tree    

```
uv run python -m src --input data/input/SHORT_function_calling_tests.json --output out.json
```

TODOs

- Prompt Caching    [DONE?]
- check about the output folders
- quick returns if the chunk recently hasn't contains }, simply return False [DONE]

- cleanup old Python < 3.10 codes
- the ToolSelection model which generated from dunction definition, make sure the output always valid
- pytest testunit, handling the total BS prompts
- readme

 Time Limit Solutions
- precached the system prompt tokens, only do the custom_prompt on each loop
- Shaved the JSON functions definition, try remove the args_types, which couldn't be trusted from LLM and have to enforce manually


# Description

*Call me maybe* is the project that wants you to learn about how the LLM works in the low-level, and the adaptation into MCP.

The goal is to use the provided TinyLM class, a given set of available functions (functions_definition.json) JSON file as and a set of prompts JSON file (function_calling_tests.json), choose the best suit function from the provided list, extract or create what could be used as the arguments for such function. Then output as a JSON file in given format.


## TinyLM - Tiny Language Model 
A Transformer -based Neuron Network Model which can process the natural languages and "answer" your question to some extent based on it's own knowledge and context given. The Tiny prefix usually describe that the model was trained on fewer than a billion parameters.

The LLM (Large Language Model) is more generic term on this kind of model, but the models were trained on much larger parameters .. the exact number of parameters to use to classify the LM is still debatable

## Prompt
The instructions / query / question as plain text given to LLM to predict the output. 

The prompt text itself could not be processed directly by the LLM, it requires encoding so the LLM can use the regression on such numbers ..then we need to decode them back into plain text again

The LLMs use the Autoregression process to generate the output, meaning that they try to predict the next "token" from it's context until there's nothing to output anymore (`<endoftext>`) or maximum tokens reached.


## "Small language models are notoriously unreliable at generating structured output"
After spent hours into prompt to make it produce the perfect JSON for the output, i failed. No matter how desriptive, how compact the prompt is, the LLM is ready to ignore yours. There're always be explanation, the prelude, and some hallucination text along in the output.

I also tried to mess with the output logits, but there's no easy way to try parsing the JSON from that unless you implement the parser directly and knows which kind of token you'll expect next.

The solution? Since the LLM is Autoregression and try to feed itself over and over again, i forced the beginning of the preferred JSON output into the "initial" prompt itself, so the prompt will end with:
```
Blah Blah
Answer: { "fn_name: "
```
this somehow forced the LLM that the part of the JSON response is already there, it has to complete it from the template. This solve half of the issue, if you're not stopping the LLM at the right time, it tends to give you extra garbage tokens.

The solution for the latter half, is to try to check if the JSON responses output from the LLM is already finished, i use the json.loads() to check if the appended string is valid JSON already, then break the loop and proceed to the cleaning up process.

JSON cleaning up, this will make sure that the output JSON is in correct order, no extra stuff were added to the output.

## Resources:
https://en.wikipedia.org/wiki/Large_language_model

## Instructions

## Resources
# NOT YET DONE


## Algorithm explanation
#### Algorithm explanation: Describe your constrained decoding approach in detail



## Design decisions
#### Explain key choices in your implementation

## Performance analysis
#### Discuss accuracy, speed, and reliability of your solution


## Challenges faced
(Document difficulties encountered and how you solved them)
### Encoding
- Sherk issue with the BPE merge table
The additional rule required for encoding, the issues with the common greedy algorithm shows on the prompt "greet shrek". By using simple greedy algorithm, the encoding found the best match for "shrek" token as ["shr" , "e", "k"], which make the invalid token in argument section ... technically, the LLM provide the merges.txt which explain how to properly merge the token, but since we cannot access the private attribute of the Small_LLM_Model, we need to use the alternatives

for example, these tokens exist in the dictionary
```
sh (ID: 927) — High popularity (learned early)
shr (ID: 66039) — Low popularity (learned late)
rek (ID: 41861) — Medium popularity
```
by using simple greedy algorithm, the nearest macth should be ["shr", "e", "k"] ...which could be fine in some higher parameters LLM, but since the one we're using is low-parameters, and the token is kinda "ugly", the model apruptly ends the word, thus make a false argument token

BPE (Byte Pairing Encoding) helps by trying to make the most "nice" pairs, instead of just the "longest" ... so no tokens should left as an orphan token, the result are the equally "strong" tokens instead of a "very strong" token along 2 weak tokens


## Testing strategy
#### Describe how you validated your implementation

## Example usage:
#### Provide clear examples of running your progr

 run
 ```
 uv run python3 -m src
 ```

 Generative AI Helps
 The explanation about the BPE
 Makefile creation and exclude option for flake8, mypy



