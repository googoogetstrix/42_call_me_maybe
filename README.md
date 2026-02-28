*This project has been created as part of the 42 curriculum by __REDACTED__*


# Instructions 

## Requirements:
 - Python3.1x , [uv](https://github.com/astral-sh/uv)
 ```
 > pip install --user pipx
 > pipx install uv
 ```
 - A LOT of diskspace, around 7.5 GB
 - Patience, you'll need lots of it

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
(The actual prompt ends here)
Answer: { "fn_name: "
```
This somehow forced the LLM to recognized that the part of the JSON response is already there, it has to complete it from the template. This solve half of the issue, if you're not stopping the LLM at the right time, it tends to give you extra garbage tokens.

The solution for the latter half, is to try to check if the JSON responses output from the LLM is already finished, i use the json.loads() to check if the appended string is valid JSON already, then break the loop and proceed to the cleaning up process.

JSON cleaning up, this will make sure that the output JSON is in correct order, no extra stuff were added to the output.

## Resources:
https://en.wikipedia.org/wiki/Large_language_model


## Algorithm explanation

```
Prompt -> Tokenization -> Input IDs -> LLM -> Logits -> Next Token Selection
```

As described in the subject, the plainer description is the process of LLM is repeatedly feeding the information (both prompts & answers) into the LLM until it stops. But there're lots of conversions need to be done during the process (tokens, logits)

Now, let's look into what we have in the Small_LLM_Model, we have 2 public functions we can make use _*GODDAMNIT, THERE'S NEW VERSION OF SUBJECT WHICH IS A LOT OF EASIER*_
- get_logits_from_input_ids()
- get_path_to_vocab_file()

So, you have to implement your own encode() and decode() using the LLM's dictionary file, the simple one may not work ..see [encoding](#encoding) 

these 2 functions will allow you to turn string prompt into token list (back and forth), which you can feed into get_logits_from_input_ids() and start the autoregression process

the output from get_logits_from_input_ids() will return the logits which you can make use of by find the highest value. 

Then goe to the process which was mentioned in the [output format](#output-format)

## Design decisions
The Small_LLM_Model class was supposed to be left untouched. So the design pattern i chose is the Proxy (custom class encapsulate predefined class)  


## Performance analysis
Since the execution time is the part of the evaluation.Putting effort to shave off every seconds contribute to the project. See [Execution times exceeding the expected time](#execution-times-exceeding-the-expected-time)


## Challenges faced
### Encoding
- Sherk issue with the BPE merge table
The additional rule required for encoding.The issues with the common greedy algorithm shows on the prompt "greet shrek". By using simple greedy algorithm, the encoding found the best match for "shrek" token as ["shr" , "e", "k"], which make the invalid token in argument section ... technically, the LLM provide the merges.txt which explain how to properly merge the token, but since we cannot access the private attribute of the Small_LLM_Model, we need to use the alternatives

for example, these tokens exist in the dictionary
```
sh (ID: 927) — High popularity (learned early)
shr (ID: 66039) — Low popularity (learned late)
rek (ID: 41861) — Medium popularity
```
by using simple greedy algorithm, the nearest macth should be ["shr", "e", "k"] ...which could be fine in some higher parameters LLM, but since the one we're using is low-parameters, and the token is kinda "ugly", the model apruptly ends the word, thus make a false argument token

BPE (Byte Pairing Encoding) helps by trying to make the most "nice" pairs, instead of just the "longest" ... so no tokens should left as an orphan token, the result are the equally "strong" tokens instead of a "very strong" token along 2 weak tokens

### Output format
 - ANY valid prompts should be answered, no matter how BS they are, the function name, the arhuments & tehir types have to be valid and consist to functions definition input. So by default, if no function or not in the list is returned from LLM, use the first one from the defintion
 - Extra step to force the output response by initializing the order of attributes as specified from the subject, make sure they have the correct type by casting them according to the definitions
 - If the TinyLM is not picked the correct one from the BS prompt, let's hope it did their best and we'll just handle (and verfiy) the output JSON again


### Execution times exceeding the expected time
- most of the LLM related operations is quite painfully slow, so cache whatever you can, stop early if you can ... and hope you won;t spend too much time on handling the errors
- In this script, the beginning (the PRE) of the system prompts and the sample JSON output format (the POST) are always the same, so why bother encode them again and again? just use the cached and using encode() to handle the actual prompt, that will shave you down a lot especially if you have extra detail system prompt
- since we know that the output from the LLM should be in a valid JSON format and since we cannot force TinyML to blabber the output after that, we make sure by checking each returned token (from softmax logits) that if the content contains "}" , it could be the end of the output JSON .. you simply test by parsing the "JSON" section if it's actually done, then you can skip from the rest of the loop uintil the endoftext token


## Testing strategy
- starts with the file structures, handling the basic IOs
- then handling the of the LLM, structure & validity of the input files
- then test with the subject's prompts, try some borderline BS prompts just to make sure the script still give correct answers, the go fully BS , just to make sure that the formats are OK.. no weird function name or arguments & types were output

## Example usage:

 run 
 ```
 uv run python3 -m src
 ``` 
 for the basic (default) usage

 optional runs:
```
uv run python -m src --input data/input/SHORT_function_calling_tests.json --output out.json
```

* note that the functions definition file are fixed in ```./data/input/functions_defintion.json```


 ## Generative AI Helps
- The explanation about the BPE.
- Makefile creation and exclude option for flake8, mypy.
- Explain about the mypy error messages & what exactly needs to fix.


<!-- TODOs
- Prompt Caching    [DONE?]
- check about the output folders [DONE]
- quick returns if the chunk recently hasn't contains }, simply return False [DONE]
- cleanup old Python < 3.10 codes [DONE]
- the ToolSelection model which generated from function definition, make sure the output always valid [DONE]
- pytest testunit, handling the total BS prompts [DONE]
- readme [DONE] -->