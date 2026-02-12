from llm_sdk import Small_LLM_Model
from pydantic import BaseModel, PrivateAttr, model_validator
from typing import Dict, Optional, Any
import torch
import json
import numpy as np
import sys
import logging

logger = logging.getLogger(__name__)



class CallMeMaybe_LLM_Model(BaseModel):
    """Wrapper class of Small_LLM_Model, make use of the existing LLM class
    and implement own decode/encode method"""

    _llm: Optional[Small_LLM_Model] = PrivateAttr(None)
    _vocab: Optional[str] = PrivateAttr(None)
    _max_tokens_limit: int = 256
    _token_to_id: Dict[str, int] = PrivateAttr({})
    _id_to_token: Dict[int, str] = PrivateAttr({})
    _function_definitions: Optional[str] = None
    _system_prompt: Optional[Dict[str, str]] = None
    _system_prompt_tokens: Optional[Dict[str, list[int]]] = None
    _system_prompt_tensor: Optional[torch.Tensor] = None
    EOT_TOKEN_ID: int = 16141
    DEBUG_OUTPUT: bool = True


    @staticmethod
    def argmax(logits: list[float]) -> int:
        """
        Helper function for np.argmax()

        Args:
            logits (list[float]): list of floats you want to find the argmax

        Returns:
            int: index of the maximum value from the list
        """
        return int(np.argmax(logits))

    def model_post_init(self, __context: Any) -> None:
        """
        pydantic way to initialize some private members

        Args:
            __context (Any): passed by default

        Returns:
            None
        """

        self._llm = Small_LLM_Model()
        self._vocab = self._llm.get_path_to_vocabulary_json()

        with open(self._vocab, "r") as f:
            raw_vocabs = json.load(f)

        # reset the mappings
        self._token_to_id = {}
        self._id_to_token = {}

        # handling spaces
        for token_str, token_id in raw_vocabs.items():
            processed_token = token_str.replace('Ġ', ' ').replace(' ', ' ')
            self._token_to_id[processed_token] = token_id
            self._id_to_token[token_id] = processed_token


    def encode(self, text: str) -> torch.Tensor:
        """
        Encode the input text using internal vocab into Tensor.

        Equivalent of llm.tokenizer.encode().

        Args:
            text (str): string to encode.

        Returns:
            torch.Tensor: Tensor of the mappings.

        """
        # convert the text into list of single character token
        word_tokens = [self._token_to_id.get(char, 0) for char in text]

        while len(word_tokens) > 1:
            best_pair_id = float('inf')
            best_pair_idx = -1
            merged_token_id = None

            # 2. Look at every adjacent pair
            for i in range(len(word_tokens) - 1):
                # Convert IDs back to tokens to see if their combination exists
                pair_str = (
                    self._id_to_token[word_tokens[i]]
                    + self._id_to_token[word_tokens[i+1]]
                )

                if pair_str in self._token_to_id:
                    # the pair exists, let's keep it
                    current_id = self._token_to_id[pair_str]
                    # 3. PRIORITY: The pair with the LOWEST ID is the one that
                    # MUST merge first
                    if current_id < best_pair_id:
                        best_pair_id = current_id
                        best_pair_idx = i
                        merged_token_id = current_id

            # 4. If we found a valid merge, do it and restart the check
            if best_pair_idx != -1:
                new_tokens = word_tokens[:best_pair_idx]
                if merged_token_id is not None:
                    new_tokens.append(merged_token_id)
                new_tokens.extend(word_tokens[best_pair_idx + 2:])
                word_tokens = new_tokens
            else:
                # No more possible merges found in our vocab
                break
        return torch.tensor([word_tokens])

    def decode(self, ids: list[int]) -> str:
        """
        decode list of integer into text using LLM vocab
        """
        return "".join(self._id_to_token[i] for i in ids)

    def set_functions_definition(self, json_str: str) -> None:
        """
        validate input parameters, make sure it is valid JSON object
        and should contains "fn_name" in each object

        Args:
            json_str (str): valid JSON string to use as function definitions

        Raises:
            ValueError: Invalid JSON str
            KeyError: JSON object does not have a "fn_name", "args_names"
            attribute
        """
        
        try:
            fds = json.loads(json_str)
        except json.JSONDecodeError:
            raise ValueError("invalid JSON str")
        if not isinstance(fds, list):
            raise ValueError('array of objects expected')
        for fd in fds:
            if "fn_name" not in fd:
                raise KeyError('object does not have "fn_name" attribute')
            if "args_names" not in fd:
                raise KeyError('object does not have "args_names" attribute')

        self._function_definitions = json_str

        sp = f"""
You are helpful JSON generator, you will response only a valid JSON,
no explanation.
You always return JSON in this format {{ "fn_name": "fn_xxx",
"args": {{"xxx": "yyy"}} }}.
From provided JSON: {self._function_definitions}, choose the best
function and arguments to solve this question.
Question: """

        self._system_prompt = {}
        self._system_prompt['PRE_PROMPT'] = sp.strip()
        sp = "\nAnswer: { \"fn_name\" : \""
        self._system_prompt['POST_PROMPT'] = sp.strip()

        self._system_prompt_tokens = {}
        self._system_prompt_tokens['PRE_PROMPT'] = (
            self.encode(self._system_prompt['PRE_PROMPT'])
        )
        self._system_prompt_tokens['POST_PROMPT'] = (
            self.encode(self._system_prompt['POST_PROMPT'])
        )

    def get_custom_prompt_str(self, prompt: str) -> str:
        """
        Add the custom prompt into CMM System prompt

        Args:
            prompt(str): Prompt to find the best suit functions

        Returns:
            str: full prompts which will be used in the Autoregression process
        Raises:
            ValueError: Invalid prompt str or self._functions_definition
        """

        if not isinstance(prompt, str):
            raise ValueError("invalid prompt, str expected")
        if prompt.strip() == '':
            raise ValueError("invalid prompt, non-empty str expected")

        if self._function_definitions is None:
            raise ValueError("functions definition was not yet set")
        sp = f"""
You are helpful JSON generator, you will response only a valid JSON,
no explanation.
You always return JSON in this format {{ "fn_name": "fn_xxx",
"args": {{"xxx": "yyy"}} }}.
From provided JSON: {self._function_definitions}, choose the best
function and arguments to solve this question.
Question: {prompt}\n"
Answer: {{ "fn_name" : \""""

        return sp.strip()

    # def get_system_prompt_tensor(self, custom_prompt: str) -> torch.Tensor:
    #     """
    #     Get the Tensor from 3 sections, the Pre / prompt / Post

    #     Args:
    #         custom_prompt (str): prompt for looking up functions
    #     Returns:
    #         torch.Tensor: tensor cretaed from system + custom prompt
    #     """
    #     # if previously set, simply return the whol base prompt
    #     try:
    #         if (
    #             self._system_prompt['prompt'] == custom_prompt and 
    #             self._system_prompt_tokens['prompt'] is not None and
    #             self._system_prompt_tensor is not None
    #         ):
    #             print(f"cached tensor: {custom_prompt}", file=sys.stderr)
    #             return self._system_prompt_tensor
    #     except KeyError:
    #         pass

    #     self._system_prompt['prompt'] = custom_prompt
    #     self._system_prompt_tokens['prompt'] = (
    #         self.encode(self._system_prompt['prompt'])
    #     )
    #     # print(f"PRE: {len(self._system_prompt_tokens['PRE_PROMPT'])}, PROMPT: {len(self._system_prompt_tokens['prompt'])}, POST: {len(self._system_prompt_tokens['POST_PROMPT'])}")
    #     # print(f"Types: {type(self._system_prompt_tokens['PRE_PROMPT'][0])}, {type(self._system_prompt_tokens['prompt'][0])}, {type(self._system_prompt_tokens['POST_PROMPT'][0])}")

    #     self._system_prompt_tensor = torch.cat(
    #         [
    #             self._system_prompt_tokens['PRE_PROMPT'],
    #             self._system_prompt_tokens['prompt'],
    #             self._system_prompt_tokens['POST_PROMPT']
    #         ],
    #         dim=1
    #     )
    #     # combined = (
    #     #     self._system_prompt_tokens['PRE_PROMPT'] +
    #     #     self._system_prompt_tokens['prompt'] +
    #     #     self._system_prompt_tokens['POST_PROMPT']
    #     # )
    #     # self._system_prompt_tensor = torch.Tensor(combined)
        
    #     return self._system_prompt_tensor

    def get_system_prompt_ids(self, custom_prompt: str) -> list[int]:
        """
        Get the Tensor from 3 sections, the Pre / prompt / Post

        Args:
            custom_prompt (str): prompt for looking up functions
        Returns:
            torch.Tensor: tensor cretaed from system + custom prompt
        """
        # if previously set, simply return the whole base prompt
        user_prompt = self.encode(custom_prompt)
        return (
                    self._system_prompt_tokens['PRE_PROMPT'] +
                    user_prompt +
                    self._system_prompt_tokens['POST_PROMPT'] 
                )

        # try:
        #     if (
        #         self._system_prompt['prompt'] == custom_prompt and 
        #         self._system_prompt_tokens['prompt'] is not None and
        #         self._system_prompt_tensor is not None
        #     ):
        #         print(f"cached tensor: {custom_prompt}", file=sys.stderr)
        #         return self._system_prompt_tensor
        # except KeyError:
        #     pass

        # self._system_prompt['prompt'] = custom_prompt
        # self._system_prompt_tokens['prompt'] = (
        #     self.encode(self._system_prompt['prompt'])
        # )

        # return (
        #     self._system_prompt_tokens['PRE_PROMPT'] +
        #     self._system_prompt_tokens['prompt'] +
        #     self._system_prompt_tokens['POST_PROMPT']
        # )

    def _push_json_chunk(self, src: list[str], chunk: str) -> bool:
        chunk = chunk.replace('Ċ', '\n')
        src.append(chunk)
        temp = ''.join(src)
        # if self.DEBUG_OUTPUT:
        #     logger.info("_push_json_chunk():", temp)

        try:
            # add quick JSON ending check, so doesn't have to fully parse
            if "}" not in chunk:
                return False
            # extra check to remove special token
            json.loads(temp)
        except json.JSONDecodeError:
            return False
        return True

    def _cleanup_json(
            self,
            src: list[str],
            prompt: str,
            reset_as_well: bool = True
            ) -> object:
        """
        Adjust the obejct so the output JSON is in "prompt", "fn_name", "args"
        format

        Args:
            src (list[str]): List of the (str) token
            prompt (str): prompt to be added to the returned object
            reset_as_well (bool): cleaned up the src list once done

        Returns:
            object with pre-defined formatted
        """
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

    def prompt_selection(self, custom_prompt: str) -> object:
        # 1. Get the base IDs (System + User + Post)
        # Ensure this returns a FLAT LIST of ints, not a Tensor yet.
        ids = self.get_system_prompt_ids(custom_prompt) 
        
        # 2. Pre-fill the start of the JSON so the LLM doesn't have to "guess" it
        prefill_text = '{"fn_name": "'
        ids.extend(self.encode(prefill_text))
        
        # Initialize your result accumulator with the prefilled text
        json_result = [prefill_text]

        for i in range(self._max_tokens_limit):
            # This is the slow part (The Model Math)
            out_tokens = self._llm.get_logits_from_input_ids(ids)
            next_token_id = self.argmax(out_tokens)
            
            # Update IDs so the model remembers this token for the next round
            ids.append(next_token_id)
            
            # Decode only the NEW token
            new_text = self.decode([next_token_id])
            json_result.append(new_text)

            # 3. Quick exit check
            if "}" in new_text:
                # Only now do the expensive JSON validation/cleanup
                if self._push_json_chunk(json_result, ""): # Simplified check
                    return self._cleanup_json(json_result, custom_prompt)

            if next_token_id == self.EOT_TOKEN_ID:
                break

        return self._cleanup_json(json_result, custom_prompt)

    def __prompt_selection(self, custom_prompt: str) -> object:
        """
        Select the best suit functions , arguments from the given prompt,
        returns as a proper object

        Args:
            custom_prompt (str): prompt to find the function

        Returns:
            object: JSON comapatible Dict which will contains prompt, fn_name
                  and args
        """

        torch_tensors = self.get_system_prompt_tensor(custom_prompt)

        ids = torch_tensors[0].tolist()
        decoded = self._llm._decode(ids)

        json_result = ["{", '"fn_name"', ": \""]
        if self._llm is None:
            raise RuntimeError("Model was not properly initialized")

        for i in range(0, self._max_tokens_limit):
            out_tokens = self._llm.get_logits_from_input_ids(ids)
            next_token_id = CallMeMaybe_LLM_Model.argmax(out_tokens)
            decoded = self.decode([next_token_id])
            # print(f"{next_token_id:<10} {decoded}", end='')

            if self._push_json_chunk(json_result, decoded):
                obj = self._cleanup_json(json_result, custom_prompt)
                return obj

            if next_token_id == self.EOT_TOKEN_ID:
                break
            ids.append(next_token_id)

        obj = self._cleanup_json(json_result, custom_prompt)
        return obj
