from llm_sdk import Small_LLM_Model
from pydantic import BaseModel, PrivateAttr
from typing import Any, TypedDict, Annotated

import json
import numpy as np
import sys
import logging

logger = logging.getLogger(__name__)


class LLMResponseFunctionSchema(TypedDict):
    fn_name: str
    args: dict[str, Any]


class FunctionSchema(TypedDict):
    fn_name: str
    args_types: dict[str, Any]
    args_names: list[Any]
    return_type: str


FunctionsRepoType = Annotated[
    list[FunctionSchema],
    "The available functions can be selected"
]


class ToolSelection(BaseModel):
    """
    DTO(Data Transfer Object) class as a map between input functions definition
    and the expected output JSON
    """
    prompt: str
    fn_name: str
    args: dict[str, Any]
    args_types: dict[str, Any]
    args_names: list[Any]
    is_complete: bool = False

    def export(self, mode: str = "str") -> str | dict[str, Any]:
        """
        Returns ready to use formatted JSON on this item

        Returns:
            str of the python "JSON" dict in the subject specification format
        Raises:
            AttributeError: if any of the attributes weren't previously set
        """
        if self.fn_name is None:
            raise AttributeError("the fn_name attribute was not set")
        if self.args is None:
            raise AttributeError("the args attribute was not set")
        if self.prompt is None:
            raise AttributeError("the prompt attribute was not set")
        out = {
            "prompt": self.prompt,
            "fn_name": self.fn_name,
            "args": self.args
        }

        return json.dumps(out) if mode == "str" else out

    @classmethod
    def create_safe_instance(
        cls,
        custom_prompt: str,
        llm_response: dict[str, Any],
        repo: list[FunctionsRepoType]
    ) -> "ToolSelection":
        try:

            repo[0]

            fx_ptr = [
                f for f in repo if f['fn_name'] == llm_response['fn_name']
                ]
            matched_fx = fx_ptr[0]
            cls.is_complete = True

        except IndexError:
            matched_fx = repo[0]
        except Exception as e:
            raise e

        # make arguments list in preferred types
        temp_args = {}
        cls.args = {}
        for arg, arg_type in matched_fx['args_types'].items():
            # set whatever returned from llm_responses to cls.args
            try:
                cls.args[arg] = llm_response['args'][arg]
            except AttributeError:
                cls.args[arg] = 0

            if arg_type == 'float':
                try:
                    temp_args[arg] = float(cls.args[arg])
                except KeyError:
                    temp_args[arg] = 0.0
            elif arg_type == 'int':
                try:
                    temp_args[arg] = int(cls.args[arg])
                except KeyError:
                    temp_args[arg] = 0
            elif arg_type == 'str':
                try:
                    temp_args[arg] = str(cls.args[arg])
                except KeyError:
                    temp_args[arg] = ""
            else:
                raise ValueError(f"unknown datatype {arg_type}")

        instance = cls(
            prompt=custom_prompt,
            fn_name=matched_fx['fn_name'],
            args=temp_args,
            args_types=matched_fx['args_types'],
            args_names=matched_fx['args_names']
        )

        return instance


class CallMeMaybe_LLM_Model(BaseModel):
    """
    Wrapper class of Small_LLM_Model, make use of the existing LLM class
    and implement own decode/encode method
    """

    _llm: Small_LLM_Model | None = PrivateAttr(None)
    _vocab: str | None = PrivateAttr(None)
    _max_tokens_limit: int = 256
    _token_to_id: dict[str, int] = PrivateAttr({})
    _id_to_token: dict[int, str] = PrivateAttr({})
    _function_definitions: str | None
    _function_def_json: list[Any] | None
    _system_prompt: dict[str, Any] | None = None
    _system_prompt_tokens: dict[str, list[int]] = PrivateAttr(
        default_factory=dict
        )
    # _system_prompt_tensor: Optional[torch.Tensor] = None
    EOT_TOKEN_ID: int = 151643
    DEBUG_OUTPUT: bool = False

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

    def encode(self, text: str) -> list[int]:
        """
        Encode the input text using internal vocab into list[int].

        Equivalent of llm.tokenizer.encode().

        Args:
            text (str): string to encode.

        Returns:
            list[int]: list of the ids mapping

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
        return word_tokens

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
            temp = []
        except json.JSONDecodeError:
            raise ValueError("invalid JSON str")
        if not isinstance(fds, list):
            raise ValueError('array of objects expected')
        for fd in fds:
            if "fn_name" not in fd:
                raise KeyError('object does not have "fn_name" attribute')
            if "args_names" not in fd:
                raise KeyError('object does not have "args_names" attribute')
            if "args_types" not in fd:
                raise KeyError('object does not have "args_types" attribute')
            obj = {
                'fn_name': fd['fn_name'],
                'args_names': fd['args_names']
            }
            temp.append(obj)

        self._function_def_json = fds
        self._function_definitions = json.dumps(temp)

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

        self._system_prompt_tokens['PRE_PROMPT'] = (
            self.encode(self._system_prompt['PRE_PROMPT'])
        )
        self._system_prompt_tokens['POST_PROMPT'] = (
            self.encode(self._system_prompt['POST_PROMPT'])
        )

    def get_system_prompt_ids(self, custom_prompt: str) -> list[int]:
        """
        Get the Tensor from 3 sections, the Pre / prompt / Post

        Args:
            custom_prompt (str): prompt for looking up functions
        Returns:
            torch.Tensor: tensor cretaed from system + custom prompt
        Raises:
            AttribueError: when function was called without setting functions
            definition
        """
        # print(f"SYSTEM PROMPT: {self._system_prompt['PRE_PROMPT']}")
        # print(f"{custom_prompt} {self._system_prompt['POST_PROMPT']}\n\n")
        # if previously set, simply return the whole base prompt
        try:
            if self._system_prompt_tokens['PRE_PROMPT'] is None:
                pass
        except KeyError:
            raise AttributeError("functions definition was not set")
        user_prompt = self.encode(custom_prompt)

        return (
                    self._system_prompt_tokens['PRE_PROMPT'] +
                    user_prompt +
                    self._system_prompt_tokens['POST_PROMPT']
                )

    def _cleanup_json(
            self,
            src: str,
            prompt: str,
            reset_as_well: bool = True
            ) -> dict[str, Any]:
        """
        Adjust the obejct so the output JSON is in "prompt", "fn_name", "args"
        format

        Args:
            src (list[str]): List of the (str) token
            prompt (str): prompt to be added to the returned object
            reset_as_well (bool): cleaned up the src list once done

        Returns:
            dict[str, Any]: python "JSON" dict
        """
        # print(f"RAW: {src}\n\n")
        # print(f"FD_JSON: {src}\n\n")
        try:
            # obj = json.loads(src)
            obj = json.loads(src)
            fn_name = obj['fn_name']

            return_obj = {}
            return_obj['prompt'] = prompt
            for valid_prop in ["fn_name", "args"]:
                try:
                    if obj[valid_prop]:
                        return_obj[valid_prop] = obj[valid_prop]
                except KeyError:
                    continue
            # find the exact used function from the list base JSON
            if self._function_def_json is None:
                raise ValueError(f"key fn_name not found {fn_name}")
            lc = (
                fn for fn in self._function_def_json
                if fn['fn_name'] == fn_name
            )
            if lc is None:
                raise ValueError(f"key fn_name not found {fn_name}")
            fnx = next(lc)

            # type conversion
            args_data = return_obj.get('args')

            if isinstance(args_data, dict):
                for k, v in fnx['args_types'].items():
                    try:
                        if v == "float":
                            return_obj['args'][k] = (
                                float(return_obj['args'][k])
                            )
                        elif v == "int":
                            return_obj['args'][k] = int(return_obj['args'][k])
                        elif v == "str":
                            return_obj['args'][k] = str(return_obj['args'][k])
                        elif v == "bool":
                            return_obj['args'][k] = bool(return_obj['args'][k])
                    except (ValueError, TypeError):
                        continue

            if reset_as_well:
                src = ""
        except Exception as e:
            print(f"ERROR JSON: src = {src}", file=sys.stderr)
            print(f"ERROR return_obj: src = {return_obj}", file=sys.stderr)
            raise e
        return return_obj

    def prompt_selection(self, custom_prompt: str) -> dict[str, Any]:
        """
        get the JSON object in the format
        { "fn_name" : "XXX" , args: { "a": "aaa" , "b": "bbb" }}
        for the selected prompt

        Args:
            custom_prompt (str): prompt for the selection
        Returns:
            dict[str, Any]: the JSON "object"
        """
        if custom_prompt is None or custom_prompt.strip() == "":
            raise ValueError("prompt cannot be empty")

        if self._llm is None:
            raise ValueError("LLM is not initialised")
        ids = self.get_system_prompt_ids(custom_prompt)

        prefill_text = '{"fn_name": "'
        ids.extend(self.encode(prefill_text))

        json_result = prefill_text

        for _ in range(self._max_tokens_limit):

            out_tokens = self._llm.get_logits_from_input_ids(ids)
            next_token_id = self.argmax(out_tokens)

            ids.append(next_token_id)
            new_text = self.decode([next_token_id])
            if self.DEBUG_OUTPUT:
                print(f"{next_token_id:<10} {new_text:<15}",
                      end='\t',
                      file=sys.stderr)
            # json_result += new_text
            # if self.DEBUG_OUTPUT:
            #     print(''.join(json_result), file=sys.stderr)
            if self.DEBUG_OUTPUT:
                print(''.join(json_result + new_text), file=sys.stderr)
            # 3. Quick exit check
            if "}" in new_text:
                for c in new_text:
                    json_result += c
                    try:
                        llm_response = json.loads(json_result)
                        # original line
                        # return self._cleanup_json(json_result, custom_prompt)

                        tool = ToolSelection.create_safe_instance(
                            custom_prompt=custom_prompt,
                            llm_response=llm_response,
                            repo=self._function_def_json
                        )
                        return tool.export("json")
                    except json.JSONDecodeError:
                        pass
            else:
                json_result += new_text
            if next_token_id == self.EOT_TOKEN_ID:
                break

        tool = ToolSelection.create_safe_instance(
                            custom_prompt=custom_prompt,
                            llm_response=llm_response,
                            repo=self._function_def_json
                        )
        return tool.export("json")
