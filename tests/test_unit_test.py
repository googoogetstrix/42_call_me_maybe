import pytest
import json
from typing import Any
from cmm_llm import CallMeMaybe_LLM_Model, ToolSelection
# def test_simple(tmp_path):

#     out_dir = tmp_path / "data" / "out"
#     out_dir.mkdir(parents=True)

#     os.chmod(out_dir, 0o555)

#     try:
#         # 3. Execution & Verification: Expect an OSError or your custom error
#         with pytest.raises(PermissionError):
#             save_data(target_path=out_dir)
#     finally:
#         # Cleanup: Must restore permissions to allow pytest to delete the
#           temp folder
#         os.chmod(out_dir, 0o777)
#     assert 1 == 1
# def test_creates_directory_if_missing(tmp_path):
#     target = tmp_path / "new_folder"
#     # Call your function on a folder that doesn't exist yet
#     # Check if the folder was created automatically
#     assert target.exists()
#     assert target.is_dir()


def test_code_test() -> None:
    """
    Test for tool selection,
    """
    # test for usage of ToolSelection
    def_json_file = "./data/input/functions_definition.json"
    with open(def_json_file, "r") as f:
        f_repo = json.load(f)

    custom_prompt = "What is the sum of 42.0 and 54.3"
    response = json.loads("""{
        "fn_name": "fn_reverse_string",
        "args": {
            "s": "world"
        }
    }""")
    response['prompt'] = custom_prompt
    out = ToolSelection.create_safe_instance(
        custom_prompt=custom_prompt,
        llm_response=response,
        repo=f_repo
    )
    assert out.export("json") == response

    custom_prompt = "Who is Leonardo DaVinci?"
    out = ToolSelection.create_safe_instance(
        custom_prompt=custom_prompt,
        llm_response=response,
        repo=f_repo
    )

    output = out.export("json")
    assert "prompt" in output
    assert output["prompt"] == custom_prompt
    assert "fn_name" in output
    assert "args" in output


def test_unittest(tmp_path: Any) -> None:
    """
    test for internal code

    Returns:
        None
    """

    model = CallMeMaybe_LLM_Model()
    # 1.) model is not supposed to predict without setting proper functions
    # definition
    with pytest.raises(AttributeError):
        model.prompt_selection("Something")

    # 2.) try setting up various kind of JSON functions definition
    with pytest.raises(ValueError) as e:
        model.set_functions_definition("")
        print(e)
        assert False

    with pytest.raises(ValueError):
        model.set_functions_definition("{}")

    with pytest.raises(KeyError):
        model.set_functions_definition("[ { \"fn\" : 10 } ]")

    with pytest.raises(KeyError):
        model.set_functions_definition(
            "[ { \"fn_name\" : \"ft_add_numbers\" } ]"
            )

    with pytest.raises(KeyError):
        model.set_functions_definition(
            "[ { \"args_names\" :  { \"a\" : 10 } } ]"
            )

    with pytest.raises(KeyError):
        model.set_functions_definition(
            "[ { \"args_types\" :  { \"a\" : 10 } } ]"
            )

    # seems OK, try set with the actual input data
    with open("./data/input/functions_definition.json", "r") as f:
        f_def = f.read()
    model.set_functions_definition(f_def)

    # start working on test prompts
    prompt = "What is the value of 3.14 multiply with 10?"
    predicted = model.prompt_selection(prompt)
    expected = {
        "prompt": prompt,
        "fn_name": "fn_multiply_numbers",
        "args": {
            "a": 3.14,
            "b": 10
        }
    }
    assert predicted == expected
    # what happen to the absolute BS

    # 3.) custom_prompt is not supposed to be empty or None
    with pytest.raises(ValueError):
        model.prompt_selection("")

    with pytest.raises(ValueError):
        model.prompt_selection(None)
