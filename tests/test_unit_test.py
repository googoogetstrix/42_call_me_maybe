import pytest
import json
from typing import Any
from cmm_llm import CallMeMaybe_LLM_Model
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
        model.set_functions_definition("[ { \"fn_name\" : \"ft_add_numbers\" } ]")

    with pytest.raises(KeyError):
        model.set_functions_definition("[ { \"args_names\" :  { \"a\" : 10 } } ]")

    with pytest.raises(KeyError):
        model.set_functions_definition("[ { \"args_types\" :  { \"a\" : 10 } } ]")

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

    prompt = "How old is the Mona Lisa?"
    predicted = model.prompt_selection(prompt)
    assert False
    




    # 3.) custom_prompt is not supposed to be empty or None
    with pytest.raises(ValueError):
        model.prompt_selection("")

    with pytest.raises(ValueError):
        model.prompt_selection(None)

# def test_irectories_ok():
