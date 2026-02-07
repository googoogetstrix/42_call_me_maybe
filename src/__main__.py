from .cmm_llm import CallMeMaybe_LLM_Model
import json
import argparse
from pathlib import Path


def main() -> None:
    """
    Run the Call Me Maybe project with the default options
    """

    try:
        parser = argparse.ArgumentParser(
            description="Parser for call me maybe"
        )
        parser.add_argument("--input",
                            type=str,
                            default="data/input/function_calling_tests.json",
                            help="Input file")
        parser.add_argument(
            "--output",
            type=str,
            default="data/output/function_calling_results.json",
            help="Output file"
            )
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


if __name__ == "__main__":
    main()
