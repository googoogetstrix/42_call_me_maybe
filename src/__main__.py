from .cmm_llm import CallMeMaybe_LLM_Model
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime
import time

def main() -> None:
    """
    Run the Call Me Maybe project with the default options
    """

    now = datetime.now()
    start = time.perf_counter()

    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} INFO: Starts Execution",file=sys.stderr)

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

        elapsed = time.perf_counter() - start
        print(f"{elapsed:.3f}s after init")

        with open(definition_path, "r") as f:
            f_def = f.read()
        with open(input_path, "r") as f:
            prompts = json.load(f)

        model.set_functions_definition(f_def)
        items = []
        with open(output_path, "w") as f:
            for k in prompts:

                start = time.perf_counter()
                out = model.prompt_selection(k['prompt'])
                elapsed = time.perf_counter() - start
                
                items.append(out)
                print(out, file=sys.stderr)
                print(f"Prompt {k}: {elapsed:.3f}s", file=sys.stderr)
            json.dump(items, f, indent=4, ensure_ascii=False)
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp} INFO: Ends Execution", file=sys.stderr)

    except Exception as e:
        print("Error:", e)
        raise e


if __name__ == "__main__":
    main()
