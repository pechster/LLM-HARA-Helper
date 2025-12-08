import aisuite as ai
import json
import re
from dotenv import load_dotenv
from typing import Any
import ast

_ = load_dotenv()
client = ai.Client()


def run_chat(messages: list, model: str, expected_format="text"):
    try:
        response = client.chat.completions.create(model=model, messages=messages)
        content = response.choices[0].message.content

        if expected_format == "json":
            clean_content = content.replace("json", "").replace("", "").strip()

            try:
                return json.loads(clean_content)
            except json.JSONDecodeError:
                pass

            try:
                return ast.literal_eval(clean_content)
            except (ValueError, SyntaxError):
                pass

            match = re.search(r'([.]|{.})', clean_content, re.DOTALL)
            if match:
                try:
                    candidate = match.group(1)

                    return json.loads(candidate.replace("'", '"'))
                except:
                    pass

            print(f"Warning: Parsing failed completely for {model}. Raw: {clean_content}...")
            return [] if "list" in str(messages) else {}

        return content
    except Exception as e:
        print(f"Error calling model {model}: {e}")
        return {} if expected_format == "json" else ""


def run_chat_hara(messages: list, model: str, expected_format: str = "text", output_file:str = None, **kwargs) -> Any:
    try:
        response = client.chat.completions.create(model=model, messages=messages)
        content = response.choices[0].message.content

        if expected_format == "json":
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)

            if match:
                clean_content = match.group(1).strip()
            else:
                json_start = re.search(r"[\[\{]", content)
                if json_start:
                    clean_content = content[json_start.start():].strip()
                else:
                    clean_content = content.strip()

            try:
                return json.loads(clean_content)
            except json.JSONDecodeError:
                try:
                    fixed_content = clean_content.replace("'", '"')
                    parsed_data = json.loads(fixed_content)
                except json.JSONDecodeError:
                    print(f"Warning: Parsing failed completely for {model}.")
                    print(f"Raw Content: {clean_content[:100]}...")
                    return [] if "list" in str(messages).lower() else {}
        
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    if expected_format == "json":
                        json.dump(parsed_data, f, ensure_ascii=False, indent=4)
                    else:
                        f.write(str(parsed_data))
                print(f"   [Saved output to: {output_file}]")
            except Exception as e:
                print(f"   [Error saving file {output_file}: {e}]")
        return parsed_data
        

    except Exception as e:
        print(f"Error calling model {model}: {e}")
        return [] if expected_format == "json" else ""
