import aisuite as ai
import ast
import json
from dotenv import load_dotenv
import re

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