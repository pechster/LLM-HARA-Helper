import os
import aisuite as ai

try:
    from dotenv import load_dotenv
    _ = load_dotenv()
    print("Loaded .env (if present).")
except Exception:
    print("python-dotenv not installed or .env not found. That's okay.")

client = ai.Client()

def run_chat(model: str, messages: list, **kwargs):
    resp = client.chat.completions.create(model=model, messages=messages, **kwargs)
    return resp.choices[0].message.content

def improve_prompt(prompt: str) -> str:
    messages = [
        {"role": "system", "content": "Rewrite the following prompt for clarity, conciseness, and high-quality HARA analysis."},
        {"role": "user", "content": prompt}
    ]
    return run_chat("openai:gpt-4o-mini", messages)


def get_model_responses(prompt: str, model_list: list) -> dict:
    replies = {}
    messages = [
        {"role": "system", "content": "You are a precise step by step HARA analyser" },
        {"role": "user", "content": prompt}
    ]
    for model in model_list:
        try:
            replies[model] = run_chat(model, messages)
        except Exception as e:
            replies[model] = f"(error: {e})"
    return replies

def critique_and_refine(answer: str, iterations: int = 3, model: str = "openai:gpt-4o-mini") -> str:
    current = answer
    for _ in range(iterations):
        critique = run_chat(model, [
            {"role": "system", "content": "You are a strict critique for the report of HARA analysis. Point out possible improvements and errors."},
            {"role": "user", "content": current}
        ])
        improved = run_chat(model, [
            {"role": "system", "content": "Make the following HARA analysis better, after addressing this critique:"},
            {"role": "user", "content": f"Analysis:\n{current}\n\nCritique:\n{critique}"}
        ])
        current = improved
    return current



original_prompt = "Create a hara analysis for the following: A robot arm that works with human. Give the analysis in steps."


refined_prompt = improve_prompt(original_prompt)
print("Refined Prompt:", refined_prompt)

models = ["openai:gpt-4o-mini", "anthropic:claude-3-5-sonnet"]
responses = get_model_responses(refined_prompt, models)
for model, reply in responses.items():
    print(f"\nModel: {model}\n{reply}")


comparison = run_chat("openai:gpt-4o-mini", [
    {"role": "system", "content": "You are an expert HARA analyst. Compare and critique the following model analyses. Rank them with pros/cons and provide a final improved, unified answer."},
    {"role": "user", "content": "\n\n".join([f"{m}:\n{a}" for m, a in responses.items()])}
])
print("\nModel Comparison and improved Answer:\n", comparison)


best_model_answer = responses["openai:gpt-4o-mini"]  
refined = critique_and_refine(best_model_answer)
print("\nRefined Answer:\n", refined)