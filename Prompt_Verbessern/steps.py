import aisuite as ai
from dotenv import load_dotenv
import json
import os
from typing import List, Dict, Any

_ = load_dotenv()
client = ai.Client()

# --- Helper Functions ---

def clean_json_string(content: str) -> str:
    """Removes markdown formatting to ensure clean JSON parsing."""
    return content.replace("```json", "").replace("```", "").strip()

def run_chat(messages: list, model: str, expected_format="text"):
    try:
        response = client.chat.completions.create(model=model, messages=messages)
        content = response.choices[0].message.content
        
        if expected_format == "json":
            # 1. Strip Markdown (```json ... ```)
            clean_content = content.replace("```json", "").replace("```", "").strip()
            
            # 2. Try standard JSON parse
            try:
                return json.loads(clean_content)
            except json.JSONDecodeError:
                pass # Fall through to backup methods

            # 3. Backup: Try parsing as a Python Literal (Handles single quotes: ['A', 'B'])
            try:
                # Only safe if content is a list/dict structure
                return ast.literal_eval(clean_content)
            except (ValueError, SyntaxError):
                pass

            # 4. Last Resort: Regex to find the first list [...] or object {...}
            # Sometimes models add text like "Here is the list: [...]"
            match = re.search(r'(\[.*\]|\{.*\})', clean_content, re.DOTALL)
            if match:
                try:
                    candidate = match.group(1)
                    # Try JSON again on the extracted part
                    return json.loads(candidate.replace("'", '"')) # Naive single-to-double quote swap
                except:
                    pass
            
            print(f"Warning: Parsing failed completely for {model}. Raw: {clean_content[:50]}...")
            return [] if "list" in str(messages) else {}

        return content
    except Exception as e:
        print(f"Error calling model {model}: {e}")
        return {} if expected_format == "json" else ""

# --- The Core HARA Logic (Encapsulated) ---

def run_single_hara_pass(user_prompt: str, model: str) -> Dict[str, Any]:
    """
    Runs the full HARA chain on a SINGLE model and returns the dictionary.
    """
    print(f"\n>>> 🤖 STARTING RUN ON MODEL: {model}")
    
    # Local storage for this specific run
    run_data = {
        "model_used": model,
        "system": "",
        "persons": [],
        "hazards": [],
        "harms_analysis": [], 
        "scenarios": [] # Changed to list for JSON consistency
    }

    # 1. Extract System
    msg = [
        {"role": "system", "content": "Extract the technical system description."},
        {"role": "user", "content": f"User Prompt: {user_prompt}\n\nExtract the system description concisely."}
    ]
    run_data["system"] = run_chat(msg, model)

    # 2. Identify Persons at Risk
    msg = [
        {"role": "system", "content": "Identify persons at risk. Output a JSON list of strings."},
        {"role": "user", "content": f"System: {run_data['system']}\n\nReturn JSON list e.g. ['Operator', 'Bystander']"}
    ]
    run_data["persons"] = run_chat(msg, model, expected_format="json")

    # 3. Identify Hazards
    msg = [
        {"role": "system", "content": "Identify hazards. Output a JSON list of strings."},
        {"role": "user", "content": f"System: {run_data['system']}\n\nReturn JSON list e.g. ['Shearing', 'Electrical Shock']"}
    ]
    run_data["hazards"] = run_chat(msg, model, expected_format="json")

    # 4. Analyze Harms (Simplified for token efficiency in this demo)
    # We ask for a batch JSON analysis instead of a loop to save time/cost
    if run_data["persons"] and run_data["hazards"]:
        msg = [
            {"role": "system", "content": "Analyze specific harms. Output a JSON list of objects."},
            {"role": "user", "content": f"""
            System: {run_data['system']}
            Persons: {run_data['persons']}
            Hazards: {run_data['hazards']}
            
            Create a list of risk pairs. 
            Format: [ {{"person": "...", "hazard": "...", "harm": "...", "severity": "High/Med/Low"}} ]
            """}
        ]
        run_data["harms_analysis"] = run_chat(msg, model, expected_format="json")

    # 5. Scenarios (Strict JSON)
    msg = [
        {"role": "system", "content": "Generate 3 refined failure scenarios in JSON."},
        {"role": "user", "content": f"""
        Based on:
        System: {run_data['system']}
        Harms: {json.dumps(run_data['harms_analysis'])}
        
        Return a JSON List of objects:
        [ {{"title": "...", "narrative": "...", "consequence": "..."}} ]
        """}
    ]
    run_data["scenarios"] = run_chat(msg, model, expected_format="json")
    
    print(f"<<< ✅ COMPLETED RUN ON {model}")
    return run_data

# --- The Consensus Engine ---

def synthesize_consensus(results_list: List[Dict], judge_model: str = "openai:gpt-4o") -> Dict[str, Any]:
    """
    Takes N result dictionaries, compares them, and keeps only the findings 
    that appear in the majority (conceptually) using an LLM Judge.
    """
    print(f"\n⚖️  CALCULATING CONSENSUS USING JUDGE: {judge_model}...")
    
    # We serialize the results to pass to the judge
    data_str = json.dumps(results_list, indent=2)
    
    system_prompt = """
    You are a Senior Safety Auditor. Your job is to review HARA (Hazard Analysis) reports from 3 different junior analysts (models).
    
    You must generate a FINAL CONSENSUS JSON Report.
    
    Rules for Consensus:
    1. MERGE duplicates: If Analyst A says "Cut" and Analyst B says "Laceration", merge them into one entry "Laceration".
    2. FILTER outliers: If only one Analyst mentions a bizarre hazard that makes no sense, exclude it. 
    3. KEEP majority: If a hazard/scenario appears in at least 2 of the 3 reports (semantically), keep it.
    
    Return ONLY the valid JSON object matching this structure:
    {
        "consensus_system_description": "...",
        "verified_persons": ["..."],
        "verified_hazards": ["..."],
        "verified_harms": [ {"person": "...", "hazard": "...", "harm": "..."} ],
        "verified_scenarios": [ {"title": "...", "narrative": "..."} ]
    }
    """
    
    user_prompt = f"""
    Here are the 3 reports from the analysts:
    
    {data_str}
    
    Generate the consolidated JSON now.
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    final_json = run_chat(messages, model=judge_model, expected_format="json")
    return final_json

# --- Main Execution ---

def main():
    # The user input
    user_input = "A mobile robot (AGV) transports heavy pallets in a warehouse shared with human workers. It has a lifting fork mechanism."
    
    # Define your 3 models (Ensure you have keys for these providers)
    # Example: Using OpenAI for all 3 just for demo, but you should swap these!
    # models_to_test = ["openai:gpt-4o", "anthropic:claude-3-5-sonnet", "google:gemini-1.5-pro"]
    
    # For this specific run, I will use variations or just repeat to demonstrate the logic
    # In your real environment, change these strings to your actual providers.
    models_to_test = [
        "openai:gpt-4o-mini", 
        "anthropic:claude-sonnet-4-20250514"
    ]

    results_buffer = []

    # 1. Run Models in Loop
    for model in models_to_test:
        result = run_single_hara_pass(user_input, model)
        results_buffer.append(result)

    # 2. Run Consensus
    final_data = synthesize_consensus(results_buffer, judge_model="openai:gpt-4o")

    # 3. Output Final JSON
    print("\n======== 🏆 FINAL CONSENSUS DATA (JSON) ========")
    print(json.dumps(final_data, indent=2))
    
    # Optional: Save to file
    with open("hara_consensus_result.json", "w") as f:
        json.dump(final_data, f, indent=2)
        print("\nSaved to hara_consensus_result.json")

if __name__ == "__main__":
    main()