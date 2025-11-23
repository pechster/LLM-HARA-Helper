import re
import aisuite as ai  # which imports are really needed?
from dotenv import load_dotenv
import json
import os
import ast
from typing import List, Dict, Any
from HELPERS import *

_ = load_dotenv()
client = ai.Client()


def clean_json_string(content: str) -> str:
    """Removes markdown formatting to ensure clean JSON parsing."""
    return content.replace("```json", "").replace("```", "").strip()


def run_single_hara(user_prompt: str, model: str) -> Dict[str, Any]:
    """
    Runs the full HARA chain on a SINGLE model and returns the dictionary.
    """
    print(f"\n>>> ==== STARTING RUN ON MODEL: {model} ====")

    # Local storage for this specific run
    run_data = {
        "model_used": model,
        "system": "",
        "persons": [],
        "hazards": [],
        "harms_analysis": [],
        "scenarios": []
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
        {"role": "user",
         "content": f"System: {run_data['system']}\n\nReturn JSON list e.g. ['Shearing', 'Electrical Shock']"}
    ]
    run_data["hazards"] = run_chat(msg, model, expected_format="json")

    # 4. Analyze Harms
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

    print(f"<<< ==== COMPLETED RUN ON {model} ====")
    return run_data


def synthesize_consensus(results_list: List[Dict], judge_model: str = "openai:gpt-4o") -> Dict[str, Any]:
    """
    Takes N result dictionaries, compares them, and keeps only the findings
    that appear in the majority (conceptually) using an LLM Judge.
    """
    print(f"\nCALCULATE HOW TO USE ANSWERS : {judge_model}...")

    data_str = json.dumps(results_list, indent=2)

    system_prompt = f"""
    You are a an expert in HARA analyis. Your job is to review HARA (Hazard Analysis) reports from muliple different 
    junior analysts.

    TASK:
    - You must generate a FINAL CONSENSUS JSON Report
    - All contradictory information must be refined into a consistent, technically accurate statement
    - For semantical similar responses remove any redundancy while perserving all information which is relevant for the 
      HARA anlysis
    - Any information only mentioned by one anaylysis must be refactored into a consistent statement with respect to 
      all other claims
    - You are not allowed to include incorrect statements or extend them with addtionial information so that they fit
      better
    
    OUTPUT REQUIREMENTS:
    - Return only a valid JSON object
    - You are not allowed to leave a key out which is defined in the JSON FORMAT section
    
    JSON FORMAT:
    {{
        "consensus_system_description": "...",
        "verified_persons": ["..."],
        "verified_hazards": ["..."],
        "verified_harms": [ {{"person": "...", "hazard": "...", "harm": "..."}} ],
        "verified_scenarios": [ {{"title": "...", "narrative": "..."}} ]
    }}
    """

    user_prompt = f"""
    Here are the reports from the HARA analysts:

    {data_str}

    Generate the consolidated JSON.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    final_json = run_chat(messages, model=judge_model, expected_format="json")
    return final_json

def main():
    run_single_hara("A mobile robot (AGV) transports heavy pallets in a warehouse shared with human workers. It has a lifting fork mechanism.", "openai:gpt-4o-mini")

if __name__ == main():
    main()
