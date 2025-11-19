import aisuite as ai
from dotenv import load_dotenv
import json
import os
from typing import List, Dict, Any

_ = load_dotenv()
client = ai.Client()

# Global dictionary to store the chain of thought/results
hara_data = {
    "system": "",
    "persons": [],
    "hazards": [],
    "harms_analysis": [], 
    "physical_analysis": [], 
    "actuation_analysis": [], 
    "scenarios": ""
}

def run_chat(messages: list, model: str = "openai:gpt-4o-mini", expected_format="text"):
    """
    Helper to run chat and handle JSON parsing if required.
    Now accepts 'model' dynamically.
    """
    # We pass the model parameter here dynamically
    response = client.chat.completions.create(model=model, messages=messages)
    content = response.choices[0].message.content
    
    if expected_format == "json":
        clean_content = content.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(clean_content)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse JSON from {model}. Raw content: {clean_content[:50]}...")
            return []
    return content

# --- Step 1: Extract System ---

def extract_system(user_prompt: str, model: str):
    print(f"--- Step 1: Extracting System ({model}) ---")
    messages = [
        {"role": "system", "content": "You are a HARA expert. Extract the technical system description."},
        {"role": "user", "content": f"User Prompt: {user_prompt}\n\nExtract the system description conciseley."}
    ]
    hara_data["system"] = run_chat(messages, model=model)
    print(f"System: {hara_data['system']}\n")

# --- Step 2: Identify Persons at Risk ---

def identify_persons(model: str):
    print(f"--- Step 2: Identifying Persons at Risk ({model}) ---")
    messages = [
        {"role": "system", "content": "Identify potential persons at risk. Output a JSON list of strings."},
        {"role": "user", "content": f"System: {hara_data['system']}\n\nReturn a JSON list of persons at risk (e.g. ['Operator', 'Bystander'])."}
    ]
    hara_data["persons"] = run_chat(messages, model=model, expected_format="json")
    print(f"Persons: {hara_data['persons']}\n")

# --- Step 3: Hazards & Loop for Harms ---

def analyze_hazards_and_harms(model: str):
    print(f"--- Step 3: Hazards & Harm Loop ({model}) ---")
    
    # Extract Hazards first
    messages = [
        {"role": "system", "content": "Identify potential hazards. Output a JSON list of strings."},
        {"role": "user", "content": f"System: {hara_data['system']}\n\nReturn a JSON list of hazards (e.g. ['Mechanical shearing', 'Electrical shock'])."}
    ]
    hara_data["hazards"] = run_chat(messages, model=model, expected_format="json")
    print(f"Hazards Identified: {hara_data['hazards']}")
    
    # Analyze Harms for each Person + Hazard
    results = []
    if not hara_data["persons"] or not hara_data["hazards"]:
        print("Skipping harm loop due to missing persons or hazards.")
        return

    for person in hara_data["persons"]:
        for hazard in hara_data["hazards"]:
            msg = [
                {"role": "system", "content": "You are a risk analyst. Determine the specific harm."},
                {"role": "user", "content": f"System: {hara_data['system']}\nPerson: {person}\nHazard: {hazard}\n\nWhat specific harm (injury type/severity) could be caused to this person by this hazard? If none, return 'None'. Keep it concise."}
            ]
            harm = run_chat(msg, model=model)
            
            if "None" not in harm and "none" not in harm:
                entry = {"person": person, "hazard": hazard, "harm": harm}
                results.append(entry)
                print(f"  -> {person} + {hazard} = {harm}")
    
    hara_data["harms_analysis"] = results
    print("")

# --- Step 4: Physical Attributes & Impact Class ---

def analyze_physical_impact(model: str):
    print(f"--- Step 4: Physical Values & Impact Classes ({model}) ---")
    
    # Extract System Parameters first
    msg_params = [
        {"role": "system", "content": "Extract system physical attributes and impact classes. Output JSON."},
        {"role": "user", "content": f"System: {hara_data['system']}\n\nIdentify:\n1. Physical Attributes (e.g., Speed, Force, Temperature)\n2. Impact Classes (e.g., Clamping, Collision, Radiation)\n\nOutput JSON format: {{'attributes': [], 'impact_classes': []}}"}
    ]
    params = run_chat(msg_params, model=model, expected_format="json")
    attributes = params.get("attributes", [])
    impacts = params.get("impact_classes", [])
    
    print(f"Attributes: {attributes}")
    print(f"Impact Classes: {impacts}")
    
    # Loop through each harm
    unique_harms = list(set([x["harm"] for x in hara_data["harms_analysis"]]))
    
    results = []
    for harm in unique_harms:
        msg = [
            {"role": "system", "content": "Link the harm to physical system values. Output JSON."},
            {"role": "user", "content": f"""
            System: {hara_data['system']}
            Harm: {harm}
            Available Attributes: {attributes}
            Available Impact Classes: {impacts}
            
            Question: By influencing which Physical Value (from list) through which Impact Class (from list) is this harm caused?
            
            Return JSON: {{'harm': '{harm}', 'physical_value': '...', 'impact_class': '...'}}
            """}
        ]
        res = run_chat(msg, model=model, expected_format="json")
        if res:
            results.append(res)
            print(f"  -> Harm: {harm[:20]}... caused by {res.get('physical_value')} via {res.get('impact_class')}")

    hara_data["physical_analysis"] = results
    print("")

# --- Step 5: Actuation Failures ---

def analyze_actuation(model: str):
    print(f"--- Step 5: Actuation Failures ({model}) ---")
    
    # Loop through the results of Step 4
    results = []
    for item in hara_data["physical_analysis"]:
        phy_val = item.get("physical_value")
        imp_cls = item.get("impact_class")
        
        msg = [
            {"role": "system", "content": "Identify actuator failures. Output JSON."},
            {"role": "user", "content": f"""
            System: {hara_data['system']}
            Problem: A hazardous event involving Physical Value '{phy_val}' and Impact Class '{imp_cls}'.
            
            Question: To which failure would a failure mode of a specific actuator lead to this impact?
            
            Return JSON: {{'actuator': '...', 'failure_mode': '...', 'mechanism': '...'}}
            """}
        ]
        res = run_chat(msg, model=model, expected_format="json")
        if res:
            res["linked_harm_context"] = item 
            results.append(res)
            print(f"  -> Actuator: {res.get('actuator')} failed via {res.get('failure_mode')}")
            
    hara_data["actuation_analysis"] = results
    print("")

# --- Step 6: Scenario Refinement ---

def generate_scenarios(model: str):
    print(f"--- Step 6: Scenario Generation ({model}) ---")
    
    msg = [
        {"role": "system", "content": "You are a HARA expert. Generate detailed risk scenarios."},
        {"role": "user", "content": f"""
        Based on the following analysis data, play through and generate 3 refined, narrative failure scenarios.
        
        System: {hara_data['system']}
        Persons: {hara_data['persons']}
        Hazards Identified: {hara_data['hazards']}
        Actuation Failures Identified: {json.dumps(hara_data['actuation_analysis'])}
        
        Format:
        1. Scenario Title
        2. Narrative (Description of sequence)
        3. Consequence
        """}
    ]
    hara_data["scenarios"] = run_chat(msg, model=model)
    print(hara_data["scenarios"])

# --- Main Execution ---

def run_hara_tool(prompt: str, model: str):
    """
    Runs the full HARA chain using the specified model.
    """
    # Reset before new run
    global hara_data
    hara_data = {
        "system": "",
        "persons": [],
        "hazards": [],
        "harms_analysis": [],
        "physical_analysis": [],
        "actuation_analysis": [],
        "scenarios": ""
    }
    
  
    print(f"======== STARTING ANALYSIS WITH MODEL: {model} ========")
   

    extract_system(prompt, model)
    identify_persons(model)
    analyze_hazards_and_harms(model)
    analyze_physical_impact(model)
    analyze_actuation(model)
    generate_scenarios(model)
    
    return hara_data["scenarios"]

# --- Test ---
if __name__ == "__main__":
    user_input = "A mobile robot (AGV) transports heavy pallets in a warehouse shared with human workers. It has a lifting fork mechanism."
    scenarios_openai = run_hara_tool(user_input, model="openai:gpt-4o-mini")
    # possible to do multiple runs and compare the results
    # scenarios_anthropic = run_hara_tool(user_input, model="anthropic:claude-3-5-sonnet")