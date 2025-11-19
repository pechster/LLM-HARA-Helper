from typing import List, Dict
import asyncio
import aisuite as ai
from dotenv import load_dotenv
import json
_ = load_dotenv()
client = ai.Client()



# Helper to create chat prompt message dictionaries
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

# Step 1: Identify applicable safety standard and risk parameters
def identify_standard_prompt(system_description: str, model: str) -> str:
    messages = [
        {"role": "system", "content":
            "You are an expert functional safety engineer. Identify the relevant "
            "functional safety standard (e.g., IEC 61508, ISO 26262, etc.) for the given system and list the associated risk assessment "
            "parameters that will be used (e.g., Severity, Exposure, Controllability, Probability)."},
        {"role": "user", "content":
            f"System description:\n{system_description}\n\n"
            "Please provide the name of the applicable safety standard and the list of risk parameters "
            "with brief definitions."}
    ]
    return run_chat(messages, model=model, expected_format="text")


# Step 2: For each hazard/failure, prompt LLM to assign risk parameters with reasoning
def risk_parameters_prompt(hazard_list: List[Dict], standard: str, model: str) -> List[Dict]:
    results = []
    for idx, hazard in enumerate(hazard_list, start=1):
        hazard_desc = hazard.get("description", "No hazard description provided.")
        messages = [
            {"role": "system", "content":
                f"You are an expert safety engineer familiar with the {standard} standard. "
                "For the given hazard scenario, assign all necessary risk parameters (e.g. Severity, Exposure, Controllability, Probability). "
                "Provide detailed reasoning for each parameter assignment."},
            {"role": "user", "content":
                f"Hazard scenario #{idx}:\n{hazard_desc}\n\n"
                "List each risk parameter with assigned value and reasoning.\n"
                "Return the results as a JSON object with keys as parameter names and values as assigned values and rationale."}
        ]
        response = run_chat(messages, model=model, expected_format="json")
        results.append(response)
    return results

# Step 3: Ask LLM to perform risk graph mapping and SIL assignment internally
def risk_assessment_prompt(risk_parameters_outputs: List[Dict], standard: str, model: str) -> List[Dict[str, str]]:
    messages = [
        {"role": "system", "content":
            f"You are a highly experienced safety engineer using the {standard} standard. "
            "Using the provided risk parameters for each hazard, perform risk graph mapping, "
            "calculate the risk levels, and assign Safety Integrity Level (SIL) or equivalent. "
            "Provide the complete hazard risk assessment table or JSON."},
        {"role": "user", "content":
            "Here are the risk parameters for hazards:\n"
            f"{json.dumps(risk_parameters_outputs, indent=2)}\n\n"
            "Please return a structured JSON array with each hazard, its calculated risk level, SIL assignment, "
            "and rationale for each assignment."}
    ]
    return run_chat(messages, model=model, expected_format="json")


def run_risk_assessment(system_description: str, hazard_list: List[Dict], model: str = "openai:gpt-4o-mini") -> Dict:    

    # Step 1: Identify standard and parameters
    identified_standard = identify_standard_prompt(system_description, model)

    # Step 2: Get risk parameters per hazard
    risk_params_outputs = risk_parameters_prompt(hazard_list, identified_standard, model)

    # Step 3: Final risk assessment with SIL assignment
    final_risk_assessment = risk_assessment_prompt(risk_params_outputs, identified_standard, model)

    return {
        "identified_standard": identified_standard,
        "risk_parameters_outputs": risk_params_outputs,
        "final_risk_assessment": final_risk_assessment,
    }