from typing import List, Dict
import asyncio
import aisuite as ai
from dotenv import load_dotenv
import json
from HELPERS import *


# Step 1: Identify applicable safety standard and risk parameters
def identify_standard_prompt(system_description: str, model: str) -> str:
    messages = [
        {"role": "system", "content":
            f"""You are an expert functional safety engineer. Identify the relevant
            
            TASK: 
             - Identify the most relevant functional safety standard for the given system (e.g. IEC 61508, ISO 26262)
             - Define the risk parameters in this standard (e.g Severity, Exposure, Controllabilty and Probabilty)
             and what the mean in the context of the system
             
            OUTPUT REQUIREMENTS:
             - Respond only with a valid JSON object 
             
            JSON FORMAT:
            {{
                "standard_name": "Formal name of the standard",
                "standard_reference": "Reference, e.g. ICE 61508",
                "risk_parameters": {{
                "name": "risk parameter name"
                "description": "what is the risk parameter referring to"
                }}            
            }}
        """},
        {
            "role": "user", "content":
            f"System description: {system_description}\n"
            "Return the valid JSON in the format described above. Use good logical reasoning for the selection of the"
            "standard which should not be provided in the JSON."
        }
    ]
    return run_chat(messages, model=model, expected_format="json")


# DO WE WANT TO USE THE RISK PARAMETERS OF THE STANDARD OR THE 4?
# Step 2: For each hazard/failure, prompt LLM to assign risk parameters with reasoning
def risk_parameters_prompt(hazard_list: List[Dict], standard: str, model: str) -> List[Dict]:
    results = []
    for idx, hazard in enumerate(hazard_list, start=1):  # maybe sort by severity
        messages = [
            {"role": "system", "content":
                f"""You are an expert functional safety engineer familiar with the {standard} standard.
                TASK:
                - assign risk parameters for one hazard 
                
                RISK PARAMETERS:
                - Severity
                - Exposure      
                - Controllability
                - Probability
                
                OUTPUT REQUIREMENTS:
                - The response has to be one single JSON
                - Make use of the predifined formta
                - If the standard does not provide the risk parameter then map the parameter to the four defined above
                                
                JSON FORMAT:
                {{
                 "hazard": "{hazard}"
                 "Severity: " {{
                    "value" : "Low/Medium/High/Very High"
                    "reason" : "short explanation why this value is assigned"
                }}, 
                 "Exposure: " {{
                    "value" : "Low/Medium/High"
                    "reason" : "short explanation what explains the frequency of the occurence"
                }},
                "Controllability: " {{
                    "value" : "Easy/Moderate/Difficult/Uncontrollable
                    "reason" : "short explanation what could possibly avoid the occurence"
                }}
                "Probability: " {{
                    "value" : "Rare/Occasional/Frequent"
                    "reason" : "short explanation on how exposure and controllability determine the explain 
                    the probability"
                }}
                }}"""
            },
            {
                "role": "user", "content":
                f"Hazard scenario {hazard}: "
                "Return ONLY the JSON object in the previously specified scheme."
            }
        ]
        response = run_chat(messages, model=model, expected_format="json")
        results.append(response)
    return results


# Step 3: Ask LLM to perform risk graph mapping and SIL assignment internally
def risk_assessment_prompt(risk_parameters_outputs: List[Dict], standard: str, model: str) -> List[Dict[str, str]]:
    messages = [
        {"role": "system", "content":
            f"""You are an expert safety engineer using the {standard} standard. "
            TASK:
            - Perform risk graph mapping for the provided risk parameters for each hazard
            - Determine a risk level and Saftey Integrity Level (SIL)
            
            OUTPUT REQUIREMENTS:
            - The only with a valid JSON array
            - The length of the output has to match the number of input hazards
            
            JSON FORMAT:
            {{
                "hazard: " : "hazard description",
                "risk parameters: " : "summary of the risk parameters for this hazard",
                "risk level: " : "Low/Medium/High/Very High"
                "sil: " : "SIL1/SIL2/SIL3 or a matching from a other Saftey Integrity Level to corresponding SIL 
                definition",
                "rationale": "short explanation for risk level and SIL"  
            }}
        """},
        {"role": "user", "content":
            "Here are the risk parameters for hazards:\n"
            f"{json.dumps(risk_parameters_outputs, indent=2)}\n"
            "Return only the JSON object in the previously specified scheme."}
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


def synthesize_consensus(results_list: List[Dict], judge_model: str = "openai:gpt-4o") -> Dict[str, str]:
    """
    Takes N result dictionaries, compares them, and keeps only the findings
    that appear in the majority (conceptually) using an LLM Judge.
    """
    print(f"\nCALCULATE HOW TO USE ANSWERS : {judge_model}...")

    data_str = json.dumps(results_list, indent=2)

    system_prompt = f"""
    You are a an expert safety engineer. Your job is to review final risk assessment of mulitple junior safety engineers
    with different reports. 

    TASK:
    - You must generate a FINAL CONSENSUS JSON Report
    - All contradictory information must be refined into a consistent, technically accurate statement
    - For semantical similar responses remove any redundancy while perserving all information which is relevant for the 
      risk assessment
    - Any information only mentioned by one anaylysis must be refactored into a consistent statement with respect to 
      all other claims
    - You are not allowed to include incorrect statements or extend them with addtionial information so that they fit
      better

    OUTPUT REQUIREMENTS:
    - Return only a valid JSON object
    - You are not allowed to leave a key out which is defined in the JSON FORMAT section
    - You are not allowed to change the general format of the junior safety engineers reports
    
    JSON FORMAT:
    {{
        "identified_standard: " : "...",
        "risk_parameters_outputs: " : "...",
        "final_risk_assessment: " : "...",
    }}
    """

    user_prompt = f"""
    Here are the reports from the junior safety engineers:

    {data_str}

    Generate the consolidated JSON.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    final_json = run_chat(messages, model=judge_model, expected_format="json")
    return final_json
