import re
import aisuite as ai
from dotenv import load_dotenv
import json
import os
import ast
from typing import List, Dict, Any
from HELPERS import *
from rich.console import Console
#added semantic embeddings - pip install sentence-transformers scikit-learn
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from collections import defaultdict

console = Console()

_ = load_dotenv()
client = ai.Client()


def extract_system(user_input:str, model:str="google:gemini-1.5-pro"):
    system_prompt = {
        "role": "system",
        "content": f"""
        You are an expert in Hazard Analysis and Risk Assessment (HARA). 

        TASK: 
        Analyze the user's input text and extract the system description.

        OUTPUT FORMAT:
        Return only a valid JSON object with exactly these keys:
        - "name": (string, extremely concise, e.g., "Cargo Drone", "Humanoid Robot", "Automated Guided Vehicle")
        - "description": (string, a concise summary of capabilites and limits)
        """}
    
    few_shot_user = {
        "role": "user",
        "content": "A cargo drone that carry cargos up to 5kg with a speed of 3m/s."
    }

    few_shot_assistant = {
        "role": "assistant",
        "content": '{"name": "Cargo Drone", "description": "Carries cargo up to 5kg with a speed of 3m/s"}'
    }

    response = run_chat_hara(
        messages=[
            system_prompt,
            few_shot_user,
            few_shot_assistant,
            {
                "role": "user",
                "content": user_input
            }],
            model=model,
            expected_format="json",
            temperature=0.5)
    

    print(response)
    return response


def extract_persons(system:json, model:str="openai:gpt-4o-mini"):
    system_prompt = {
        "role": "system",
        "content": f"""
        You are an expert in Hazard Analysis and Risk Assessment (HARA). 

        TASK: 
        Analyze the given system description that was extracted from the user input and extract all relevant persons that could interact with the system.

        OUTPUT FORMAT:
        Return only a valid JSON array where each element has exactly these keys:
        - "name": (string, extremely concise, e.g., "Operator", "Maintenance Technician", "Bystander")
        - "role": (string, a concise summary of their interaction with the system)
        """}
    
    few_shot_user = {
        "role": "user",
        "content": "Identifty all relevant persons for the system: {'name': 'Cargo Drone', 'description': 'Carries cargo up to 5kg with a speed of 3m/s'}."
    }

    few_shot_assistant = {
        "role": "assistant",
        "content": """[
            {"name": "Operator", "role": "Controls and monitors the cargo drone during its operations."},
            {"name": "Maintenance Technician", "role": "Performs regular maintenance and repairs on the cargo drone."},
            {"name": "Bystander", "role": "Individuals in the vicinity who may be affected by the drone's operations."},
            {"name": "Cleaning Staff", "role": "Responsible for cleaning the areas where the cargo drone operates."}
        ]"""
    }
    
    response = run_chat_hara(
        messages=[
            system_prompt,
            few_shot_user,
            few_shot_assistant,
            {
                "role": "user",
                "content": f"Identify all relevant persons for the given system. Here is the given system: {system}."
            }],
            model=model,
            expected_format="json",
            temperature=0.5)
    
    print(response)
    return response


def extract_hazards(system:json, model:str="google:gemini-1.5-pro"):
    system_prompt = {
        "role": "system",
        "content": f"""
        You are an expert in Hazard Analysis and Risk Assessment (HARA) and functional safety engineering. Your primary expertise is classifying hazards based on system specifications.

        TASK: 
        Analyze the given system description to identify ALL high-level hazard classes that could potentially arise from its operation, malfunction, or failure.

        OUTPUT FORMAT:
        Retur a single, valid JSON array of strings. Do not include any text before or after the JSON. 
        
        IMPORTANT ABSTRACTION RULES:
        1. **Be Abstract:** The output must use broad, high-level categories (e.g., 'Electrical/Thermal', 'Kinetic') and must NOT use specific failure events (e.g., 'Collision', 'Lifting', 'Falling').
        2. **Group Related Concepts:** Combine related hazards into a single, comprehensive class (e.g., combine 'Software', 'Control System', and 'Cybersecurity' into one class).

        Each element in the array must be the concise name of a hazard class (e.g., "Mechanical", "Electrical/Thermal").
        """}
    
    few_shot_user = {
        "role": "user",
        "content": "Identifty all relevant persons for the system: {'name': 'Cargo Drone', 'description': 'Carries cargo up to 5kg with a speed of 3m/s'}"
    }

    few_shot_assistant = {
    "role": "assistant",
    "content": """[
        "Mechanical", 
        "Electrical", 
        "Kinetic", 
        "Chemical", 
        "Information"
    ]"""
}
    
    response = run_chat_hara(
        messages=[
            system_prompt,
            few_shot_user,
            few_shot_assistant,
            {
                "role": "user",
                "content": f"Identify all potential hazard classes for the given system. Here is the given system: {system}."
            }],
            model=model,
            expected_format="json",
            temperature=0.5)
    
    print(response)
    return response


def define_harm(system:json, person:str, hazard_class:str, model:str="google:gemini-1.5-pro"):
    system_prompt = {
        "role": "system",
        "content": f"""
        You are an expert in Hazard Analysis and Risk Assessment (HARA). 

        TASK: 
        Based on the provided system description, person, and hazard class, please answer the following guide phrase: How could the <<system>> potentially cause harm to the <<person>> through <<hazard class>>?
        When answering, fill in the guide phrase so it is formated in proper English. So intead of "How could the drone potentially cause harm to the Operator through Electric" which doesn't sound right, you would answer "How could the drone potentially cause harm to the Operator through Electricity?".
        The answer returned at the end should be as general as possible without going into specific failure modes or events. Do not give any reasons or explanations, just answer the question directly. So instead of "Mechanic gets electric shock from open wires because of improper maintenance" you would answer "Mechanic gets electric shock.".

        
        OUTPUT FORMAT:
        Return only a valid JSON object with exactly these keys:
        - "guide_phrase": (string, the filled guide phrase)
        - "hazard_class": (string, the provided hazard class)
        - "person": (string, name of the provided person)
        - "harm": (string, a concise description of the specific harm that could result)
        """}
    
    few_shot_user = {
        "role": "user",
        "content": "Define a specific hazard based on the system, person, and hazard class. Here is the given system: {'name': 'Cargo Drone', 'description': 'Carries cargo up to 5kg with a speed of 3m/s'}, the person: 'Mechanic', and the hazard class: 'Electric'."
    }

    few_shot_assistant = {
        "role": "assistant",
        "content": """{
            "guide_phrase": "How could the cargo drone potentially cause harm to the Maintenance Technician through Electricity?",
            "hazard_class": "Electric",
            "person": "Mechanic",
            "harm": "Mechanic gets electric shock."
        }"""
    }
    
    response = run_chat_hara(
        messages=[
            system_prompt,
            few_shot_user,
            few_shot_assistant,
            {
                "role": "user",
                "content": f"Define a specific hazard based on the system, person, and hazard class. Here is the given system: {system}, the person: {person}, and the hazard class: {hazard_class}."
            }],
            model=model,
            expected_format="json",
            temperature=0.5)
    
    print(response)
    return response


def harms(system:json, persons:List[Dict[str, str]], hazards:List[str], model:str="google:gemini-1.5-pro"):
    harms = {}
    for p in persons:
        harm_caused = []
        for h in hazards:
            harm = define_harm(system, p, h, model=model)
            harm_caused.append(harm)
        harms[p['name']] = harm_caused  
    return harms

def harms_summary(harms_dict: Dict[str, List[Dict[str, Any]]], persons: List[Dict[str, str]]) -> List[str]:
    # Given the harms for many persons, there will be many harms that are the same. Please summarise all harms into a single generalised list without duplicates.
    # You could use a LLM to do this, or go through each json object and extract the harm key, then add it to a set to remove duplicates, and finally convert the set back to a list.
    # However it's tough as there are variations in the way harms are described. So using a LLM might be better.
    harms_list = []
    persons_roles = []
    for p in persons:
            role = p["name"].lower()
            split = role.split()
            if len(split) > 1:
                persons_roles.append(split[-1])
            persons_roles.append(role)
    persons_roles = sorted(persons_roles, key=len, reverse=True)

    regex = r"^(" + "|".join(re.escape(p) for p in persons_roles) + r")\s+"
    print("Printing harms turned to Person")

    for listed in harms_dict.values():
        for dic in listed:
            harm = dic["harm"]
            harm = re.sub(regex, "Person ", harm, flags=re.IGNORECASE)
            print(harm)
            harms_list.append(harm)
    
    harms_list = [str(harm) for harm in harms_list]
    print(f"Original harms length: {len(harms_list)}")
    harms_list = list(set(harms_list))
    print(f"Removing all length: {len(harms_list)}")

    transformer = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = transformer.encode(harms_list)
    clusterer = DBSCAN(eps=0.3, min_samples=1, metric="cosine")
    labels = clusterer.fit_predict(embeddings)
    
    harms_embeddings = {}
    for label, harm in zip(labels, harms_list):
        if label not in harms_embeddings:
            harms_embeddings[label] = []
        harms_embeddings[label].append(harm)
    
    cleaned_harms = []
    for id, harms in harms_embeddings.items():
        if id == -1:
            cleaned_harms.extend(harms)
        else:
            cleaned_harms.append(max(harms, key=len))
    print(f"Final length: {len(cleaned_harms)}") 

    return cleaned_harms




def extract_iclasses(system:json, model:str="google:gemini-1.5-pro"):
    system_prompt = {
        "role": "system",
        "content": f"""
        You are an expert in Hazard Analysis and Risk Assessment (HARA). 

        TASK: 
        Based on the provided system description, please extract all the impact class(physical value) of the system. Impact classes would be general operational aspects of the system that could lead to harm, such as "Rotating parts", "Moving parts", "High Voltage", "Chemical exposure", etc.
        The given system will include a Name, namely the type of system, and a Description, which outlines its capabilities and limitations. Please only consider the characteristics that are directly relevant to the system's described functionalities and operational context.

        OUTPUT FORMAT:
        Return only a valid JSON array with strings as elements.
        """}
    
    few_shot_user = {
        "role": "user",
        "content": "Classify the impact class of the given system. Here is the given system: {'name': 'Cargo Drone', 'description': 'Carries cargo up to 5kg with a speed of 3m/s'}."
    }

    few_shot_assistant = {
        "role": "assistant",
        "content": """[
            "Rotating parts", 
            "Position", 
            "Moving parts", 
            "Moving actuators"
        ]"""
    }
    
    response = run_chat_hara(
        messages=[
            system_prompt,
            few_shot_user,
            few_shot_assistant,
            {
                "role": "user",
                "content": f"Classify the impact class of the given system. Here is the given system: {system}."
            }],
            model=model,
            expected_format="json",
            temperature=0)
    
    print(response)
    return response


def define_impact(system:json, impact_class:str, harms: str, model:str="google:gemini-1.5-pro"):

    system_prompt = {
        "role": "system",
        "content": f"""
        You are an expert in Hazard Analysis and Risk Assessment (HARA). 

        TASK: 
        Based on the provided system description, given impact class, and a given harm, please answer the following guide phrase: By influencing which physical value through <<impact class>> could the <<system>> cause <<harm>>?
        The harm given is defined for a specific person with a role, like an engineer or a mechanic. I want you to generalise the harm so that it is not linked to a specific person, but rather just the harm itself. 
        For example, instead of "Safety Officer gets struck by the vehicle" you would change it to "Vehicle strikes a person".


        OUTPUT FORMAT:
        Return only a valid JSON object with exactly these keys:
        - "impact_class": (string, the provided impact class)
        - "physical_value": (array of strings, the physical values that could be influenced through the impact class)
        - "harm_caused": (string, generalised harm that could result)
        """}
    
    few_shot_user = {
        "role": "user",
        "content": "Define the impact based on the system, impact class, and harm. Here is the given system: {'name': 'Cargo Drone', 'description': 'Carries cargo up to 5kg with a speed of 3m/s'}, the impact class: 'Rotating parts', and the harm: 'Mechanic gets electric shock.'}."
    }

    few_shot_assistant = {
        "role": "assistant",
        "content": """{
            "guide_phrase": "By influencing which physical value through rotating parts could the cargo drone hit a person?",
            "impact_class": "Rotating parts",
            "physical_value": ["The drone's propelles are rotating (RPM)", "The drone's propellers generate lift force"]}
            """
    }

    response = run_chat_hara(
        messages=[ 
            system_prompt,
            few_shot_user,
            few_shot_assistant,
            {
                "role": "user",
                "content": f"Define the impact based on the system, impact class, and harm. Here is the given system: {system}, the impact class: {impact_class}, and the harm: {harms}."
            }],
            model=model,
            expected_format="json",
            temperature=0.5)
    
    print(response)
    return response


def impacts(system:json, impact_classes:List[str], harms_dict:Dict[str, Any], model:str="google:gemini-1.5-pro"):
    impacts = {}
    for ic in impact_classes:
        impact_list = []
        for harms in harms_dict.items():
            for harm in harms:
                impact = define_impact(system, ic, harm['harm'], model=model)
                impact_list.append(impact)
        impacts[ic] = impact_list  
    return impacts


def identify_failure_modes(system:json, model:str="google:gemini-1.5-pro"):
    system_prompt = {
        "role": "system",
        "content": f"""
        You are an expert in Hazard Analysis and Risk Assessment (HARA) and functional safety engineering.

        TASK: 
        Identify the **predefined generic failure modes** that can apply to the actuation and control functions of the given system.

        CONSTRAINT: 
        The failure modes MUST be based ONLY on the temporal (timing) and value deviations of an action. 
        - DO NOT create failure modes specific to the system's components (e.g., 'Sensor failure,' 'Brake Malfunction').
        - DO NOT combine classes (e.g., 'Provision Commission/Omission').

        OUTPUT FORMAT:
        Return ONLY a single, valid JSON array where each element has exactly these keys, adhering strictly to the **predefined failure model** shown in the few-shot example:
        - "failure_mode": (string, concise, **predefined** classification like Timing or Provision)
        - "description": (string, brief, **generic** explanation of the failure mode)
        """}
    
    few_shot_user = {
        "role": "user",
        "content": "Based on the system description: {'name': 'Cargo Drone', 'description': 'Carries cargo up to 5kg with a speed of 3m/s'}, select only the relevant predefined temporal and value failure modes."
    }

    few_shot_assistant = {
        "role": "assistant",
        "content": """[
            {"failure_mode": "Provision Commission", "description": "Something is actuated even though it must not at the point in time."},
            {"failure_mode": "Provision Omission", "description": "Something is not actuated although it must be at the point in time."},
            {"failure_mode": "Timing Early", "description": "Something is actuated earlier than intended."},
            {"failure_mode": "Timing Late", "description": "Something is actuated later than intended."}
            {"failure_mode": "Value Too High", "description": "Something is actuated to a higher value than intended."},
            {"failure_mode": "Value Too Low", "description": "Something is actuated to a lower value than intended."},
            {"failure_mode": "Value Incorrect", "description": "Something is actuated to an incorrect value."}
        ]"""
    }

    response = run_chat_hara(
        messages=[
            system_prompt,
            few_shot_user,
            few_shot_assistant,
            {
                "role": "user",
                "content": f"Based on the system description: {system}, select only the relevant predefined temporal and value failure modes from the list.."
            }],
            model=model,
            expected_format="json",
            temperature=0.5)
    
    print(response)
    return response

    


    


if __name__ == "__main__":
    system = extract_system("A mobile robot (AGV) transports heavy pallets in a warehouse shared with human workers. It has a lifting fork mechanism.", model="openai:gpt-4o")
    persons = extract_persons(system, model="openai:gpt-4o")
    hazards = extract_hazards(system, model="openai:gpt-4o")
    harms_dict = harms(system, persons, hazards, model="openai:gpt-4o")
    harms_summary_list = harms_summary(harms_dict, persons)
    print(harms_summary_list)
    #impact_classes = extract_iclasses(system, model="openai:gpt-4o")
    #impact = define_impact(system, impact_classes[0], "Bystander gets hit by the vehicle.", model="openai:gpt-4o")
    #impacts_dict = impacts(system, impact_classes, harms_dict, model="openai:gpt-4o")
    #failure_modes = identify_failure_modes(system, model="openai:gpt-4o")
            
        

    
