import re
import aisuite as ai
from dotenv import load_dotenv
import json
import os
import ast
from typing import List, Dict, Any
from HELPERS import *
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# added semantic embeddings - pip install sentence-transformers scikit-learn
# from sentence_transformers import SentenceTransformer
# from sklearn.cluster import DBSCAN

console = Console()

_ = load_dotenv()
client = ai.Client()


def extract_system(user_input: str, model: str = "google:gemini-1.5-pro"):
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

    few_shot_user1 = {
        "role": "user",
        "content": "A cargo drone that carry cargos up to 5kg with a speed of 3m/s."
    }

    few_shot_assistant1 = {
        "role": "assistant",
        "content": '{"name": "Cargo Drone", "description": "Carries cargo up to 5kg with a speed of 3m/s"}'
    }

    few_shot_user2 = {
        "role": "user",
        "content": "A mobile robot transports heavy pallets in a warehouse. It has a lifting fork mechanism."
    }

    few_shot_assistant2 = {
        "role": "assistant",
        "content": '{"name": "Automated Guided Vehicle", "description": "Mobile robot transporting heavy pallets using a lifting fork mechanism in a warehouse."}'
    }

    response = run_chat_hara(
        messages=[
            system_prompt,
            few_shot_user1,
            few_shot_assistant1,
            few_shot_user2,
            few_shot_assistant2,
            {
                "role": "user",
                "content": user_input
            }],
        model=model,
        expected_format="json",
        temperature=0.8)

    return response


def extract_persons(system: json, model: str = "openai:gpt-4o-mini"):
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

    few_shot_user1 = {
        "role": "user",
        "content": "Identifty all relevant persons for the system: {'name': 'Cargo Drone', 'description': 'Carries cargo up to 5kg with a speed of 3m/s'}."
    }

    few_shot_assistant1 = {
        "role": "assistant",
        "content": """[
            {"name": "Operator", "role": "Controls and monitors the cargo drone during its operations."},
            {"name": "Maintenance Technician", "role": "Performs regular maintenance and repairs on the cargo drone."},
            {"name": "Bystander", "role": "Individuals in the vicinity who may be affected by the drone's operations."},
            {"name": "Cleaning Staff", "role": "Responsible for cleaning the areas where the cargo drone operates."}
        ]"""
    }

    few_shot_user2 = {
        "role": "user",
        "content": "Identifty all relevant persons for the system: {'name': 'Automated Guided Vehicle', 'description': 'Mobile robot transporting heavy pallets using a lifting fork mechanism in a warehouse.'}"
    }

    few_shot_assistant2 = {
        "role": "assistant",
        "content": """[
            {"name": "Warehouse Worker", "role": "Performs manual tasks and operates other machinery in the AGV's vicinity."},
            {"name": "Maintenance Technician", "role": "Performs diagnostic checks, repairs, and calibration on the vehicle's mechanics and software."},
            {"name": "Fleet Supervisor", "role": "Manages the AGV's schedule, path planning, and emergency stops."},
            {"name": "Bystander", "role": "Non-affiliated visitors or personnel temporarily passing through the operational area."}
        ]"""
    }

    response = run_chat_hara(
        messages=[
            system_prompt,
            few_shot_user1,
            few_shot_assistant1,
            few_shot_user2,
            few_shot_assistant2,
            {
                "role": "user",
                "content": f"Identify all relevant persons for the given system. Here is the given system: {system}."
            }],
        model=model,
        expected_format="json",
        temperature=0.8)

    return response


def extract_hazards(system: json, model: str = "google:gemini-1.5-pro"):
    system_prompt = {
        "role": "system",
        "content": f"""
        You are an expert in Hazard Analysis and Risk Assessment (HARA) and functional safety engineering. Your primary expertise is classifying hazards based on system specifications.

        TASK: 
        Analyze the given system description to identify ALL relevant high-level hazard classes.    

        CRITICAL ABSTRACTION RULES:
        1. **Use ONLY** the following high-level categories: ["Mechanical", "Kinetic", "Electrical", "Thermal", "Chemical", "Radiation", "Ergonomics", "Software/Control System", "Noise/Vibration"].
        2. Select only the classes that are logically applicable to the system's described functionality and environment.
        3. DO NOT include any classes not in the provided list.

        OUTPUT FORMAT:
        Return a single, valid JSON array of strings. Do not include any text before or after the JSON.
        """}

    few_shot_user1 = {
        "role": "user",
        "content": "Identifty all relevant hazards for the system: {'name': 'Cargo Drone', 'description': 'Carries cargo up to 5kg with a speed of 3m/s'}"
    }

    few_shot_assistant1 = {
        "role": "assistant",
        "content": """[
        "Mechanical", 
        "Electrical", 
        "Kinetic", 
        "Chemical", 
        "Information"
    ]"""
    }

    few_shot_user2 = {
        "role": "user",
        "content": "Identifty all relevant hazards for the system: {'name': 'Automated Guided Vehicle', 'description': 'Mobile robot transporting heavy pallets using a lifting fork mechanism in a warehouse.'}"
    }

    few_shot_assistant2 = {
        "role": "assistant",
        "content": """[
            "Mechanical", 
            "Kinetic", 
            "Electrical", 
            "Thermal", 
            "Software/Control System"
        ]"""
    }

    response = run_chat_hara(
        messages=[
            system_prompt,
            few_shot_user1,
            few_shot_assistant1,
            few_shot_user2,
            few_shot_assistant2,
            {
                "role": "user",
                "content": f"Identify all potential hazard classes for the given system. Here is the given system: {system}."
            }],
        model=model,
        expected_format="json",
        temperature=0.8)

    return response


def define_harm(system: json, person: str, hazard_class: str, model: str = "google:gemini-1.5-pro"):
    system_prompt = {
        "role": "system",
        "content": f"""
        You are an expert in Hazard Analysis and Risk Assessment (HARA). 

        TASK: 
        Based on the provided system description, person, and hazard class, answer the guide phrase: How could the <<system>> potentially cause harm to the <<person>> through <<hazard class>>?

        CRITICAL OUTPUT RULE:
        The 'harm' value must be a concise, general description of the harm caused. Do not use specific person names, and do not include the ultimate consequence (e.g., 'Death', 'Amputation').
        Do not give any reasons as well, only consider the harm done. So for example, instead of 'Electric shock or electrical burns from contact with exposed or damaged robot wiring, connectors, or end-effector electronics during inspection.', just return 'Electric shocl or electrical burns'.

        OUTPUT FORMAT:
        Return only a valid JSON object with exactly these keys:
        - "guide_phrase": (string, the filled guide phrase)
        - "hazard_class": (string)
        - "person": (string)
        - "harm": (string)
        """}

    few_shot_user1 = {
        "role": "user",
        "content": "Define a specific hazard based on the system, person, and hazard class. Here is the given system: {'name': 'Cargo Drone', 'description': 'Carries cargo up to 5kg with a speed of 3m/s'}, the person: 'Mechanic', and the hazard class: 'Electric'."
    }

    few_shot_assistant1 = {
        "role": "assistant",
        "content": """{
            "guide_phrase": "How could the cargo drone potentially cause harm to the Mechanic through Electricity?",
            "hazard_class": "Electric",
            "person": "Mechanic",
            "harm": "Mechanic gets electric shock."
        }"""
    }

    few_shot_user2 = {
        "role": "user",
        "content": "Define a specific hazard based on the system, person, and hazard class. Here is the given system: {'name': 'Automated Guided Vehicle', 'description': 'Mobile robot transporting heavy pallets'}, the person: 'Warehouse Worker', and the hazard class: 'Mechanical'."
    }

    few_shot_assistant2 = {
        "role": "assistant",
        "content": """{
                "guide_phrase": "How could the Automated Guided Vehicle potentially cause harm to the Warehouse Worker through Mechanical means?",
                "hazard_class": "Mechanical",
                "person": "Warehouse Worker",
                "harm": "Warehouse Worker gets crushed or trapped."
        }"""
    }

    response = run_chat_hara(
        messages=[
            system_prompt,
            few_shot_user1,
            few_shot_assistant1,
            few_shot_user2,
            few_shot_assistant2,
            {
                "role": "user",
                "content": f"Define a specific hazard based on the system, person, and hazard class. Here is the given system: {system}, the person: {person}, and the hazard class: {hazard_class}."
            }],
        model=model,
        expected_format="json",
        temperature=0.8)

    return response


def harms(system: json, persons: List[Dict[str, str]], hazards: List[str], model: str = "google:gemini-1.5-pro"):
    harms = {}
    for p in persons:
        harm_caused = []
        for h in hazards:
            harm = define_harm(system, p, h, model=model)
            harm_caused.append(harm)
        harms[p['name']] = harm_caused
    return harms


def harms_summary(harms_dict: Dict[str, List[Dict[str, Any]]], model: str = "google:gemini-1.5-pro") -> List[str]:
    # Given the harms for many persons, there will be many harms that are the same. Please summarise all harms into a single generalised list without duplicates.
    # You could use a LLM to do this, or go through each json object and extract the harm key, then add it to a set to remove duplicates, and finally convert the set back to a list.
    # However it's tough as there are variations in the way harms are described. So using a LLM might be better.
    """if 0:
        harms_list = []
        persons_roles = []
        for p in persons:
                role = p["name"].lower()
                split = role.split()
                if len(split) > 1:
                    persons_roles.append(split[-1])
                persons_roles.append(role)
        persons_roles = sorted(persons_roles, key=len, reverse=True)

        regex = r"^(" + "|".join(re.escape(p) for p in persons_roles) + r")"
        print("Printing harms turned to Person")

        for listed in harms_dict.values():
            for dic in listed:
                harm = dic["harm"]
                harm = re.sub(regex, "Person", harm, flags=re.IGNORECASE)
                print(harm)
                harms_list.append(harm)

        harms_list = [str(harm) for harm in harms_list]
        print(f"Original harms length: {len(harms_list)}")
        harms_list = list(set(harms_list))
        print(f"Removing all length: {len(harms_list)}")

        transformer = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = transformer.encode(harms_list)
        clusterer = DBSCAN(eps=0.1, min_samples=1, metric="cosine")
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
        print(f"Final length: {len(cleaned_harms)}") """
    harms = []
    for harms_list in harms_dict.values():
        for dic in harms_list:
            harms.append(dic["harm"])

    system_prompt = {
        "role": "system",
        "content": f"""
        You are a diligent thinker and you hate repeating yourself.

        TASK:
        You must review a list of scenarios, endangering the life of a person. Semantically deduplicate them into an array of unique harms.

        UNBREAKABLE RULE:
        If you find redundancy:
        - Prefer rewriting the more detailed harm to remove the overlapping information.
        - Remove a harm only if all of its information is contained in another harm.

        RULES:
        1. Eliminate only scenarios that describe the SAME harm, even if paraphrased.
        2. Do not eliminate scenarios if they have a difference in:
            - injury mechanism (e.g. struck, crushed, trapped)
            - source (vehicle moving <-> lifting mechanism <-> electrical)
            - failure (normal behavior <-> malfunctioning or unintended behavior)
        3. Keep the most information-dense and generalized scenario for each UNIQUE danger.
        4. Do not oversaturate a harm with information - a harm must containt no more than 3 semantically significant phrases.

        OUTPUT FORMAT:
        Return only a valid JSON array of strings without any trailing or preceding spaces.
        """
    }

    few_shot_user = {
        "role": "user",
        "content": f"{['Electric shock or electrical burns', 'Thermal burns', 'Chemical irritation or burns', 'Musculoskeletal strain or repetitive stress injury',
                       'Unexpected robot motion causing impact or crushing injuries', 'Hearing impairment or hand-arm vibration injury', 'Pinching or impact injuries', 'Impact injuries or bruising',
                       'Electric shock or electrical burns', 'Thermal burns', 'Skin or eye irritation', 'Musculoskeletal strain', 'Impact injury or pinching', 'Hearing discomfort or vibration-related hand and arm discomfort',
                       'Pinching or impact injuries', 'Impact injuries or bruising']}"
    }

    few_shot_assistant = {
        "role": "assistant",
        "content": """{
        'Electric shock or electrical burns', 'Thermal burns', 'Chemical irritation or burns', 'Musculoskeletal strain or repetitive stress injury',
        'Unexpected robot motion causing impact or crushing injuries', 'Hearing discomfort or vibration-related hand and arm discomfort or injury',
        'Pinching or impact injuries', 'Bruising', 'Skin or eye irritation'
        }"""
    }

    response = response = run_chat_hara(
        messages=[
            system_prompt,
            few_shot_user,
            few_shot_assistant,
            {
                "role": "user",
                "content": f"""{harms}"""
            }],
        model=model,
        expected_format="json",
        temperature=1)

    return response


def extract_iclasses(system: json, model: str = "google:gemini-1.5-pro"):
    system_prompt = {
        "role": "system",
        "content": f"""
        You are an expert in Hazard Analysis and Risk Assessment (HARA). 

        TASK: 
        Based on the provided system description, please extract all the impact class(physical value) of the system. Impact classes would be general operational aspects of the system that could lead to harm, such as "Rotating parts", "Moving parts", etc.
        Do not yet consider any interaction with persons or specific harms, only focus on the system itself.
        The given system will include a Name, namely the type of system, and a Description, which outlines its capabilities and limitations. Please only consider the characteristics that are directly relevant to the system's described functionalities and operational context.

        OUTPUT FORMAT:
        Return only a valid JSON array with strings as elements.
        """}

    few_shot_user1 = {
        "role": "user",
        "content": "Classify the impact class of the given system. Here is the given system: {'name': 'Cargo Drone', 'description': 'Carries cargo up to 5kg with a speed of 3m/s'}."
    }

    few_shot_assistant1 = {
        "role": "assistant",
        "content": """[
            "Rotating parts", 
            "Position", 
            "Moving parts", 
            "Moving actuators"
        ]"""
    }

    few_shot_user2 = {
        "role": "user",
        "content": "Classify the impact class of the given system. Here is the given system: {'name': 'Automated Guided Vehicle', 'description': 'Mobile robot transporting heavy pallets using a lifting fork mechanism in a warehouse.'}"
    }

    few_shot_assistant2 = {
        "role": "assistant",
        "content": """[
            "Moving parts",
            "Lifting mechanism",
            "Moving actuators",
            "Position"
        ]"""
    }

    response = run_chat_hara(
        messages=[
            system_prompt,
            few_shot_user1,
            few_shot_assistant1,
            few_shot_user2,
            few_shot_assistant2,
            {
                "role": "user",
                "content": f"Classify the impact class of the given system. Here is the given system: {system}."
            }],
        model=model,
        expected_format="json",
        temperature=0.8)

    return response


def define_impact(system: json, impact_class: str, harms: str, model: str = "google:gemini-1.5-pro"):
    system_prompt = {
        "role": "system",
        "content": f"""
        You are an expert in Hazard Analysis and Risk Assessment (HARA). 

        TASK: 
        Based on the provided system description, given impact class, and a given harm, please answer the following guide phrase: By influencing which physical value through <<impact class>> could the <<system>> cause <<harm>>?
        The harm given is defined for a specific person with a role, like an engineer or a mechanic. I want you to generalise the harm so that it is not linked to a specific person, but rather just the harm itself. 
        For example, instead of "Safety Officer gets struck by the vehicle" you would change it to "Vehicle strikes a person".
        The physical value at the end must make sense with the impact class and the harm caused.


        OUTPUT FORMAT:
        Return only a valid JSON object with exactly these keys:
        - "impact_class": (string, the provided impact class)
        - "physical_value": (array of strings, the physical values that could be influenced through the impact class)
        - "harm_caused": (string, generalised harm that could result)
        """}

    few_shot_user1 = {
        "role": "user",
        "content": "Define the impact based on the system, impact class, and harm. Here is the given system: {'name': 'Cargo Drone', 'description': 'Carries cargo up to 5kg with a speed of 3m/s'}, the impact class: 'Rotating parts', and the harm: 'Mechanic hit by drone.'}."
    }

    few_shot_assistant1 = {
        "role": "assistant",
        "content": """{
            "guide_phrase": "By influencing which physical value through rotating parts could the cargo drone hit a person?",
            "impact_class": "Rotating parts",
            "physical_value": ["The drone's propelles are rotating (RPM)"]}
            """
    }

    few_shot_user2 = {
        "role": "user",
        "content": "Define the impact based on the system, impact class, and harm. Here is the given system: {'name': 'Automated Guided Vehicle', 'description': 'Mobile robot transporting heavy pallets'}, the impact class: 'Moving parts', and the harm: 'Warehouse Worker gets crushed.'}."
    }

    few_shot_assistant2 = {
        "role": "assistant",
        "content": """{
            "guide_phrase": "By influencing which physical value through moving parts could the Automated Guided Vehicle cause injury?",
            "impact_class": "Moving parts",
            "physical_value": ["The vehicle's fork lift is moving"]}
    }"""
    }

    # few_shot_user3 = {
    # "role": "user",
    # "content": "Define the impact based on the system, impact class, and harm. Here is the given system: {'name': 'Automated Guided Vehicle', 'description': 'Mobile robot transporting heavy pallets'}, the impact class: 'Moving parts', and the harm: 'Warehouse worker gets electric shock.'}."
    # }
    #
    # few_shot_assistant3 = {
    #    "role": "assistant",
    #    "content": """{
    #            "guide_phrase": "By influencing which physical value through moving parts could the Automated Guided Vehicle cause injury?",
    #            "impact_class": "Moving parts",
    #            "physical_value": ["N/A"]}
    #    }"""
    # }

    response = run_chat_hara(
        messages=[
            system_prompt,
            few_shot_user1,
            few_shot_assistant1,
            few_shot_user2,
            few_shot_assistant2,
            # few_shot_user3,
            # few_shot_assistant3,
            {
                "role": "user",
                "content": f"Define the impact based on the system, impact class, and harm. Here is the given system: {system}, the impact class: {impact_class}, and the harm: {harms}."
            }],
        model=model,
        expected_format="json",
        temperature=0.8)

    return response


def impacts(system: json, impact_classes: List[str], harms_summary, model: str = "google:gemini-1.5-pro"):
    impacts = {}
    for ic in impact_classes:
        impact_list = []
        for harm in harms_summary:
            impact = define_impact(system, ic, harm, model=model)
            impact_list.append(impact)
        impacts[ic] = impact_list
    return impacts


def identify_failure_modes(system: json, model: str = "google:gemini-1.5-pro"):
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
        temperature=0.8)

    return response


def define_actuators(system: json, impact_classes: List[str], model: str = "google:gemini-1.5-pro"):
    system_prompt = {
        "role": "system",
        "content": f"""
        You are an expert in Hazard Analysis and Risk Assessment (HARA) and the field elechtromechanics.

        TASK: 
        You are given a system description and a list of impact classes. 
        Identify the mechanical actuators that directly or indirectly influence the physical parameters (force, speed, RPM, etc.) of the system.

        RULES:
        1. The extracted actuators must be a specific physical part of the system hardware mechanisms.
        2. Add descriptive adjectives only when they bring significant additional information (e.g. generate "Drone rotor" instead of "Bottom left drone rotor").
        3. If an actuator is part of a specific component in the machine, then keep the details for it (e.g. "Engine cog wheel" instead of "Cog wheel").
        4. Keep the total list of actuators unique across all impact classes; don't repeat components.

        OUTPUT FORMAT:
        Return a valid JSON list of dictionaries, where the key is an impact class (string) and the value is a string of all associated actuators, e.g.:
        - "impact_class" : "Flying Mechanism"
        - "actuators" : "["Propeller rotors", "Electronic speed controllers"]" 
        """}

    few_shot_user = {
        "role": "user",
        "content": "Given the system: {'name': 'Robot arm', 'description': 'Loads and offloads packages from a truck.'} \
        and the impact_classes: ['Moving parts', 'Heavy loads'], identify all unique actuators connected with each impact class."
    }

    few_shot_assistant = {
        "role": "assistant",
        "content": """[
            {"impact_class": "Moving parts", "actuators": ["Arm gripper", "Forearm lever", "Joint motors"]},
            {"impact_class": "Heavy loads", "actuators": ["Hydraulic cylinder", "Load-bearing frame"]}
        ]"""
    }

    response = run_chat_hara(
        messages=[
            system_prompt,
            few_shot_user,
            few_shot_assistant,
            {
                "role": "user",
                "content": f"Based on the system description: {system}, and the impact classes: {impact_classes}, generate and associate a comprehensive, but unique, collection of actuators for each impact class."
            }],
        model=model,
        expected_format="json",
        temperature=0.8)

    return response


def extract_failure(system: json, failure_mode: str, actuator: str, impact: str, model: str = "google:gemini-1.5-pro"):
    system_prompt = {
        "role": "system",
        "content": """
        You are an expert in Hazard Analysis and Risk Assessment (HARA).

        TASK:
        Analyze the situation given the description of a system, the specified failure mode, actuator and an impact.
        Infer the answer to the question:
        "To which failure would a <<failure mode>> of actuator <<actuator>> lead, causing <<impact>>?"
        List all plausible system failures that:
        - share logical connection with the failure mode and the actuator.
        - can directly or indirectly cause the impact.
        Return the most relevant 5 failures.

        DEFINITIONS:
        - Failure modes are a predefined set: ("Provision Commission", "Provision Ommision", "Value too low", "Value too high", "Value incorrect", "Timing early", "Timing late")
        - Actuator is the active physical component (e.g. Arm Gripper).
        - Impact is a hazardous/dangerous situation, or harm. (e.g. explosion, fire, electric arc).
        - A failure is a system deviation in an actuator or system behavior.

        RULES:
        1. Failures MUST be a clear detailed system event (no abstraction or ambiguity).
        2. The impact and MUST NOT be repeated in the failure.
        2. A failure MUST NOT contain information about what the failure causes, because it is already contained in the impact.

        OUTPUT FORMAT:
        Return only a valid JSON dictionary of the form:
        {'question': '*the original wording of the question with the replaced variables*'}, 'failures': *[list of all identified failures]*}
        """
    }

    few_shot_user = {
        "role": "user",
        "content": """Give me all failures connected to this situation: 
        - system: {'name': 'Cargo Drone', 'description': 'Carries cargo up to 5kg with a speed of 3m/s'}
        - failure mode:  {'failure_mode': 'Provision Commission', 'description': 'Something is actuated even though it must not at the point in time.'}
        - actuator : {'actuator': 'Rotors'}. 
        - impact: {'impact': 'drone moving unsafely'}
        """
    }

    few_shot_assistant = {
        "role": "assistant",
        "content": """{
        'question': 'To which failure would a Provision Comission of actuator Rotors lead, causing the drone moving unsafely?', 
        'failures': 
        ['One or more rotors start to / are rotating even though they must not at that point in time', 'One or more rotors start to rotate too late',
        'One or more rotors stop rotating too late', 'One or more rotors change rotation speed too late']
        }]
        """
    }

    few_shot_user2 = {
        "role": "user",
        "content": """Give me all failures connected to this situation: 
        - system: {'name': 'Robot arm', 'description': 'A stationary robot arm that can pick up and drop packages loaded onto a pallet, weight capacity is 100 kg.}
        - failure mode:  {'failure_mode': 'Value Incorrect', 'description': 'Something is actuated to an incorrect value.'}
        - actuator : {'actuator': 'Electromechanical brake'}. 
        - impact: {'impact': 'Impact injury from the robot arm'}
        """
    }

    few_shot_assistant2 = {
        "role": "assistant",
        "content": """{
        "question": "To which failure would a Value too low of actuator Electromechanical brake lead, causing impact injury from the robot arm?",
        "failures": [
            "Electromechanical brake does not activate upon machine switch off", "Electromechanical brake does not stop initiated motion", "Electromechanical brake in gripper deactivates prematurely"
        ]
        }"""
    }

    response = run_chat_hara(
        messages=[
            system_prompt,
            few_shot_user,
            few_shot_assistant,
            few_shot_user2,
            few_shot_assistant2,
            {
                "role": "user",
                "content": f"""Define a multitude of unique failures associated with:
                - system: {system}
                - failure mode: {failure_mode}
                - actuator: {actuator}
                - impact: {impact}
            """}],
        model=model,
        expected_format="json",
        temperature=0.8)

    return response


def collect_failures(system: json, failure_modes: List[Dict[str, str]], actuators: List[Dict[str, List[str]]],
                     impacts: List[str], model: str = "google:gemini-1.5-pro"):
    failures = {}
    for failure_mode in failure_modes:
        for actuator in actuators:
            failures_list = []
            for a in actuator["actuators"]:
                for impact in impacts:
                    failure = extract_failure(system, failure_mode, a, impact, model)
                    failures_list.append(failure)
                if a not in failures.keys():
                    failures[a] = failures_list
                else:
                    failures[a].extend(failures_list)

    return failures


from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import List, Dict, Any


def display_system(system: Dict[str, str]):
    console.rule("[bold #6495ED]==== STEP 1: SYSTEM DEFINITION (extract_system) ====")
    try:
        system_panel = Panel(
            f"[bold cyan]SYSTEM NAME:[/bold cyan] {system.get('name', 'N/A')}\n"
            f"[bold cyan]DESCRIPTION:[/bold cyan] {system.get('description', 'N/A')}",
            title="[bold blue]System Under Analysis (S.U.A.)[/bold blue]",
            border_style="blue"
        )
        console.print(system_panel)
    except Exception:
        console.print("[bold red]ERROR: Could not process System data. Check input format.[/bold red]")

def display_persons(persons: List[Dict[str, str]]):
    console = Console()
    console.rule("[bold #6495ED]==== STEP 2: PERSONS AT RISK (extract_persons) ====")
    try:
        if persons:
            persons_table = Table(title="Persons Interacting with the System", show_header=True,
                                  header_style="bold magenta")
            persons_table.add_column("Name", style="#90EE90", justify="left")
            persons_table.add_column("Role and Interaction Summary", style="dim", justify="left")

            for p in persons:
                persons_table.add_row(p.get('name', 'N/A'), p.get('role', 'N/A'))

            console.print(persons_table)
        else:
            console.print("No persons identified.")
    except Exception as e:
        print(e)
        console.print("[bold red]ERROR: Could not process Persons data. Check input format.[/bold red]")

def display_hazards(hazards: List[str]):
    console = Console()
    console.rule("[bold #6495ED]==== STEP 3: HIGH-LEVEL HAZARD CLASSES (extract_hazards) ====")
    try:
        if hazards and isinstance(hazards, list):
            #
            hazard_str = f"[bold yellow]{', '.join(hazards)}[/bold yellow]"
            console.print(Panel(
                hazard_str,
                title="[bold yellow]Applicable Hazard Categories[/bold yellow]",
                border_style="yellow"))
        else:
            console.print("No hazards identified or data format is incorrect.")
    except Exception:
        console.print("[bold red]ERROR: Could not process Hazard Classes data. Check input format.[/bold red]")

def display_harms(harms_summary_list: List[str]):
    console = Console()
    console.rule("[bold #6495ED]==== STEP 4: CONSOLIDATED UNIQUE HARMS (harms_summary) ====")
    if harms_summary_list:
        harm_list_formatted = []
        for idx, harm in enumerate(harms_summary_list):
            harm_list_formatted.append(f"[bold red]{idx + 1}.[/bold red] {harm}")

        console.print(Panel(
            "\n".join(harm_list_formatted),
            title="[bold red]Unique Harm Events (Post-Clustering)[/bold red]",
            border_style="red"))
    else:
        console.print("No unique harms identified.")

def display_impacts(impacts_dict: Dict[str, List[Dict[str, Any]]]):
    console = Console()
    console.rule("[bold #6495ED]==== STEP 5: IMPACT MATRIX (extract_iclasses / define_impact) ====")
    impact_table = Table(
        title="Impact Classes & Associated Physical Values",
        show_header=True,
        header_style="bold green"
    )
    impact_table.add_column("Impact Class", style="#ADFF2F", justify="left")
    impact_table.add_column("Associated Quantifiable Physical Values", style="dim", justify="left")

    for ic, impact_list in impacts_dict.items():
        all_pvs = set()
        if impact_list and isinstance(impact_list, list):
            # Aggregate unique physical values across all harms for this Impact Class
            for impact_dict in impact_list:
                if 'physical_value' in impact_dict and isinstance(impact_dict['physical_value'], list):
                    # Filter out 'N/A' responses if they exist
                    for pv in impact_dict['physical_value']:
                        if pv and 'n/a' not in pv.lower():
                            all_pvs.add(pv)

            # Sort and display a maximum of 4 key values
            sorted_pvs = sorted(list(all_pvs))
            value_str = ", ".join(sorted_pvs[:4])
            if len(sorted_pvs) > 4:
                value_str += ", [i]... more[/i]"
        else:
            value_str = "[i]No relevant physical values extracted.[/i]"

        impact_table.add_row(ic, value_str or "[i]None identified[/i]")

    console.print(impact_table)

def display_failure_modes(failure_modes: List[Dict[str, str]]):
    console = Console()
    console.rule("[bold #6495ED]==== STEP 6: IDENTIFIED FAILURE MODES ====")
    try:
        if failure_modes:
            failure_modes_table = Table(title="Failure Modes of the System", show_header=True, header_style="bold red")
            failure_modes_table.add_column("Failure Mode", style="#6A0C0C", justify="left")
            failure_modes_table.add_column("Description", style="dim", justify="left")

            for m in failure_modes:
                failure_modes_table.add_row(m.get('failure_mode', 'N/A'), m.get('description', 'N/A'))

            console.print(failure_modes_table)
        else:
            console.print("No failure modes identified error.")
    except Exception:
        console.print("[bold red]ERROR: Could not process failure modes data. Check input format.[/bold red]")

def display_actuators(actuators: List[Dict[str, Any]]):
    console = Console()
    console.rule("[bold #6495ED]==== STEP 7: ASSIGNED ACTUATORS ====")
    try:
        if actuators:
            actuators_table = Table(title="Actuators Associated with Impact Classes", show_header=True,
                                    header_style="bold green")
            actuators_table.add_column("Impact classes", style="#26AF00", justify="left")
            actuators_table.add_column("Actuators", style="dim", justify="left")

            for a in actuators:
                actuators_table.add_row(a.get('impact_class', 'N/A'), ", ".join(a.get('actuators', [])))

            console.print(actuators_table)
        else:
            console.print("No actuators identified error.")
    except Exception:
        console.print("[bold red]ERROR: Could not process actuators data. Check input format.[/bold red]")

def display_hara_summary(
        system: Dict[str, str],  # Expects Python Dict: {'name': '...', 'description': '...'}
        persons: List[Dict[str, str]],  # Expects Python List of Dicts: [{'name': '...', 'role': '...'}, ...]
        hazards: List[str],  # Expects Python List of Strings: ['Mechanical', 'Kinetic', ...]
        harms_summary_list: List[str],
        impacts_dict: Dict[str, List[Dict[str, Any]]],
        failure_modes_list: List[Dict[str, str]],
        actuators_list: List[Dict[str, Any]],
        failures_dict: Dict[str, List[Dict[str, List[str]]]]
):
    """
    Formats and prints the results of the HARA pipeline steps using the 'rich' library
    for clear, segmented, and professional output.
    """
    console = Console()

    # --- 1. System Definition (STEP 1) ---
    console.rule("[bold #6495ED]==== STEP 1: SYSTEM DEFINITION (extract_system) ====")
    try:
        system_panel = Panel(
            f"[bold cyan]SYSTEM NAME:[/bold cyan] {system.get('name', 'N/A')}\n"
            f"[bold cyan]DESCRIPTION:[/bold cyan] {system.get('description', 'N/A')}",
            title="[bold blue]System Under Analysis (S.U.A.)[/bold blue]",
            border_style="blue"
        )
        console.print(system_panel)
    except Exception:
        console.print("[bold red]ERROR: Could not process System data. Check input format.[/bold red]")

    # --- 2. Persons at Risk (STEP 2) ---
    console.rule("[bold #6495ED]==== STEP 2: PERSONS AT RISK (extract_persons) ====")
    try:
        if persons:
            persons_table = Table(title="Persons Interacting with the System", show_header=True,
                                  header_style="bold magenta")
            persons_table.add_column("Name", style="#90EE90", justify="left")
            persons_table.add_column("Role and Interaction Summary", style="dim", justify="left")

            for p in persons:
                persons_table.add_row(p.get('name', 'N/A'), p.get('role', 'N/A'))

            console.print(persons_table)
        else:
            console.print("No persons identified.")
    except Exception:
        console.print("[bold red]ERROR: Could not process Persons data. Check input format.[/bold red]")

    # --- 3. Hazard Classes (STEP 3) ---
    console.rule("[bold #6495ED]==== STEP 3: HIGH-LEVEL HAZARD CLASSES (extract_hazards) ====")
    try:
        if hazards and isinstance(hazards, list):
            #
            hazard_str = f"[bold yellow]{', '.join(hazards)}[/bold yellow]"
            console.print(Panel(
                hazard_str,
                title="[bold yellow]Applicable Hazard Categories[/bold yellow]",
                border_style="yellow"))
        else:
            console.print("No hazards identified or data format is incorrect.")
    except Exception:
        console.print("[bold red]ERROR: Could not process Hazard Classes data. Check input format.[/bold red]")

    # --- 4. Consolidated Harm Summary (STEP 4) ---
    console.rule("[bold #6495ED]==== STEP 4: CONSOLIDATED UNIQUE HARMS (harms_summary) ====")
    if harms_summary_list:
        harm_list_formatted = []
        for idx, harm in enumerate(harms_summary_list):
            harm_list_formatted.append(f"[bold red]{idx + 1}.[/bold red] {harm}")

        console.print(Panel(
            "\n".join(harm_list_formatted),
            title="[bold red]Unique Harm Events (Post-Clustering)[/bold red]",
            border_style="red"))
    else:
        console.print("No unique harms identified.")

    # --- 5. Impact Matrix (STEP 5) ---
    console.rule("[bold #6495ED]==== STEP 5: IMPACT MATRIX (extract_iclasses / define_impact) ====")
    impact_table = Table(
        title="Impact Classes & Associated Physical Values",
        show_header=True,
        header_style="bold green"
    )
    impact_table.add_column("Impact Class", style="#ADFF2F", justify="left")
    impact_table.add_column("Associated Quantifiable Physical Values", style="dim", justify="left")

    for ic, impact_list in impacts_dict.items():
        all_pvs = set()
        if impact_list and isinstance(impact_list, list):
            # Aggregate unique physical values across all harms for this Impact Class
            for impact_dict in impact_list:
                if 'physical_value' in impact_dict and isinstance(impact_dict['physical_value'], list):
                    # Filter out 'N/A' responses if they exist
                    for pv in impact_dict['physical_value']:
                        if pv and 'n/a' not in pv.lower():
                            all_pvs.add(pv)

            # Sort and display a maximum of 4 key values
            sorted_pvs = sorted(list(all_pvs))
            value_str = ", ".join(sorted_pvs[:4])
            if len(sorted_pvs) > 4:
                value_str += ", [i]... more[/i]"
        else:
            value_str = "[i]No relevant physical values extracted.[/i]"

        impact_table.add_row(ic, value_str or "[i]None identified[/i]")

    console.print(impact_table)

    console.rule("[bold #6495ED]==== STEP 6: IDENTIFIED FAILURE MODES ====")
    try:
        failure_modes = failure_modes_list
        if failure_modes:
            failure_modes_table = Table(title="Failure Modes of the System", show_header=True, header_style="bold red")
            failure_modes_table.add_column("Failure Mode", style="#6A0C0C", justify="left")
            failure_modes_table.add_column("Description", style="dim", justify="left")

            for m in failure_modes:
                failure_modes_table.add_row(m.get('failure_mode', 'N/A'), m.get('description', 'N/A'))

            console.print(failure_modes_table)
        else:
            console.print("No failure modes identified error.")
    except Exception:
        console.print("[bold red]ERROR: Could not process failure modes data. Check input format.[/bold red]")

    console.rule("[bold #6495ED]==== STEP 7: ASSIGNED ACTUATORS ====")
    try:
        actuators = actuators_list
        if actuators:
            actuators_table = Table(title="Actuators Associated with Impact Classes", show_header=True,
                                    header_style="bold green")
            actuators_table.add_column("Impact classes", style="#26AF00", justify="left")
            actuators_table.add_column("Actuators", style="dim", justify="left")

            for a in actuators:
                actuators_table.add_row(a.get('impact_class', 'N/A'), ", ".join(a.get('actuators', [])))

            console.print(actuators_table)
        else:
            console.print("No actuators identified error.")
    except Exception:
        console.print("[bold red]ERROR: Could not process actuators data. Check input format.[/bold red]")

    # Note: Call display_actuator_failure_summary next in your main block
    # console.rule("[bold #6495ED]==== HARA ANALYSIS COMPLETE ====")


if __name__ == "__main__":
    system = extract_system(
        "A small, six-axis collaborative robot, designed to work alongside human assembly workers on a shared workbench. The Robot's primary task is to pick up small electronic components and accurately place them into circuit boards. It moves slowly, with a maximum payload of 1kg and a speed of 0.5m/s",
        model="openai:gpt-4o")
    persons = extract_persons(system, model="openai:gpt-5.2")
    hazards = extract_hazards(system, model="openai:gpt-5.2")
    print(hazards)
    harms_dict = harms(system, persons, hazards, model="openai:gpt-5.2")
    harms_summary_list = harms_summary(harms_dict, model="openai:gpt-5.2")
    impact_classes = extract_iclasses(system, model="openai:gpt-5.2")
    impacts_dict = impacts(system, impact_classes, harms_summary_list, model="openai:gpt-5.2")
    failure_modes = identify_failure_modes(system, model="openai:gpt-5.2")
    actuators = define_actuators(system, impact_classes, model="openai:gpt-5.2")
    # function extract_failure and collect_failures
    # failures = collect_failures(system, failure_modes, actuators, impacts_dict, model="openai:gpt-5.2")
    failures = {}
    display_hara_summary(system, persons, hazards, harms_summary_list, impacts_dict, failure_modes, actuators, failures)
