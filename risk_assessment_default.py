import itertools
from typing import List, Dict
import aisuite as ai
from dotenv import load_dotenv
import json
import re
_ = load_dotenv()
client = ai.Client()

risk_parameters = """
C = "Severity / Consequence [C1: no injury, C2: minor injury, C3: major injury, C4: fatal injury]"
F = "frequency of exposure [F1: rare exposure, F2: medium exposure, F3: regular exposure]"
P = "possibility of avoiding the hazard [P1: possible >= 10%, P2: impossible < 10%]"
W = "probability that external measures mitigate the hazard [W1, W2, W2]"
"""
"""
Injury_Data_Info = 
                Industry: 
                Total Number of workers:
                Number of minor injuries per year:
                Number of major injuries per year:
                Number of fatal injuries per year:
"""
Injury_Data = ["", 600_000, 31_000, 500, 10]

risk_graph = {}

# Cleans JSON data
def normalize_hazard_data(raw_input):

    # 1) Flatten nested structure (handles [[{...}]] etc.)
    def flatten(items):
        out = []
        for x in items:
            if isinstance(x, list):
                out.extend(flatten(x))
            else:
                out.append(x)
        return out

    hazards = flatten(raw_input)

    # 2) Extract parameters C/F/P/W 
    param_regex = {
        "C": re.compile(r"\bC([1-4])\b"),
        "F": re.compile(r"\bF([1-3])\b"),
        "P": re.compile(r"\bP([1-2])\b"),
        "W": re.compile(r"\bW([1-3])\b"),
    }

    def extract_param(entry_dict, param):
        text = str(entry_dict)
        match = param_regex[param].search(text)
        if match:
            return f"{param}{match.group(1)}"
        return "?"

    # 3) Build output
    clean_list = []

    for entry in hazards:
        if not isinstance(entry, dict):
            continue

        idx = entry.get("idx", "?")
        hz = entry.get("Hazard", "UNKNOWN HAZARD")

        clean_list.append({
            "idx": idx,
            "Hazard": hz,
            "C": extract_param(entry, "C"),
            "F": extract_param(entry, "F"),
            "P": extract_param(entry, "P"),
            "W": extract_param(entry, "W")
        })

    return clean_list

# Helper to create chat prompt message dictionaries
def extract_json_array(text):
    # Find first JSON array; allows for multiline spanning brackets
    match = re.search(r"(\[[\s\S]*?\])", text)
    if match:
        return match.group(1)
    return None

def run_chat(messages, model="openai:gpt-4o", expected_format="text"):
    response = client.chat.completions.create(model=model, messages=messages)
    content = response.choices[0].message.content

    if expected_format == "json":
        clean_content = content.replace("``````", "").strip()
        try:
            return json.loads(clean_content)
        except json.JSONDecodeError:
            extracted = extract_json_array(clean_content)
            if extracted:
                try:
                    return json.loads(extracted)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse even after extraction. Raw: {extracted[:   50]}...")
                    return []
            print(f"Warning: Could not find JSON array. Raw: {clean_content[:100]}...")
            return []
    return content

# Send an LLM call to determine the Injury Numbers for the given system's Industry
# For more Accuracy input the Country of the Industry in the system_description
# Or use the pdf assumed values [600_000, 31_000, 500, 10]
def get_injury_stats(system_description: str, model: str = "openai:gpt-4o") -> List:
    print("--- Calculating Industry Stats")
    messages = [
        {"role": "system", "content": (
            "You are a functional safety expert capable of retrieving sector accident statistics."
            " When given a specific system or industry description, you return only the key annual injury statistics as structured data."
        )},
        {"role": "user", "content": (
            f"System description: {system_description}\n\n"
            "Provide the following as JSON array ONLY with these keys:\n"
            "[{\n"
            "\"Industry\": <industry>,\n"
            "\"Total Number of workers\": <total_workers>,\n"
            "\"Number of minor injuries per year\": <minor_injuries>,\n"
            "\"Number of major injuries per year\": <major_injuries>,\n"
            "\"Number of fatal injuries per year\": <fatal_injuries>\n"
            "}]\n"
            "Do not include ANY explanation, markdown, or extra text."
        )}
    ]
    injury_stats = run_chat(messages, model=model, expected_format="json")
    print(f"inj: {injury_stats}")
    injury_stats = injury_stats[0]
    injury_Data = [
    injury_stats.get("Industry", ""), 
    injury_stats.get("Total Number of workers", 0), 
    injury_stats.get("Number of minor injuries per year", 0),
    injury_stats.get("Number of major injuries per year", 0),
    injury_stats.get("Number of fatal injuries per year", 0)]
    print(f"Data: {injury_Data}")
    return injury_Data

# Calculating the Risk Graph with the PFHACC value using the Injury Data

def calculate_risk_graph(injury_data: List = Injury_Data):
    print("--- Calculating Risk Graph")
    C = {'C2': 2, 'C3': 3, 'C4': 4}
    F = {'F1': 1, 'F2': 2, 'F3': 3}
    P = {'P1': 1, 'P2': 2}
    W = {'W3': 1}

    combinations = list(itertools.product(
        C.values(), 
        F.values(), 
        P.values(),
        W.values(),
    ))
    PFH_minor = 1/((1/(injury_data[2]/injury_data[1]))*8760)
    PFH_major = 1/((1/(injury_data[3]/injury_data[1]))*8760)
    PFH_fatal = 1/((1/(injury_data[4]/injury_data[1]))*8760)
    PFH_minor = int(format(PFH_minor, ".1e").split("e")[1])
    PFH_major = int(format(PFH_major, ".1e").split("e")[1])
    #PFH_major = -7
    PFH_fatal = int(format(PFH_fatal, ".1e").split("e")[1])
    concatenated_numbers = [int(''.join(map(str, combo))) for combo in combinations]
    for num in concatenated_numbers:
        risk_graph[num] = [0, 0, 0, 0]
        c = int(num/1000)
        match c:
            case 2:
                risk_graph[num][0] = PFH_minor
            case 3:
                risk_graph[num][0] = PFH_major
            case 4:
                risk_graph[num][0] = PFH_fatal
            case _:
                risk_graph[num][0] = 0
        f = (num // 100) % 10
        match f:
            case 1:
                risk_graph[num][1] = risk_graph[num][0] + 2
            case 2:
                risk_graph[num][1] = risk_graph[num][0] + 1
            case 3:
                risk_graph[num][1] = risk_graph[num][0]
        p = (num // 10) % 10
        match p:
            case 1:
                risk_graph[num][2] = risk_graph[num][1] + 1
            case 2:
                risk_graph[num][2] = risk_graph[num][1]
        sil_val = risk_graph[num][2]
        if sil_val == -8:
            risk_graph[num][3] = 4
        elif sil_val == -7:
            risk_graph[num][3] = 3
        elif sil_val == -6:
            risk_graph[num][3] = 2
        elif sil_val == -5:
            risk_graph[num][3] = 1
        elif sil_val <= -9:
            risk_graph[num][3] = 10
        else:
            risk_graph[num][3] = 0

# Send an LLM Call to determine for every HAZARD what the appropriate risk parameters' values they should have.
def risk_parameters_prompt(hazard_list: List[str], standard: str, model: str, parameters: str = risk_parameters) -> List[Dict]:
    print("--- Assigning values to the Risk parameters of every Scenario")
    result = []
    for idx, hazard in enumerate(hazard_list, start=1):
        messages = [
            {"role": "system", "content":
                f"You are an expert safety engineer familiar with the {standard} standard and HARA analysis."
            },
            {"role": "user", "content":
                f"Hazard scenario #{idx}:\n{hazard}\n\n"
                f"Risk parameters to assess: {parameters}.\n"
                "Return results as a JSON array of objects with these keys: idx, Hazard, [risk_parameter1], [risk_parameter2], ... , [risk_parameterN]. "
                "Each object should include the assigned value and rationale for each risk parameter."
            }]
        response = run_chat(messages, model, expected_format="json")
        result.append(response)
    return result

# Using the calculated Risk-parameters-values and the calculated Risk graph, the SIL-value of every HAZARD scenario is determined.

def risk_assessment_prompt(hazard_param_mat: List[dict]) -> List[dict]:
    print("---Given the assigned risk parameters and the risk graph assign the SIL value for every hazard scenario")
    for i, hazard in enumerate(hazard_param_mat):
        paras = {key: hazard[key] for key in ['C', 'F', 'P'] if key in hazard}
        numbers = ''.join([value[1] for value in paras.values()])
        combined_number = int(numbers)
        combined_number = (combined_number*10)+1
        sil = risk_graph[combined_number][3]
        if sil == 10:
            hazard_param_mat[i]["SIL"] = 'P'
        elif sil == 0:
            hazard_param_mat[i]["SIL"] = '-'
        else:
            hazard_param_mat[i]["SIL"] = sil
        hazard_param_mat[i]["W"] = "W3"
    return hazard_param_mat

# The center running method

def run_risk_assessment(hazard_list: List[str], system_description: str, standard: str = "IEC 61508", model: str = "openai:gpt-4o") -> List[dict]:
    print("--- Started Risk Assessment")
    inj_data = get_injury_stats(system_description)
    # calculate_risk_graph(inj_data)
    calculate_risk_graph()
    hazard_paras = risk_parameters_prompt(hazard_list=hazard_list, standard=standard, model=model)
    cleaned_paras = normalize_hazard_data(hazard_paras)
    result = risk_assessment_prompt(cleaned_paras)
    print("Finished Risk Assessment: ")
    return result

if __name__ == "__main__":
    hazards = [
    "Collision between AGV and human worker in narrow aisles",
    "Overturning of AGV due to improperly stacked loads",
    "AGV malfunctioning and blocking the aisle",
    "AGV moving too quickly for human workers to react",
    "Human worker being struck by the AGV while distracted",
    "Failure of AGV's safety sensors due to dust, dirt, or obstruction",
    "Human worker entering restricted areas due to a malfunction of protective doors",
    "AGV crashing into stationary obstacles due to software or programming errors",
    "Human worker dropping items onto an AGV while stacking",
    "AGV's load being unbalanced, causing tipping while navigating corners",
    "AGV moving into public areas when workers are present",
    "Confusion due to mixed pedestrian and AGV traffic",
    "Forklift or other vehicles colliding with the AGV",
    "AGV failing to properly recognize a worker’s presence in a protected area",
    "Electrical fire due to AGV battery malfunction or overheating",
    "Poor communication between AGV and human workers",
    "Workers unaware of AGV's operational zones due to unclear markings",
    "AGV failing to stop in time due to unexpected obstacle in path",
    "Battery depletion causing AGV to stop in critical areas",
    "Human worker being caught between AGV and a fixed object in a narrow aisle"]

    sys_des = """Suppose your company builds an AGV with a maximum weight of 300kg and a 
    maximum speed of 2m/s. Also assume that the items transported by the AGV weigh 
    up to 50kg. The load items are loosely stacked on the AGV. The AGV operates in a 
    mixed environment with human workers and in a warehouse built for humans without 
    special preparation for robot operation. The warehouse has limited access areas 
    where humans can only enter through protected doors, and public areas where any 
    number of humans also work in the warehouse. The warehouse has narrow aisles (as 
    wide as the robot) and wide aisles (3 times as wide as the robot)."""

    res = run_risk_assessment(hazard_list=hazards, system_description=sys_des)
    for thing in res:
        print(thing)

