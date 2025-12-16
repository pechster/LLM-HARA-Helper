from typing import List, Dict
from HELPERS import *

_ = load_dotenv()
client = ai.Client()

standard_guideline = """
Controllability (C), i.e. the ability to avoid the specific harm or damage through timely reactions of the persons 
involved
C0: Controllable in general, 
C1: Simply controllable / More than 99 % of the average operating person or other personnel are able to avoid harm, 
C2: Normally controllable / Between 90 % an 99 % of the average operating person or other personnel are able to avoid 
harm, 
C3: Difficult to control or uncontrollable / Less than 90 % of the average operating person or other personnel are able 
to avoid harm
NOTE: IF the percentage was unclear / vague to calculate use the normal terms stated first.

NOTE:
- The evaluation of controllability assesses the likelihood that an individual can gain sufficient control over a 
hazardous event to prevent the specific harm from occurring. It is assumed that the driver is in an appropriate 
condition to operate the vehicle, has received proper driver training, and is adhering to relevant legal regulations, 
including necessary precautions to protect other road users.
- If the hazardous event is not related to controlling the vehicle's direction or speed, controllability can be seen as 
an estimation of the likelihood that the person at risk can either remove themselves or be removed by others from the 
dangerous situation.
- When considering controllability, note that the person at risk might not be familiar with the operation of the item or 
may not even realize that a hazardous situation is developing.
-----------------------------------------------------------------------------------------------
The potential severity (S) of the resulting harm or damage:
S0:No injuries, S1:Light and moderate injuries Class, S2:Severe and life-threatening injuries(survival probable), 
S3:Life-threatening injuries (survival uncertain) / fatal injuries

NOTES:
- S0 is also assigned if the hazard analysis and risk assessment determines that the consequences are clearly limited to
 material damage, where the damage only occurs to the machinery. 
- To describe the severity, the AIS classification is used.
		AIS 0: No injuries.
		AIS 1: Minor injuries, including superficial cuts, muscle soreness, whiplash, etc.
		AIS 2: Moderate injuries such as deep lacerations, concussions with up to 15 minutes of unconsciousness, 
		uncomplicated fractures of long bones or ribs, etc.
		AIS 3: Severe, non-life-threatening injuries, including skull fractures without brain damage, spinal 
		dislocations below the fourth cervical vertebra without spinal cord injury, multiple rib fractures without 
		paradoxical breathing, etc.
		AIS 4: Serious injuries (life-threatening, but survival is likely) such as concussions with or without skull 
		fractures, resulting in up to 12 hours of unconsciousness, or 	paradoxical breathing.
		AIS 5: Critical injuries (life-threatening, uncertain survival), including spinal fractures below the fourth 
		cervical vertebra with spinal cord damage, intestinal or cardiac tears, unconsciousness lasting more than 
		12 hours, or intracranial bleeding.
		AIS 6: Extremely critical or fatal injuries, such as fractures of the cervical vertebrae above the third 
		vertebra with spinal cord damage, or severe open wounds in the thoracic or abdominal cavities.
WHERE this is the correlation between the S table and the AIS table:
 S0 -> AIS 0 and less than 10 % probability of AIS 1-6; or damage that cannot be classified safety-related
 S1 -> More than 10 % probability of AIS 1-6 (and not S2 or S3)
 S2 -> More than 10 % probability of AIS 3-6 (and not S3)
 S3 -> More than 10 % probability of AIS 5-6
-----------------------------------------------------------------------------------------------
Exposure (E) probability of the operational situation taking place in which the hazardous event can occur
E0:Incredible, E1:Very low probability, E2: Low probability, E3:Medium probability, E4:High probability

NOTES:
- The number of vehicles equipped with the item shall not be considered when estimating the probability of exposure, 
assuming each vehicle is equipped with the item. This means that the argument “the probability of exposure can be 
reduced, because the item is not present in every vehicle (as only some vehicles are equipped with the item)” is not 
valid.
- A rationale shall be recorded for the exclusion of the incredible situations.
	Typical examples of E0 include the following:
 	a) a very unusual, or infeasible, co-occurrence of circumstances, e.g. a vehicle involved in an incident which 
 	includes an aeroplane landing on a highway; and
 	b) natural disasters, e.g. earthquake, hurricane, forest fire.
- The remaining E1, E2, E3 and E4 levels are assigned for situations that can become hazardous depending on either the 
duration of a situation (temporal overlap) or the frequency of occurrence of a situation. In the first case the 
probability of exposure is typically estimated by the proportion of time spent in the considered situation compared to 
the total operating time, Note that in some cases the total operating time can be the vehicle life-time
-  If the time period in which a failure remains latent is comparable to the time period before the hazardous event can 
be expected to take place, then the estimation of the probability of exposure considers that time period. Typically this
will concern devices that are expected to act on demand, e.g. airbags. => In this case, the probability of exposure 
is estimated by σ × T where σ is the rate of occurrence of the operational situation and T is the duration during which 
the failure is not perceived (possibly up to the lifetime of the vehicle). 
Where if (the % of average operating time) was available then normally this is the correlation between it and the table 
of E:
 E1 -> Not specified
 E2 -> <1 % of average operating time
 E3 -> 1 % to 10 % of average operating time
 E4 -> >10 % of average operating time
Where if (Frequency of situation) was available then normally this is the correlation between it and the table of E:
 E1 -> Occurs less often than once a year for the great majority of personnel
 E2 -> Occurs a few times a year for the great majority of personnel
 E3 -> Occurs once a month or more often for an average personnel
 E4 -> Occurs during almost every drive on average
-----------------------------------------------------------------------------------------------
NOTES:
- If classification of a given hazard with respect to severity (S), probability of exposure (E) or controllability (C) 
is difficult to make, it is classified conservatively, i.e. whenever there is a reasonable doubt, a higher S, E or C 
classification is chosen.

"""
"""standard_integrity_guidline = 
- Four ASILs are defined: ASIL A, ASIL B, ASIL C and ASIL D, where ASIL A is the lowest safety integrity level and ASIL 
D the highest one.
- In addition to these four ASILs, the class QM  indicates that quality processes are sufficient to manage the 
identified risk.
- If a hazardous event is assigned controllability class C0, no ASIL assignment is required.
- If a hazardous event is assigned exposure class E0, no ASIL assignment is required.
- If a hazardous event is assigned severity class S0, no ASIL assignment is required.
"""

# Matrix = (S, E, C)
ASIL_MATRIX = {
    (1, 1, 1): "QM", (1, 1, 2): "QM", (1, 1, 3): "QM",
    (1, 2, 1): "QM", (1, 2, 2): "QM", (1, 2, 3): "QM",
    (1, 3, 1): "QM", (1, 3, 2): "QM", (1, 3, 3): "A",
    (1, 4, 1): "QM", (1, 4, 2): "A", (1, 4, 3): "B",
    (2, 1, 1): "QM", (2, 1, 2): "QM", (2, 1, 3): "QM",
    (2, 2, 1): "QM", (2, 2, 2): "QM", (2, 2, 3): "A",
    (2, 3, 1): "QM", (2, 3, 2): "A", (2, 3, 3): "B",
    (2, 4, 1): "A", (2, 4, 2): "B", (2, 4, 3): "C",
    (3, 1, 1): "QM", (3, 1, 2): "QM", (3, 1, 3): "A",
    (3, 2, 1): "QM", (3, 2, 2): "A", (3, 2, 3): "B",
    (3, 3, 1): "A", (3, 3, 2): "B", (3, 3, 3): "C",
    (3, 4, 1): "B", (3, 4, 2): "C", (3, 4, 3): "D",
}


def evaluate_hazards(hazards: List[Dict], model="openai:gpt-4o-mini") -> List[Dict]:
    results = []
    for idx, hazard in enumerate(hazards, start=1):
        messages = [
            {"role": "system", "content":
            f"""You are an expert functional safety engineer familiar with the ISO 26262 standard and HARA analysis.
                
            TASK:
            - Follow this guideline: {standard_guideline}.\n
            - Each object should include the assigned value and rationale for each risk parameter.
            - If the necessary information can not be derived from the guidance, then mark the hazards risk as 
              unknown
                
                
            OUTPUT REQUIREMENTS:
            - Respond only with one valid JSON object
            - Make use of the predifined format

            JSON FORMAT:
            {{
            "hazard": "{hazard}"
            "Severity": " {{
                "value" : "S0, S1, S2, S3, UNKNOWN"
                "reason" : "short explanation why this value is assigned"
            }}, 
            "Exposure": " {{
                "value" : "E0, E1, E2, E3, E4, UNKNOWN"
                "reason" : "short explanation what explains the frequency of the occurrence"
            }},
            "Controllability": " {{
                "value" : "C0, C1, C2, C3, UNKNOWN"
                "reason" : "short explanation what could possibly avoid the occurrence"
            }}
            }}"""},
            {"role": "user", "content":
                f"Hazard scenario: {hazard}\n"
             }]
        response = run_chat(messages=messages, model=model, expected_format=json)
        results.append(response)
    return results


def ASIL_assessment(hazards: List[Dict]) -> List[Dict]:
    for idx, hazard in enumerate(hazards):
        if not isinstance(hazard, dict):
            hazard = json.loads(hazard)
        print(json.dumps(hazard, indent=4))
        s = hazard["Severity"]["value"]
        e = hazard["Exposure"]["value"]
        c = hazard["Controllability"]["value"]
        if s != "UNKNOWN" and e != "UNKNOWN" and c != "UNKNOWN":
            s = int(s[1:])
            e = int(e[1:])
            c = int(c[1:])
            if (s * e * c) == 0:
                hazards[idx]["ASIL"] = "-"
            else:
                hazards[idx]["ASIL"] = ASIL_MATRIX[(s, e, c)]
        else:
            hazards[idx]["ASIL"] = "UNKNOWN"
    return hazards


def extract_json(block):
    cleaned = re.sub(r"^```json|```$", "", block.strip(), flags=re.MULTILINE).strip()
    return json.loads(cleaned)


def run_risk_assessment(hazards: List[dict], model: str = "openai:gpt-4o-mini") -> List[dict]:
    result = evaluate_hazards(hazards=hazards, model=model)
    result = [extract_json(item) for item in result]
    result = ASIL_assessment(result)
    return result
	
