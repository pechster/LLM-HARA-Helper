import aisuite as ai
from dotenv import load_dotenv

client = ai.Client()

keywords = {}

def run_chat(model: str, messages: list, **kwargs):
    resp = client.chat.completions.create(model=model, messages=messages, **kwargs)
    return resp.choices[0].message.content


def extract_system(user_prompt: str):
    messages = [
        {"role": "system", "content": "You are an expert in HARA (Hazard Analysis and Risk Assessment) for robotic systems. Extract and formulate a clear system description from the user's prompt to guide the HARA analysis."},
        {"role": "user", "content": f'The given user prompt is "{user_prompt}". Extract the system description relevant for HARA analysis. Please only give a concise system description without any additional commentary.'}
    ]
    keywords["system"] = run_chat("openai:gpt-4o-mini", messages)


def person_at_risk(user_prompt:str, system:str):
    messages = [
        {"role": "system", "content": "You are an expert in HARA (Hazard Analysis and Risk Assessment) for robotic systems. Identify and list all potential persons at risk based on the user's prompt and system description."},
        {"role": "user", "content": f'The given user prompt is "{user_prompt}" and the system description is "{system}". Identify all potential persons at risk relevant for HARA analysis. Please only give a concise list of persons at risk without any additional commentary.'}
    ]

    keywords["persons_at_risk"] = run_chat("openai:gpt-4o-mini", messages)


def identify_hazards(user_prompt:str, system:str):
    messages = [
        {"role": "system", "content": "You are an expert in HARA (Hazard Analysis and Risk Assessment) for robotic systems. Identify and list all potential hazards based on the user's prompt, system description, and persons at risk."},
        {"role": "user", "content": f'The given user prompt is "{user_prompt}". Identify all potential hazards relevant for HARA analysis. Please only give a concise list of hazards without any additional commentary.'}
    ]

    keywords["hazards"] = run_chat("openai:gpt-4o-mini", messages)
   


def harm_caused(user_prompt:str, system:str, persons_at_risk:str, hazards:str):
    # How could {system} potentially cause harm to {persons_at_risk} through {hazards}?

    messages = [
        {"role": "system", "content": "You are an expert in HARA (Hazard Analysis and Risk Assessment) for robotic systems. Analyze and describe how the identified hazards could potentially cause harm to the persons at risk based on the user's prompt and system description."},  
        {"role": "user", "content": f'The given user prompt is "{user_prompt}", the system description is "{system}", the persons at risk are "{persons_at_risk}", and the hazards are "{hazards}". Analyze how the hazards could potentially cause harm to the persons at risk relevant for HARA analysis. Please only give a concise list of potential harms without any additional commentary.'}

    ]

    keywords["harm_caused"] = run_chat("openai:gpt-4o-mini", messages)


def properties_physicalvalue(user_prompt:str, system:str, harms:str):


    # Loop through each harm and determine relevant properties and physical values that could cause the harm

    return None





