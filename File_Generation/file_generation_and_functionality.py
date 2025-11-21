import aisuite as ai
from dotenv import load_dotenv
import json
import os

_ = load_dotenv()
client = ai.Client()
hara_data = {
    "system": "",
    "persons":  ['Operator', 'Bystander', 'Warehouse Workers', 'Maintenance Personnel', 'Forklift Drivers', 'Supervisors', 'Loading/Unloading Staff'],
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

# Identify the request type and content
def query_detection_LLM(user_query: str, previous_queries: list, max_messages = 10, model: str = "openai:gpt-4o-mini"):
    messages = [
        {"role": "system",
        "content": f"""
            Classify the user's intent as: delete, get, post, refactor, clarification.
            Definitions:
            - post = add or generate new info
            - get = retrieve or view info
            - delete = remove excess info
            - refactor = correct, improve, or update existing info
            - clarification = the request is still unclear despite conversation history
            Rules:
            - Refactor REQUIRES an clear identifiable target (explicit or inferred by context). If the target is vague, missing, or only implied (e.g., 'it', 'that', 'this') -> clarification.
            - Critique of information WITH explicit target -> refactor.
            - Prefer delete/get/post/refactor ONLY when the traget is inferable.
            - If the intent remains ambiguous even after examining history → clarification.
            Return JSON: {{"type": "<label>", "content": "<exact description of the request>"}}
            Conversation history:"""}
    ]
    messages.extend(previous_queries)
    messages.append({"role": "user", "content": user_query})
    return run_chat(messages, model, "json")

# Load the information from the JSON file
def load_file(file_name: str):
    path = os.path.join(os.path.dirname(__file__), file_name)
    with open(path, "r") as fd:
        return json.load(fd)

# Save the current information into the JSON file
def save_file(hara_data: dict, file_name: str):
    path = os.path.join(os.path.dirname(__file__), file_name)
    with open(path, "w") as fd:
        json.dump(hara_data, fd)

def identify_request_target(query_content: str):
    messages = [
        {
         "role" : "system", 
         "content": f"""Identify which field in JSON data is being targeted by the user query. Be semantically flexible and match any possible connections over an adequate confidence level.
          If there is no significant connection between the request and the field, output: null. Respond only in JSON and give no warnings: {{"field" : "<label for data>"}}
          Possible fields:
          {hara_data}"""
        },
        {"role" : "user", "content" : query_content}
    ]
    return run_chat(messages, "openai:gpt-4o-mini", "json")

def collect_exact_data(query_content: str, field: str):
    if field == "None":
        return {'data': 'None'}
    field_data = hara_data[field]
    messages = [
        {"role": "system", "content": f"""Extract all data that semantically relates to the user query with a high level of confidence. Return JSON: {{"data": "<all matching data>"}}
                                          Data to match with: {field_data}"""},
        {"role": "user", "content" : f"""{query_content}"""}
    ]
    return run_chat(messages, "openai:gpt-4o-mini", "json")

def complete_query(user_query: dict, file_name: str):
     hara_data = load_file(file_name)
     query_type = query_detection_LLM(user_query, [])
     request = query_type["type"]
     query_content = query_type["content"]

     if request == "clarification":
         return {"error": "User needs to provide clarification"}
     
     label_data_affected = identify_request_target(query_content)
     data_affected = collect_exact_data(query_content, label_data_affected["field"])

     if request == "post":
         return {"data" : "not yet implemented"}
     elif request == "get":
         return data_affected["data"]
     elif request == "delete":
         return {"data" : "not yet implemented"}
     elif request == "refactor":
         return # Pass data present to logic from Yin's code to improve
     print("User query wasn't identified correctly")
     return

def main():
    test_queries = [
     "I think we should maybe change it a bit.",
     "Can you show me the list of people affected?",
     "Fetch the data for project 121-B.",
     "What was the last entry you added?",
     "Give me all posts tagged with ‘finance’.",
     "Search for anything mentioning heat exposure."]
    
    for query in test_queries:
        print(identify_request_target(query))
    save_file(hara_data, "name.json")

if __name__ == "__main__":
    main()