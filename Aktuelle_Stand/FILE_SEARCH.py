import aisuite as ai
from dotenv import load_dotenv
import json
import os
from HELPERS import *

_ = load_dotenv()
client = ai.Client()
hara_data = {
        "model_used": "",
        "system": [],
        "persons": ['Operator', 'Bystander', 'Warehouse Workers', 'Maintenance Personnel', 'Forklift Drivers',
                'Supervisors', 'Loading/Unloading Staff'],
        "hazards": [],
        "harms_analysis": [],
        "scenarios": []
    }


# Identify the request type and content
def query_detection_LLM(user_query: str, previous_queries: list, model: str = "openai:gpt-4o-mini"):
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
            - Refactor REQUIRES a clear identifiable target (explicit or inferred by context). If the target is vague, missing, or only implied (e.g., 'it', 'that', 'this') -> clarification.
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
    with open(path, "r", encoding="utf-8") as fd:
        return json.load(fd)


# Save the current information into the JSON file
def save_file(hara_data: dict, file_name: str):
    path = os.path.join(os.path.dirname(__file__), file_name)
    with open(path, "w", encoding="utf-8") as fd:
        json.dump(hara_data, fd)


# Match the field in the JSON file to examine
def identify_request_target(query_content: str, previous_queries: list, model="openai:gpt-4o-mini"):
    messages = [
        {
            "role": "system",
            "content": f"""Choose the most semantically relevant item targeted by the user from the list given to you.
                           Rules:
                           - ALWAYS return one of the fields. 
                           - Use semantic similarity, synonyms, roles, entities and conceptual matching.
                           - Put greater weight on entity-type matches rather than abstract ones.
                           - Always pick the closest field semanticallys
                           - If multiple fields are possible, choose the most specific one.
                           - Make use of the conversation history if nessecary to infer the field.
                           Return JSON: {{"field" : "<field from the list>"}}
                           Possible fields:
                           {list(hara_data.keys())}
                           Conversation history:
                           {previous_queries}"""
        },
        {"role": "user", "content": query_content}
    ]
    return run_chat(messages, "openai:gpt-4o-mini", "json")


# Extract information that is targeted by the user
def collect_exact_data(query_content: str, field: str, previous_queries: list):
    if field is None:
        return {'data': 'Tell the user such data does not exist'}
    field_data = hara_data[field]
    messages = [
        {"role": "system", 
         "content": f"""Perform semantic entity matching on the user request with the data provided.    
          Rules:
          - Treat each item in the data as a semantic concept, not a literal string.
          - Determine which items match the implication of the user request best.
          - You are allowed to use world knowledge to interpret what each entity label generally represents.
          - ONLY choose items from the provided data when they clearly satisfy the user intent.
          - Make use of the conversation history if nessecary to infer the field.
          - If there are no matches, return []
          Return JSON: {{"data": "<all matching data>"}}
          Data to match with: 
          {field_data}
          Conversation history:
         {previous_queries}"""
        },
        {"role": "user", "content": f"""{query_content}"""}
    ]
    return run_chat(messages, "openai:gpt-4o-mini", "json")


# Full replacement function, unfinished
def complete_query(user_query: str, file_name: str, max_history: int = 10):
    hara_data = load_file(file_name)
    query_type = query_detection_LLM(user_query, [])
    request = query_type["type"]
    query_content = query_type["content"]

    if request == "clarification":
        return {"error": "User needs to provide clarification"}
    if request == "post":
        information_to_be_added = [] # need a way to work with user queries
        hara_data[field].append(information_to_be_added)

    label_data_affected = identify_request_target(query_content, [])
    field = label_data_affected["field"]
    print(field)
    data_affected = collect_exact_data(query_content, field, [])

    if request == "get":
        print("get")
        print(data_affected)
        return data_affected["data"]
    
    elif request == "delete":
        hara_data[field] = [datapoint for datapoint in hara_data[field] if datapoint not in data_affected["data"]]
        print(f"Deleted:{data_affected['data']}")
        save_file(hara_data, file_name)
        return data_affected["data"]
    
    elif request == "refactor":
        return {"data" : "not implemented yet"} # Pass data present to logic from Yin's code to improve
    print("User query wasn't identified correctly")
    return


def main():
    save_file(hara_data, "test.json")
    test_queries = [
        "I want you to delete all persons at risk that don't work at the factory."]
    save_file(hara_data, "test.json")
    for query in test_queries:
        print(complete_query(query, "test.json"))


if __name__ == "__main__":
    main()
