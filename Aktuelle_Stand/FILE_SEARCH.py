import aisuite as ai
from dotenv import load_dotenv
import json
import os
from HELPERS import *

_ = load_dotenv()
client = ai.Client()


# Identify the request type and content
def query_detection_LLM(user_query: str, anaylsis: json, previous_queries: list, task_description: str,
                        model = "openai:gpt-5.2"):
    if task_description == "HARA":
        task_description = \
            "You are an expert in HARA analysis who has to review a HARA analysis and execute queries on it."
    elif task_description == "RISK":
        task_description = "You are an expert in functional safety engineering and execute queries on it."
    messages = [
        {
            "role": "system",
            "content": f"""
            TASK:
            - {task_description}
            - Classify the user's intent as: delete, get, post, refactor or clarification.
            - The request is referring to the following JSON object: {anaylsis}
            - The conversation history with the past user querys: 
              {previous_queries if len(previous_queries) > 0 else "No previous queries"}
            
            Definitions:
            - post = add or generate new info
            - get = retrieve or view info
            - delete = remove excess info
            - refactor = correct, improve, or update existing info
            - clarification = the request is still unclear despite conversation history
            
            Rules:
            - If the users query can not be used to produce a clear task classify it as clarification
            - In general prefer clarification over every type if the task is not hundret procent clear
            
            JSON FORMAT: 
            {{"type": "request lable", "content": "questions for clarification in case something is unclear 
            precise short description of the task"}}"""
        }
    ]
    messages.append({"role": "user", "content": user_query})
    return run_chat(messages, model, "json")


# Full replacement function, unfinished
def complete_querys(user_querys: list[dict], analysis: json, hazards=False, model="openai:gpt-5.2"):
    if hazards:
        format_requirements = f"""JSON FORMAT:
                                  - Return a JSON containing the hazards as keys and values for each hazard"""
    else:
        format_requirements = f"""JSON FORMAT:
                                  - Keep the exact style of the JSON
                                  - Only return one valid JSON object"""
    messages = [
        {"role": "system",
         "content": f"""
         Task: 
         - You get a list of user querys which you have to execute on this JSON object: {analysis}
         
         Definitions:
         - post = add or generate new info to everything which is affected by the content description
         - delete = remove everything which is affected by the content description while maintaining every information
           which is related to something not mentioned in the content description
         - refactor = correct, improve, or update existing info depending on the content description and substitute the 
           old information through the upgraded version
         - clarification = can be skipped
            
         {format_requirements}
          
        """}
    ]
    messages.append({"role": "user", "content": f"""{user_querys}"""})
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
