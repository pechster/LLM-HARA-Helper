import aisuite as ai
from dotenv import load_dotenv
import json
import os
from HELPERS import *

_ = load_dotenv()
client = ai.Client()


# Identify the request type and content
def query_detection_LLM(user_query: str, anaylsis: json, previous_queries: list, task_description: str,
                        model: str = "openai:gpt-4o-mini"):
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
            - If the user provides statements which can not be used together with the JSON and the previous queries to 
              map on a valid request type map to clarification.
            - Otherwise map to the request definition which fits the most
            
            JSON FORMAT: 
            {{"type": "request lable", "content": "questions for clarification in case something is unclear 
            precise short description of the task"}}"""
        }
    ]
    messages.append({"role": "user", "content": user_query})
    return run_chat(messages, model, "json")


# implement get query

# Full replacement function, unfinished
def complete_querys(user_querys: list[dict], analysis: json, model="openai:gpt-4o-mini"):
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
            
         JSON FORMAT:
         - Keep the exact style of the analyis
         - Only return one valid JSON object
          
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
            
