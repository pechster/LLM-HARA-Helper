import re
import aisuite as ai
from dotenv import load_dotenv
import json
import os
import ast
from typing import List, Dict, Any
from HELPERS import *
from rich.console import Console

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


def extract_persons(system:json, model:str="openai:gpt-4o-mini"):
    system_prompt = {
        "role": "system",
        "content": f"""
        You are an expert in Hazard Analysis and Risk Assessment (HARA). 

        TASK: 
        Analyze the given system description that was extracted from the user input and extract all relevant persons that could interact with the system.

        INPUT:
        The system is described as follows: {system}

        OUTPUT FORMAT:
        Return only a valid JSON array where each element has exactly these keys:
        - "name": (string, extremely concise, e.g., "Operator", "Maintenance Technician", "Bystander")
        - "role": (string, a concise summary of their interaction with the system)
        """}
    
    few_shot_user = {
        "role": "user",
        "content": "A cargo drone that carry cargos up to 5kg with a speed of 3m/s."
    }

    few_shot_assistant = {
        "role": "assistant",
        "content": """[
            {"name": "Operator", "role": "Controls and monitors the cargo drone during its operations."},
            {"name": "Maintenance Technician", "role": "Performs regular maintenance and repairs on the cargo drone."},
            {"name": "Bystander", "role": "Individuals in the vicinity who may be affected by the drone's operations."}
        ]"""
    }
    
    response = run_chat_hara(
        messages=[
            system_prompt,
            {
                "role": "user",
                "content": "Identify all relevant persons for the given system."
            }],
            model=model,
            expected_format="json",
            temperature=0.5)
    
    print(response)
    


if __name__ == "__main__":
    extract_system("A mobile robot (AGV) transports heavy pallets in a warehouse shared with human workers. It has a lifting fork mechanism.", model="openai:gpt-4o")