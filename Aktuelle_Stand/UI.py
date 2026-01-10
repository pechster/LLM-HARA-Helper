import HARA as h
import RISK_ASSESSMENT as ra
import FILE_SEARCH as fs
import json
import IEC61508 as iec
import ISO26262 as iso
from rich.console import Console
from concurrent.futures import ThreadPoolExecutor

def feedback(final_data: json, backend, hara_step):
    previous_querys = []
    query_history = []
    while True:
        if backend == "HARA":
            user_query = input(f"\nEnter what you would like to modify about the HARA step: {hara_step} or enter U if "
                               f"you want to trigger the modification.\n")
        else:
            user_query = input(f"\nEnter what you would like to modify about the RISK_ASSESSMENT or enter U if you want"
                               f" to trigger the modification.\n")
        if user_query.lower() == "u":
            break
        elif user_query.lower() == "x":
            return json
        response = fs.query_detection_LLM(user_query, final_data, previous_querys, task_description=backend)
        console = Console()
        console.print(response["type"])
        while response["type"] == "clarification":
            user_query = input("The system seems to be confused about your query could you please refine it:\n" +
                               response["content"] + "\n")
            response = fs.query_detection_LLM(user_query, final_data, previous_querys, "HARA")
        query_history.append(response)
        previous_querys.append(user_query)

    if len(query_history) > 0:
        if hara_step == "Hazard Classes":
            return fs.complete_querys(query_history, final_data, hazards=True)
        else:
            return fs.complete_querys(query_history, final_data)
    else:
        return final_data

def modify_request_cycle(to_modify : json, backend, hara_step):
    while True:
        change_requested = input(f"Would like to modify {hara_step} (y/n)?\n")
        if change_requested.lower() == "y":
            modified = feedback(to_modify, backend, hara_step)
            if hara_step == "System Under Analysis":
                h.display_system(modified)
            elif hara_step == "Persons At Risk":
                h.display_persons(modified)
            elif hara_step == "Hazard Classes":
                h.display_hazards(list(modified.keys()))
            elif hara_step == "Harms Summary":
                h.display_harms(modified)
            elif hara_step == "Impact Classes":
                h.display_impact_classes(modified)
            elif hara_step == "Failure Modes":
                h.display_failure_modes(modified)
            else:
                h.display_actuators(modified)
            return modify_request_cycle(to_modify, backend, hara_step)
        else:
            return to_modify

def person_thread(system):
    return h.extract_persons(system, model="openai:gpt-5.2")


def hazard_thread(system):
    return h.extract_hazards(system, model="openai:gpt-5.2")

def impact_classes_thread(system):
    return h.extract_impact_classes(system, model="openai:gpt-5.2")

def failure_modes_thread(system):
    return h.identify_failure_modes(system, model="openai:gpt-5.2")

def actuators_thread(system, impact_classes):
    return h.define_actuators(system, impact_classes, model="openai:gpt-5.2")

def main():
    system = ("""Electronic Parking Brake Description: The system replaces the traditional mechanical handbrake 
    lever. It utilizes electromechanical actuators to lock the rear wheels, securing the vehicle against rolling 
    away when stationary. Additionally, it provides a secondary emergency braking function while the vehicle 
    is in motion.""")

    system = h.extract_system(system, model="openai:gpt-5.2")
    h.display_system(system)

    with ThreadPoolExecutor() as executor:
        person_future = executor.submit(person_thread, system)
        hazard_future = executor.submit(hazard_thread, system)
        impact_future = executor.submit(hazard_thread, system)
        failure_modes_future = executor.submit(failure_modes_thread, system)
        new_system = modify_request_cycle(system, "HARA", "System Under Analysis")
        persons = person_future.result()
        hazards = hazard_future.result()
        impact_classes = impact_future.result()
        failure_modes = failure_modes_future.result()

    if system != new_system:
        with ThreadPoolExecutor() as executor:
            persons = executor.submit(person_thread, new_system)
            hazards = executor.submit(hazard_thread, new_system)
            impact_classes = executor.submit(hazard_thread, new_system)
            failure_modes = executor.submit(failure_modes_thread, new_system)
        system = new_system

    h.display_persons(persons)
    persons = modify_request_cycle(persons, "HARA", "Persons At Risk")
    h.display_hazards(hazards)
    hazards = modify_request_cycle(hazards, "HARA", "Hazard Classes")

    harms_dict = h.harms(system, persons, hazards, model="openai:gpt-5.2")
    harms_summary_list = h.harms_summary(harms_dict, model="openai:gpt-5.2")
    h.display_harms(harms_summary_list)
    harms_summary_list = modify_request_cycle(harms_summary_list, "HARA", "Harms Summary")

    impacts_dict = h.impacts(system, impact_classes, harms_summary_list, model="openai:gpt-5.2")
    h.display_impacts(impacts_dict)
    impact_dict = modify_request_cycle(impacts_dict, "HARA", "Impact Classes")

    with ThreadPoolExecutor() as executor:
        actuators_future = executor.submit(actuators_thread, system, impact_classes)

    h.display_failure_modes(failure_modes)
    failure_modes = modify_request_cycle(failure_modes, "HARA", "Failure Modes")

    h.display_actuators(actuators_future.result())
    actuators = h.define_actuators(system, impact_classes, model="openai:gpt-5.2")
    actuators = modify_request_cycle(actuators, "HARA", "Actuators")

    final_hara = {}
    final_hara["System Under Analysis"] = system
    final_hara["Persons At Risk"] = persons
    final_hara["Hazards"] = hazards
    final_hara["Harms Summary"] = harms_summary_list
    final_hara["Impact"] = impacts_dict
    final_hara["Failure Modes"] = failure_modes
    final_hara["Actuators"] = actuators
    fs.save_file(final_hara, "FINAL_HARA.json")
    print("Saved to HARA!\n")

    standard = ra.identify_standard_prompt(system, model="openai:gpt-5.2")
    if standard["standard_reference"] == "IEC 61508":
        print("IEC 61508")
        final_risk_assessment = iec.run_risk_assessment(harms_summary_list, system)
    elif standard["standard_reference"] == "ISO 26262":
        print("ISO 26262")
        final_risk_assessment = iso.run_risk_assessment(harms_summary_list, model="openai:gpt-5.2")
    else:
        print(standard["standard_reference"])
        final_risk_assessment = ra.run_risk_assessment(system, harms_summary_list, model="openai:gpt-5.2")

    print("\n======== AUTOMATICALLY GENERATED RISK ASSESSMENT ========\n")
    print(json.dumps(final_risk_assessment, indent=4))
    final_risk_assessment = feedback(final_risk_assessment, "RISK", "")
    print("\n======== RISK ASSESSMENT AFTER PROCESSING THE USERS FEEDBACK  ========\n")
    print(json.dumps(final_risk_assessment, indent=4))

    fs.save_file(final_risk_assessment, "RISK_ASSESSMENT.json")
    print("Saved to risk assessment!\n")


if __name__ == "__main__":
    main()

