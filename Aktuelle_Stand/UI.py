import HARA as h
import RISK_ASSESSMENT as ra
import FILE_SEARCH as fs
import json
import IEC61508 as iec
import ISO26262 as iso


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
        response = fs.query_detection_LLM(user_query, final_data, previous_querys, task_description=backend)
        print(response["type"])
        while response["type"] == "clarification":
            user_query = input("The system seems to be confused about your query could you please refine it:\n" +
                               response["content"] + "\n")
            response = fs.query_detection_LLM(user_query, final_data, previous_querys, "HARA")
        query_history.append(response)
        previous_querys.append(user_query)

    if len(query_history) > 0:
        if hara_step == "HAZARD CLASSES":
            return fs.complete_querys(query_history, final_data, hazards=True)
        else:
            return fs.complete_querys(query_history, final_data)
    else:
        return final_data

def main():
    system = ("""Electronic Parking Brake Description: The system replaces the traditional mechanical handbrake 
    lever. It utilizes electromechanical actuators to lock the rear wheels, securing the vehicle against rolling 
    away when stationary. Additionally, it provides a secondary emergency braking function while the vehicle 
    is in motion.""")
    # user_input = "Unintended Acceleration on Highway"
    # user_input = "Emergency shutdown system in chemical plant"
    #models_to_test = [
    #    "openai:gpt-4o-mini",
    #    "anthropic:claude-sonnet-4-20250514"
    #]

    #hara_buffer = []
    #for model in models_to_test:
    #    result = h.run_single_hara(user_input, model)
    #    hara_buffer.append(result)
    system = h.extract_system(system, model="openai:gpt-5.2")
    h.display_system(system)
    persons = h.extract_persons(system, model="openai:gpt-5.2")
    h.display_persons(persons)
    hazards = h.extract_hazards(system, model="openai:gpt-5.2")
    h.display_hazards(hazards)
    harms_dict = h.harms(system, persons, hazards, model="openai:gpt-5.2")
    harms_summary_list = h.harms_summary(harms_dict, model="openai:gpt-5.2")
    h.display_harms(harms_summary_list)
    impact_classes = h.extract_iclasses(system, model="openai:gpt-5.2")
    impacts_dict = h.impacts(system, impact_classes, harms_summary_list, model="openai:gpt-5.2")
    h.display_impacts(impacts_dict)
    failure_modes = h.identify_failure_modes(system, model="openai:gpt-5.2")
    h.display_failure_modes(failure_modes)
    actuators = h.define_actuators(system, impact_classes, model="openai:gpt-5.2")
    h.display_actuators(actuators)

    changes = input("You have the option to modify the results of the HARA analysis. Enter S if you want to modify\n"
                    "the system description, P if you want to modify the persons, H if you want to modify the\n"
                    "hazards. If any of those three will be modified then the harms summary will be modified\n"
                    "automatically. Altough you have the option to modify the impact classes by entering I,\n"
                    "the failure modes by entering F and the actuators by entering A. Just type in the letters\n"
                    "of all steps you would like to modify. e.g: I, A, M\n")

    letters = [c.strip().lower() for c in changes.split(",")] #Fehlerbehandlung
    final_hara = {}
    if "s" in letters:
        system = feedback(system, "HARA", "System Under Analysis")
        h.display_system(system)
    if "p" in letters:
        persons = feedback(persons, "HARA", "Persons At Risk")
        h.display_persons(persons)
    if "h" in letters:
        hazards = feedback(hazards, "HARA", "Hazard Classes")
        hazards = list(hazards.keys())
        h.display_hazards(hazards)
    if "s" in letters or "p" in letters or "h" in letters:
        harms_dict = h.harms(system, persons, hazards, model="openai:gpt-5.2")
        harms_summary_list = h.harms_summary(harms_dict, model="openai:gpt-5.2")
        h.display_harms(harms_summary_list)
    if "i" in letters:
        impacts_dict = feedback(impacts_dict, "HARA", "Impact Classes")
        h.display_impacts(impacts_dict)
    if "f" in letters:
        failure_modes = feedback(failure_modes, "HARA", "Failure Modes")
        h.display_failure_modes(failure_modes)
    if "a" in letters:
        actuators = feedback(actuators, "HARA", "Actuators")
        h.display_actuators(actuators)

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
