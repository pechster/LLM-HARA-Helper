import HARA as h
import RISK_ASSESSMENT as ra
import FILE_SEARCH as fs
import json
import IEC61508 as iec
import ISO26262 as iso


def feedback(final_data: json, backend: str):
    previous_querys = []
    query_history = []
    while True:
        user_query = input(f"\nAbove you can review the complete {backend} analysis. If you want to change something "
                           "please enter your request in natural language. In case you have entered everything which "
                           "needs to be changed you can trigger the update process with the letter U.\n")
        if user_query.lower() == "u":
            break
        response = fs.query_detection_LLM(user_query, final_data, previous_querys, task_description=backend)
        while response["type"] == "clarification":
            user_query = input("The system seems to be confused about your query could you please refine it:\n" +
                               response["content"] + "\n")
            response = fs.query_detection_LLM(user_query, final_data, previous_querys, "HARA")
        query_history.append(response)
        previous_querys.append(user_query)

    if len(query_history) > 0:
        return fs.complete_querys(query_history, final_data)


def main():
    # user_input = "A mobile robot (AGV) transports heavy pallets in a warehouse shared with human workers. It has a
    # lifting fork mechanism."
    # user_input = "Unintended Acceleration on Highway"
    user_input = "Emergency shutdown system in chemical plant"
    models_to_test = [
        "openai:gpt-4o-mini",
        "anthropic:claude-sonnet-4-20250514"
    ]

    hara_buffer = []
    for model in models_to_test:
        result = h.run_single_hara(user_input, model)
        hara_buffer.append(result)

    final_data = h.synthesize_consensus(hara_buffer, judge_model="openai:gpt-4o")
    hazards = final_data["verified_hazards"]
    print("\n======== AUTOMATICALLY GENERATED HARA ANALYSIS ========\n")
    print(json.dumps(final_data, indent=4))
    final_data = feedback(final_data, "HARA")
    print("\n======== RISK ASSESSMENT AFTER PROCESSING THE USERS FEEDBACK  ========\n")
    print(json.dumps(final_data, indent=4))

    fs.save_file(final_data, "HARA_ANALYSIS.json")
    print("Saved to HARA analysis!\n")

    standard = ra.identify_standard_prompt(user_input, "openai:gpt-4o")
    if standard["standard_reference"] == "IEC 61508":
        print("IEC 61508")
        final_risk_assessment = iec.run_risk_assessment(hazards, user_input)
    elif standard["standard_reference"] == "ISO 26262":
        print("ISO 26262")
        final_risk_assessment = iso.run_risk_assessment(hazards, "openai:gpt-4o")
    else:
        risk_assesment_buffer = []
        for model in models_to_test:
            result = ra.run_risk_assessment(user_input, hazards, model)
            risk_assesment_buffer.append(result)
        final_risk_assessment = ra.synthesize_consensus(risk_assesment_buffer, judge_model="openai:gpt-4o")

    print("\n======== AUTOMATICALLY GENERATED RISK ASSESSMENT ========\n")
    print(json.dumps(final_risk_assessment, indent=4))
    final_risk_assessment = feedback(final_risk_assessment, "RISK")
    print("\n======== RISK ASSESSMENT AFTER PROCESSING THE USERS FEEDBACK  ========\n")
    print(json.dumps(final_risk_assessment, indent=4))

    fs.save_file(final_risk_assessment, "RISK_ASSESSMENT.json")
    print("Saved to risk assessment!\n")

if __name__ == "__main__":
    main()
