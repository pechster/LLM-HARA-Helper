import HARA as h
import RISK_ASSESSMENT as ra
import FILE_SEARCH as fs
import json


def main():
    user_input = "A mobile robot (AGV) transports heavy pallets in a warehouse shared with human workers. It has a lifting fork mechanism."

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
    print(hazards)
    print("\n======== FINAL JSON DATA ========")
    print(json.dumps(final_data, indent=4))
    fs.save_file(final_data, "FIRST_HARA_ANALYSIS.json")
    print("\nSaved to FIRST_HARA ANALYSIS.json")

    risk_assesment_buffer = []

    for model in models_to_test:
        result = ra.run_risk_assessment(user_input, hazards, model)
        risk_assesment_buffer.append(result)

    final_risk_assessment = ra.synthesize_consensus(risk_assesment_buffer, judge_model="openai:gpt-4o")

    print(json.dumps(result, indent=4))


if __name__ == "main":
    main()