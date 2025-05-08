import requests
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
import re
import os
from flask import Flask, request, jsonify
import json
# import uuid # No longer needed for session IDs
import time
from flask_cors import CORS

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)

# --- Configuration ---
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://ciaiaiservices.openai.azure.com/")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "817dce22f5a548b8b11fe0b6a3cf2c36") # Replace or set env var
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-35-turbo-0613")
AZURE_API_VERSION = "2024-05-01-preview"

# --- Check if Config is Set ---
if AZURE_OPENAI_API_KEY == "YOUR_API_KEY_HERE":
    print("Warning: AZURE_OPENAI_API_KEY is not set via environment variable. Using placeholder.")
if not AZURE_OPENAI_ENDPOINT or not AZURE_DEPLOYMENT_NAME:
     raise ValueError("Azure OpenAI Endpoint and Deployment Name must be configured.")


# --- Re-paste Utility Functions for Clarity ---
def is_valid_nct_format(nct_id):
    if not nct_id or not isinstance(nct_id, str): return False
    return re.fullmatch(r"NCT\d{8}", nct_id.strip().upper()) is not None

def does_nct_id_exist(nct_id):
    if not is_valid_nct_format(nct_id): return False
    url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id.strip().upper()}"
    try: response = requests.get(url, timeout=10); return response.status_code == 200
    except requests.exceptions.RequestException as e: print(f"Exist check error {nct_id}: {e}"); return False

def suggest_nct_ids_by_indication(indication):
    if not indication or not isinstance(indication, str): return []
    query = indication.strip().replace(" ", "+"); url = f"https://clinicaltrials.gov/api/v2/studies?query.cond={query}&pageSize=3"
    try:
        response = requests.get(url, timeout=15); response.raise_for_status(); results = response.json().get("studies", []); suggestions = []
        for r in results:
            if r:
                id_module = r.get("protocolSection", {}).get("identificationModule", {}); nct_id = id_module.get("nctId"); title = id_module.get("briefTitle", "N/A")
                if nct_id: suggestions.append({"nct_id": nct_id, "title": title})
        return suggestions
    except requests.exceptions.RequestException as e: print(f"Suggest error '{indication}': {e}"); return []
    except json.JSONDecodeError as e: print(f"Suggest JSON error '{indication}': {e}"); return []

def get_clinical_trial_info(nct_id):
    if not is_valid_nct_format(nct_id): return None, "Invalid NCT ID format."
    nct_id_upper = nct_id.strip().upper(); base_url = "https://clinicaltrials.gov/api/v2/studies"
    try:
        response = requests.get(f"{base_url}/{nct_id_upper}", timeout=20)
        if response.status_code == 200:
            if not response.content: return None, f"{nct_id_upper} empty content."
            try: return response.json(), None
            except json.JSONDecodeError: return None, f"{nct_id_upper} invalid JSON."
        elif response.status_code == 404: return None, f"{nct_id_upper} not found."
        else: error_detail = response.text[:500]; return None, f"Fetch fail {response.status_code}. Detail: {error_detail}"
    except requests.exceptions.Timeout: return None, f"Timeout fetching {nct_id_upper}."
    except requests.exceptions.RequestException as e: print(f"Fetch error {nct_id_upper}: {e}"); return None, f"Network error: {e}"

def process_trial_data(json_data):
    extracted_info = {'NCT_ID': None, 'Brief_Title': None, 'Official_Title': None, 'Status': None, 'Conditions': None, 'Interventions': None, 'Intervention_Types': None, 'Minimum_Age': None, 'Maximum_Age': None, 'Sex': None, 'Inclusion_Criteria': None, 'Exclusion_Criteria': None, 'Drugs': None, 'Phase': None, 'Study_Type': None}
    if not json_data or not isinstance(json_data, dict): print("Warning: process_trial_data invalid input."); return extracted_info
    try:
        protocol = json_data.get('protocolSection', {});
        if not isinstance(protocol, dict): print(f"Warning: 'protocolSection' missing/invalid."); return extracted_info
        id_mod = protocol.get("identificationModule", {}); status_mod = protocol.get("statusModule", {}); cond_mod = protocol.get("conditionsModule", {}); arms_mod = protocol.get("armsInterventionsModule", {}); elig_mod = protocol.get("eligibilityModule", {}); design_mod = protocol.get("designModule", {})
        extracted_info['NCT_ID'] = id_mod.get("nctId"); extracted_info['Brief_Title'] = id_mod.get("briefTitle"); extracted_info['Official_Title'] = id_mod.get("officialTitle"); extracted_info['Status'] = status_mod.get("overallStatus")
        conditions_list = cond_mod.get("conditions", []); extracted_info['Conditions'] = ', '.join(conditions_list) if conditions_list else "Not specified"
        interventions_list = arms_mod.get("interventions", [])
        if interventions_list: extracted_info['Interventions'] = '; '.join(f"{i.get('type', 'N/A')}: {i.get('name', 'N/A')}" for i in interventions_list if i); extracted_info['Intervention_Types'] = list(set(i.get('type') for i in interventions_list if i and i.get('type'))); extracted_info['Drugs'] = ', '.join(i.get('name') for i in interventions_list if i and i.get('type', '').upper() == 'DRUG' and i.get('name')) or "No specific drugs listed"
        else: extracted_info['Interventions'] = "Not specified"; extracted_info['Intervention_Types'] = []; extracted_info['Drugs'] = "Not specified"
        extracted_info['Minimum_Age'] = elig_mod.get("minimumAge", "Not specified"); extracted_info['Maximum_Age'] = elig_mod.get("maximumAge", "No maximum age specified"); extracted_info['Sex'] = elig_mod.get("sex", "Not specified"); eligibility_criteria = elig_mod.get("eligibilityCriteria", "")
        if isinstance(eligibility_criteria, str) and 'Exclusion Criteria:' in eligibility_criteria: parts = eligibility_criteria.split('Exclusion Criteria:', 1); extracted_info['Inclusion_Criteria'] = parts[0].replace('Inclusion Criteria:', '').strip(); extracted_info['Exclusion_Criteria'] = parts[1].strip()
        elif isinstance(eligibility_criteria, str): extracted_info['Inclusion_Criteria'] = eligibility_criteria.replace('Inclusion Criteria:', '').strip(); extracted_info['Exclusion_Criteria'] = "Not specified"
        else: extracted_info['Inclusion_Criteria'] = "Not provided"; extracted_info['Exclusion_Criteria'] = "Not provided"
        if not extracted_info['Inclusion_Criteria']: extracted_info['Inclusion_Criteria'] = "Not provided";
        if not extracted_info['Exclusion_Criteria']: extracted_info['Exclusion_Criteria'] = "Not provided"
        phases_list = design_mod.get("phases", []); extracted_info['Phase'] = ', '.join(phases_list) if phases_list else "Not Applicable/Not Specified"; extracted_info['Study_Type'] = design_mod.get("studyType", "Not specified")
    except Exception as e: print(f"Error processing trial data for {extracted_info.get('NCT_ID', 'Unknown')}: {e}")
    return extracted_info

def initialize_llm(temperature=0.0):
    try: llm = AzureChatOpenAI(api_key=AZURE_OPENAI_API_KEY, api_version=AZURE_API_VERSION, azure_endpoint=AZURE_OPENAI_ENDPOINT, deployment_name=AZURE_DEPLOYMENT_NAME, temperature=temperature, max_retries=3, request_timeout=60); return llm
    except Exception as e: print(f"Error initializing LLM: {e}"); raise

# --- Endpoint 1: Returns Summary + All Data Needed for Endpoint 2 ---
@app.route('/fetch_and_summarize', methods=['POST'])
def fetch_and_summarize_trial_for_client_state():
    """
    Fetches trial data, processes it, generates a 3-category summary,
    and returns the summary PLUS all processed data and original inputs
    needed for the next step (client holds this state).
    Expects JSON: {"nct_id": "NCT...", "indication": "...", "product": "...", "scenario_name": "..."}
    """
    start_time = time.time()
    if not request.is_json:
        return jsonify({"status": "error", "message": "Request must be JSON"}), 400

    # Store the original request data, it's needed later
    original_input_data = request.get_json()
    nct_id = original_input_data.get('nct_id')
    indication = original_input_data.get('indication')
    # product = original_input_data.get('product', 'Not Provided') # Not directly needed here, but will be returned
    # scenario_name = original_input_data.get('scenario_name', 'Default Scenario') # Not directly needed here

    if not nct_id: return jsonify({"status": "error", "message": "Missing 'nct_id'"}), 400
    if not indication: return jsonify({"status": "error", "message": "Missing 'indication'"}), 400

    nct_id = nct_id.strip().upper()

    # --- Validation ---
    if not is_valid_nct_format(nct_id):
        suggestions = suggest_nct_ids_by_indication(indication)
        return jsonify({"status": "error", "message": f"Invalid NCT ID format: '{nct_id}'.", "suggestions": suggestions}), 400
    if not does_nct_id_exist(nct_id):
        suggestions = suggest_nct_ids_by_indication(indication)
        return jsonify({"status": "error", "message": f"NCT ID '{nct_id}' not found.", "suggestions": suggestions}), 404

    # --- Fetch Data ---
    trial_json, error_msg = get_clinical_trial_info(nct_id)
    if error_msg:
        suggestions = suggest_nct_ids_by_indication(indication)
        return jsonify({"status": "error", "message": f"Fetch error for {nct_id}: {error_msg}", "suggestions": suggestions}), 500
    if not trial_json: return jsonify({"status": "error", "message": f"No data returned for {nct_id}."}), 500

    # --- Process Data ---
    processed_data = process_trial_data(trial_json)
    if not processed_data.get('NCT_ID'): return jsonify({"status": "error", "message": f"Failed to process critical data for {nct_id}."}), 500

    # --- Generate Summary ---
    prompt_trial_template = """You are an expert clinical trial analyst. Summarize the key aspects of the following clinical trial based solely on the provided JSON data. Structure your summary into exactly three categories as specified below. Be concise and factual in your summary.

Trial JSON Data:
```json
{trial_json_string}
Please provide the summary in this format:

1: Specific Diagnosis/Condition(s) Targeted
[List the primary medical condition(s) this trial is focused on, as mentioned in the 'conditions' field.]

2: Key Comorbidities or Patient Characteristics (from Eligibility Criteria)
[Based only on the 'Inclusion Criteria' and 'Exclusion Criteria' fields, list significant comorbidities, prior treatments, or patient characteristics that determine eligibility. Focus on medical conditions mentioned.]

3: Overall Trial Objective (One Sentence)
[Provide a single sentence summarizing the main goal or purpose of the study, inferring from title, interventions, and conditions.]

Give the exact subheadings based on the summaries you are giving for these 3 categories.
"""
    trial_summary = "Summary generation failed." # Default
    try:
        llm_summarizer = initialize_llm(temperature=0.1)
        prompt = PromptTemplate.from_template(prompt_trial_template)
        chain_trial = LLMChain(llm=llm_summarizer, prompt=prompt)
        trial_json_string = json.dumps(trial_json, indent=2)
        trial_summary = chain_trial.run( trial_json_string=trial_json_string,
            nct_id=processed_data.get('NCT_ID', 'N/A'), brief_title=processed_data.get('Brief_Title', 'N/A'),
            official_title=processed_data.get('Official_Title', 'N/A'), conditions=processed_data.get('Conditions', 'N/A'),
            inclusion_criteria_snippet=processed_data.get('Inclusion_Criteria', 'N/A')[:1000],
            exclusion_criteria_snippet=processed_data.get('Exclusion_Criteria', 'N/A')[:1000]
        )
    except Exception as e:
        print(f"Error during LLM summarization for {nct_id}: {e}")
        trial_summary = f"Summary generation failed: {e}"

    # --- Return Success Response with ALL data needed later ---
    end_time = time.time()
    return jsonify({
        "status": "success",
        "message": "Trial data processed and summarized. Client should retain 'processed_data' and 'original_input' for the next step.",
        "duration_seconds": round(end_time - start_time, 2),
        "trial_summary": trial_summary,
        "processed_data": processed_data, # << Return processed data
        "original_input": original_input_data # << Return original inputs
    }), 200


# --- Endpoint 2: Generates Insights Using Data Provided by Client ---
@app.route('/generate_insights', methods=['POST'])
def generate_trial_insights_from_client_state():
    """
    Generates detailed commercial insights using processed data and original inputs
    provided directly by the client in the request body.
    Expects JSON: {
        "processed_data": { ... },
        "original_input": { "nct_id": "...", "indication": "...", "product": "...", "scenario_name": "..." }
    }
    """
    start_time = time.time()
    if not request.is_json:
        return jsonify({"status": "error", "message": "Request must be JSON"}), 400

    input_data = request.get_json()
    processed_data = input_data.get('processed_data')
    original_input = input_data.get('original_input')

    # --- Validate Received Data ---
    if not processed_data or not isinstance(processed_data, dict):
        return jsonify({"status": "error", "message": "Missing or invalid 'processed_data' in request body"}), 400
    if not original_input or not isinstance(original_input, dict):
        return jsonify({"status": "error", "message": "Missing or invalid 'original_input' in request body"}), 400
    # Basic check for essential keys within the dictionaries
    if not processed_data.get('NCT_ID') or not original_input.get('indication'):
         return jsonify({"status": "error", "message": "Received data is incomplete ('NCT_ID' or 'indication' missing)"}), 400

    # --- Prepare Inputs for Insight Generation from Received Data ---
    nct_id = processed_data.get('NCT_ID', 'Not specified') # Get NCT ID for logging
    print(f"[{nct_id}] Generating final insights from client-provided data...")
    brief_title = processed_data.get('Brief_Title', 'Not specified')
    conditions = processed_data.get('Conditions', 'Not specified')
    interventions = processed_data.get('Interventions', 'Not specified')
    min_age = processed_data.get('Minimum_Age', 'Not specified')
    max_age = processed_data.get('Maximum_Age', 'No maximum age specified')
    sex = processed_data.get('Sex', 'Not specified')
    inclusion_only = processed_data.get('Inclusion_Criteria', 'Not provided')
    exclusion_only = processed_data.get('Exclusion_Criteria', 'Not provided')
    # --- Get scenario info from original_input dict ---
    scenario_name = original_input.get('scenario_name', 'Default Scenario')
    indication = original_input.get('indication', 'Not Provided')
    product = original_input.get('product', 'Not Provided')
    # --- Construct derived fields ---
    target_population = f"Ages: {min_age} to {max_age}, Sex: {sex}"

    # --- Define the Detailed Insight Prompt (Same as before) ---
    insight_prompt_template = '''You are a pharmaceutical commercial strategist and market access analyst. Your task is to analyze the following clinical trial data and generate structured, comprehensive, and clinically valid insights tailored for life sciences commercial teams.
Using the provided medical indication name and the corresponding Inclusion and Exclusion Criteria extracted from ClinicalTrials data, identify and compile all relevant ICD-10 codes. Thoroughly interpret the clinical context described in each criterion to determine the appropriate diagnostic codes.
For the inclusion criteria, focus on identifying ICD-10 codes that accurately represent the underlying medical conditions or diagnoses specified. Organize the resulting codes into logically defined groups based on clinical similarity, comorbidities, or related pathologies.
For the exclusion criteria, evaluate each condition or contraindication described, and select corresponding ICD-10 codes that clearly reflect those exclusion parameters. Group these codes meaningfully to mirror the structure and intent of the criteria.
Exclude any codes that refer to medical procedures, surgeries, or adverse events, as the focus should remain strictly on diagnostic classifications.
Ensure that all selected codes are accurate, up-to-date, and aligned with standard classifications, referencing authoritative sources such as the WHO ICD-10 database or equivalent coding guidelines.
Carefully examine the inclusion and exclusion criteria to extract:
1. Age Groups – Analyze the clinical condition specified in the trial and categorize participants into meaningful age groups. While a separate group for ages 65+ can be considered, create 3 to 5 distinct age buckets within the 1–60 age range based on the nature of the disease and the trial's inclusion criteria. Do not restrict the categorization to just the 1–60 and 65+ groups. Ensure that each age group is clearly defined, contextually relevant, and accurately reflects the trial's requirements.
2. Age Criteria – Review the details related to age mentioned in the inclusion criteria, if available.
2. Gender eligibility, if mentioned (e.g., "All", "Male", "Female").
3. In AddressableMarketDefinition - Analyze the text in the inclusion criteria thoroughly and provide it.
Ensure the extracted insights are based strictly on the given clinical trial data and ICD definitions.

Ensure the extracted insights are based strictly on the given clinical trial data and ICD definitions.
 
CLINICAL TRIAL INFORMATION:
- Trial ID: {nct_id}
- Title: {brief_title}
- Condition(s): {conditions}
- Intervention(s): {interventions}
- Target Population: {target_population}
- Inclusion Criteria: {inclusion_only}
- Exclusion Criteria: {exclusion_only}
SCENARIO INFORMATION:
- Scenario Name: {scenario_name}
- Indication: {indication}
- Product of Interest: {product}
 
Return the following insights in the below JSON Format ONLY:
{{
  "BroadMarketDefinition": {{
  "BroadMarketDescription":"",
    "ICDCodes": []
  }},
  "AddressableMarketDefinition": "A brief blurb summarizing how we will define the addressable market (e.g., this will be along the lines of 'In order to identify...')",
  "AddressableMarketCriteriaByPatientAttribute": {{
   	"AgeCriteria":{{
	"AgeCriteria":"Age criteria value",
	"AgeCriteriaDescription":"Includes most commonly diagnosed for this condition"
	}},
    "Age": {{
      "AgeGroup1": {{
        "AgeGroup1": "Group 1 Name",
      }},
      "AgeGroup2": {{
        "AgeGroup2": "Group 2 Name",
      }},
     ...
    "Gender": "Male, Female, Both or Does not apply",
	"GenderDescription": "This condition is observed in both males and females",
    "AdditionalICDCodesRequired": {{
      "Group1": {{
        "GroupName": "Group 1 Name",
        "GroupDescription": "Description of Group 1",
        "ICDCodes": []
      }},
      "Group2": {{
        "GroupName": "Group 2 Name",
        "GroupDescription": "Description of Group 2",
        "ICDCodes": []
      }},
      "Group3": {{
        "GroupName": "Group 3 Name",
        "GroupDescription": "Description of Group 3",
        "ICDCodes": []
      }},
      "Group4": {{
        "GroupName": "Group 4 Name",
        "GroupDescription": "Description of Group 4",
        "ICDCodes": []
      }}
    }},
    "ICDCodesToExclude": {{
      "Group1": {{
        "GroupName": "Exclusion Group 1 Name",
        "GroupDescription": "Description of Exclusion Group 1",
        "ICDCodes": []
      }},
      "Group2": {{
        "GroupName": "Exclusion Group 2 Name",
        "GroupDescription": "Description of Exclusion Group 2",
        "ICDCodes": []
      }},
      "Group3": {{
        "GroupName": "Exclusion Group 3 Name",
        "GroupDescription": "Description of Exclusion Group 3",
        "ICDCodes": []
      }},
      "Group4": {{
        "GroupName": "Exclusion Group 4 Name",
        "GroupDescription": "Description of Exclusion Group 4",
        "ICDCodes": []
      }},
    }}
  }}
}}

Here is the one such example:
{{
  "BroadMarketDefinition": {{
    "BroadMarketDescription": "Includes all ICD codes related to atrial fibrillation to define the broader market population before applying inclusion/exclusion filters.",
    "ICDCodes": ["I48.0", "I48.11", "I48.19", "I48.2", "I48.20", "I48.21", "I48.3", "I48.4", "I48.91", "I48.92"]
  }},
  "AddressableMarketDefinition": "To refine the addressable population, we will stratify atrial fibrillation patients into clinically meaningful subgroups based on comorbidities and trial exclusion patterns observed in real-world data.",
  "AddressableMarketCriteriaByPatientAttribute": {{
	"AgeCriteria":{{
	"AgeCriteria":"21+",
	"AgeCriteriaDescription":"Includes adult patients aged 21 years and above, in line with typical clinical trial eligibility and disease onset."
	}},
    "Age": {{
      "AgeGroup1": {{
        "AgeGroup1": "0-18"
        }},
      "AgeGroup2": {{
        "AgeGroup2": "19-36"
      }},
      "AgeGroup3": {{
        "AgeGroup3": "37-54"
      }},
      "AgeGroup4": {{
        "AgeGroup4": "55-72"
      }},
      "AgeGroup5": {{
        "AgeGroup5": "73-90"
      }},
      "AgeGroup6": {{
        "AgeGroup6": "91-100"
      }}
    }},
    "Gender": "Both",
	"GenderDescription":"This condition is observed in both males and females"
    "AdditionalICDCodesRequired": {{
      "Group1": {{
        "GroupName": "Hypertension Comorbidity",
        "GroupDescription": "Patients with concurrent hypertension requiring management",
        "ICDCodes": ["I10", "I11", "I12", "I13", "I15"]
      }},
      "Group2": {{
        "GroupName": "Heart Failure Comorbidity",
        "GroupDescription": "Patients with coexisting heart failure (systolic/diastolic)",
        "ICDCodes": ["I50", "I50.1", "I50.2", "I50.3", "I50.4"]
      }},
      "Group3": {{
        "GroupName": "Diabetes Mellitus Comorbidity",
        "GroupDescription": "Patients with type 1/2 diabetes requiring pharmacological management",
        "ICDCodes": ["E10", "E11", "E13"]
      }},
      "Group4": {{
        "GroupName": "Other Arrhythmias",
        "GroupDescription": "Patients with concurrent supraventricular/ventricular arrhythmias",
        "ICDCodes": ["I47", "I49", "I46", "I44", "I45"]
      }}
    }},
    "ICDCodesToExclude": {{
      "Group1": {{
        "GroupName": "Permanent AF Exclusion",
        "GroupDescription": "Patients with permanent AF ineligible for rhythm control strategies",
        "ICDCodes": ["I48.21"]
      }},
      "Group2": {{
        "GroupName": "Valvular Heart Disease",
        "GroupDescription": "Patients with structural valve abnormalities requiring intervention",
        "ICDCodes": ["I34", "I35", "I36", "I37", "I38"]
      }},
      "Group3": {{
        "GroupName": "Advanced Renal Disease",
        "GroupDescription": "Patients with stage 4/5 CKD or dialysis dependence",
        "ICDCodes": ["N18.4", "N18.5", "N18.6", "Z99.2"]
      }},
      "Group4": {{
        "GroupName": "Hepatic Impairment",
        "GroupDescription": "Patients with cirrhosis or severe liver dysfunction",
        "ICDCodes": ["K70", "K71", "K72", "K73", "K74"]
      }}
    }}
  }}
}}

    IMPORTANT: Populate the JSON structure accurately based only on the provided CLINICAL TRIAL INFORMATION and SCENARIO INFORMATION. Generate valid ICD-10 codes relevant to the clinical descriptions in the criteria. If criteria are vague or don't map clearly to ICD-10, state that in the description and leave the ICDCodes array empty for that section. Fill in the group names and descriptions logically.'''

    # --- Generate Insights via LLM ---
    final_output = None
    try:
        llm_insight = initialize_llm(temperature=0.0)
        prompt = PromptTemplate.from_template(insight_prompt_template)
        chain_insight = LLMChain(llm=llm_insight, prompt=prompt)

        final_insights_str = chain_insight.run(
            # Pass variables extracted from BOTH processed_data and original_input
            nct_id=nct_id, brief_title=brief_title, conditions=conditions, interventions=interventions,
            target_population=target_population, # Use constructed string
            scenario_name=scenario_name, indication=indication, product=product, # From original_input
            exclusion_only=exclusion_only, inclusion_only=inclusion_only, # From processed_data
            min_age=min_age, max_age=max_age, sex=sex # Also needed if prompt uses them directly
        )

        # --- Post-process LLM Output (Extract and Parse JSON) ---
        print(f"[{nct_id}] Parsing LLM response for final insights...")
        try:
            match = re.search(r"```json\s*(\{.*?\})\s*```", final_insights_str, re.DOTALL | re.IGNORECASE)
            if match: json_str = match.group(1)
            elif final_insights_str.strip().startswith('{') and final_insights_str.strip().endswith('}'): json_str = final_insights_str.strip()
            else: raise ValueError("Could not find JSON block in LLM output.")
            parsed_insights = json.loads(json_str); final_output = parsed_insights
        except (json.JSONDecodeError, ValueError) as json_e:
            error_message = f"LLM output for {nct_id} could not be parsed as JSON: {json_e}"
            print(f"Warning: {error_message}"); final_output = {"parsing_warning": error_message, "raw_llm_output": final_insights_str}
        except Exception as parse_e:
            error_message = f"Unexpected error parsing LLM output for {nct_id}: {parse_e}"
            print(f"Error: {error_message}"); final_output = {"parsing_error": error_message, "raw_llm_output": final_insights_str}

    except Exception as llm_e:
        print(f"Error during LLM insight generation for {nct_id}: {llm_e}")
        return jsonify({"status": "error", "message": f"Failed to generate final insights via LLM: {llm_e}"}), 500

    # --- Return Success Response ---
    end_time = time.time()
    return jsonify({
        "status": "success",
        "message": "Final insights generated from provided data.",
        "duration_seconds": round(end_time - start_time, 2),
        "insights": final_output
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)
