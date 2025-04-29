import streamlit as st
import requests
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
import re

# --- Azure OpenAI Configuration ---
AZURE_OPENAI_ENDPOINT = "https://ciaiaiservices.openai.azure.com/"
AZURE_OPENAI_API_KEY = "817dce22f5a548b8b11fe0b6a3cf2c36"
AZURE_DEPLOYMENT_NAME = "gpt-35-turbo-0613"

# --- Utility: Check NCT ID Format ---
def is_valid_nct_format(nct_id):
    return re.fullmatch(r"NCT\d{8}", nct_id.strip().upper()) is not None

# --- Utility: Check NCT ID Existence ---
def does_nct_id_exist(nct_id):
    url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
    response = requests.get(url)
    return response.status_code == 200

# --- Alternative Suggestion Using Indication Search ---
def suggest_nct_ids_by_indication(indication):
    if not indication:
        return []
    query = indication.strip().replace(" ", "+")
    url = f"https://clinicaltrials.gov/api/v2/studies?query.cond={query}&pageSize=3"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            results = response.json().get("studies", [])
            return [
                {
                    "nct_id": r.get("protocolSection", {}).get("identificationModule", {}).get("nctId"),
                    "title": r.get("protocolSection", {}).get("identificationModule", {}).get("briefTitle", "")
                }
                for r in results if r
            ]
        else:
            return []
    except:
        return []

# --- Trial Data Fetch Function ---
def get_clinical_trial_info(nct_id):
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    try:
        response = requests.get(f"{base_url}/{nct_id}")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching trial data: {str(e)}")
        return None
    
# --- Fetch indications_and_usage from FDA API ---
def fetch_and_extract_fda_products(indication):
    encoded_indication = requests.utils.quote(f'"{indication}"')
    url = f"https://api.fda.gov/drug/label.json?search=indications_and_usage:{encoded_indication}&limit=10"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            results = response.json().get("results", [])
            if not results:
                return "No relevant FDA data found."
            usage_text = results[0].get("indications_and_usage", [""])[0]

            llm_extract = AzureChatOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                api_version="2024-05-01-preview",
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                deployment_name=AZURE_DEPLOYMENT_NAME,
                temperature=0
            )

            prompt = PromptTemplate.from_template("""
From the following FDA indications_and_usage text, extract all product (drug) names mentioned clearly. Return only the product names as a list.

"
{usage_text}
"
""")

            chain = LLMChain(llm=llm_extract, prompt=prompt)
            return chain.run(usage_text=usage_text)

        else:
            return "FDA API returned an error."
    except Exception as e:
        return f"Error fetching data from FDA API: {str(e)}"


# --- Trial Data Processing Function ---
def process_trial_data(json_data):
    extracted_info = {
        'NCT_ID': None, 'Brief_Title': None, 'Conditions': None,
        'Interventions': None, 'Minimum_Age': None,
        'Maximum_Age': None, 'Sex': None,
        'Inclusion_Exclusion': None, 'Drugs': None
    }

    if not json_data or 'protocolSection' not in json_data:
        return extracted_info

    protocol = json_data['protocolSection']
    id_mod = protocol.get("identificationModule", {})
    cond_mod = protocol.get("conditionsModule", {})
    arms_mod = protocol.get("armsInterventionsModule", {})
    elig_mod = protocol.get("eligibilityModule", {})

    extracted_info['NCT_ID'] = id_mod.get("nctId")
    extracted_info['Brief_Title'] = id_mod.get("briefTitle")
    extracted_info['Conditions'] = ', '.join(cond_mod.get("conditions", []))
    extracted_info['Interventions'] = ', '.join(
        f"{i.get('type')}: {i.get('name')}" for i in arms_mod.get("interventions", [])
    )
    extracted_info['Minimum_Age'] = elig_mod.get("minimumAge")
    extracted_info['Maximum_Age'] = elig_mod.get("maximumAge", "No maximum")
    extracted_info['Sex'] = elig_mod.get("sex")
    extracted_info['Inclusion_Exclusion'] = elig_mod.get("eligibilityCriteria", "Not provided")
    extracted_info['Drugs'] = ', '.join(
        i.get('name') for i in arms_mod.get("interventions", []) if i.get('type') == 'DRUG'
    )

    return extracted_info

# --- UI Config ---
st.set_page_config(page_title="Clinical Trial Insight Generator", layout="wide")
st.title("\U0001F9EC GPT-Powered Clinical Trial Insight Generator")

# --- Collect Inputs ---
with st.form("clinical_trial_form"):
    scenario_name = st.text_input("Scenario Name")
    indication = st.text_input("Indication")
    product = st.text_input("Product of Interest")
    trial_id = st.text_input("Clinical Trial ID (NCT ID)").strip().upper()
    submitted = st.form_submit_button("Submit")

# --- On Submit: validate and process ---
if submitted and trial_id:
    if not is_valid_nct_format(trial_id) or not does_nct_id_exist(trial_id):
        st.error("❌ The NCT ID seems incorrect or not found. Showing trials based on your indication...")
        suggested_ncts = suggest_nct_ids_by_indication(indication)
        if suggested_ncts:
            st.warning("🔍 You can try one of these valid NCT IDs:")
            for trial in suggested_ncts:
                if trial["nct_id"]:
                    if st.button(f"{trial['nct_id']}: {trial['title']}", key=trial["nct_id"]):
                        trial_id = trial["nct_id"]
                        trial_json = get_clinical_trial_info(trial_id)
                        if trial_json:
                            st.session_state['trial_json'] = trial_json
                            st.session_state['scenario_name'] = scenario_name
                            st.session_state['indication'] = indication
                            st.session_state['product'] = product

                            clean_data = process_trial_data(trial_json)
                            st.session_state['clean_data'] = clean_data

                            st.success("✅ Trial data fetched and processed. Now proceed to generate insights.")
                        st.stop()
        else:
            st.info("No matching NCT IDs found for the given indication.")
    else:
        trial_json = get_clinical_trial_info(trial_id)
        if trial_json:
            st.session_state['trial_json'] = trial_json
            st.session_state['scenario_name'] = scenario_name
            st.session_state['indication'] = indication
            st.session_state['product'] = product

            clean_data = process_trial_data(trial_json)
            st.session_state['clean_data'] = clean_data

            st.success("✅ Trial data fetched and processed. Now proceed to generate insights.")




# --- If trial_json exists in session, allow prompt and insight generation ---
if 'trial_json' in st.session_state and 'clean_data' in st.session_state:
    trial_json = st.session_state['trial_json']
    clean_data = st.session_state['clean_data']
    scenario_name = st.session_state['scenario_name']
    indication = st.session_state['indication']
    product = st.session_state['product']

    # prompt_trial = "You are an expert in analyzing clinical trial studies and extracting key insights from the trial data for quick review. Take the {trial_json} and summarize it clearly."
    # llm_trial = AzureChatOpenAI(
    #     api_key=AZURE_OPENAI_API_KEY,
    #     api_version="2024-05-01-preview",
    #     azure_endpoint=AZURE_OPENAI_ENDPOINT,
    #     deployment_name=AZURE_DEPLOYMENT_NAME,
    #     temperature=0
    # )
    # chain_trial = LLMChain(llm=llm_trial, prompt=PromptTemplate.from_template(prompt_trial))
    # insights_trial = chain_trial.run(trial_json=trial_json)
    # st.success("✅ Trial Summary:")
    # st.markdown(insights_trial)

    nct_id = clean_data.get('NCT_ID', 'Not specified')
    brief_title = clean_data.get('Brief_Title', 'Not specified')
    conditions = clean_data.get('Conditions', 'Not specified')
    interventions = clean_data.get('Interventions', 'Not specified')
    min_age = clean_data.get('Minimum_Age', 'Not specified')
    max_age = clean_data.get('Maximum_Age', 'No maximum')
    sex = clean_data.get('Sex', 'Not specified')
    drugs = clean_data.get('Drugs', 'Not specified')

    target_population = f"Ages: {min_age} to {max_age}, Sex: {sex}"
    inclusion_exclusion = clean_data.get('Inclusion_Exclusion', 'Not provided')

    if 'Exclusion Criteria:' in inclusion_exclusion:
        parts = inclusion_exclusion.split('Exclusion Criteria:')
        inclusion_only = parts[0].replace('Inclusion Criteria:', '').strip()
        exclusion_only = parts[1].strip()
    else:
        inclusion_only = inclusion_exclusion
        exclusion_only = "Not specified"

    st.markdown("### ✍️ Edit your GPT prompt")
    default_prompt = """You are a pharmaceutical commercial strategist and market access analyst. Your task is to analyze the following clinical trial data and generate structured, comprehensive, and clinically valid insights tailored for life sciences commercial teams.
Use the indication name along with Inclusion and Exclusion Criteria provided below to extract all relevant ICD-10 codes by referring to authoritative web sources (e.g., WHO ICD-10 database).Include only codes that are directly related to the indication.
Carefully examine the inclusion and exclusion criteria to extract:
1. Age eligibility range (in years), using the lowest and highest age values stated or implied.
2. Gender eligibility, if mentioned (e.g., "All", "Male", "Female").
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
example_json = ""{{
  "BroadMarketDefinition": {{
    "ICDCodes": []
  }},
  "AddressableMarketDefinition": "A brief blurb summarizing how we will define the addressable market (e.g., this will be along the lines of 'In order to identify...')",
  "AddressableMarketCriteriaByPatientAttribute": {{
    "Age": "Criteria range or Does not apply",
    "Gender": "Male, Female, Both or Does not apply",
    "AdditionalICDCodesRequired": {{
      "Group1": [],
      "Group2": [],
      "Group3": [],
      "Group4": []
    }},
    "ICDCodesToExclude": {{
      "Group1": [],
      "Group2": [],
      "Group3": [],
      "Group4": []
    }}
  }}
}}""
Example_json:
{{
  "Broad Market Definition": [
    "I48.0",
    "I48.1",
    "I48.2",
    "I48.91",
    "I48.3",
    "I48.4",
    "I48.92"
  ],
  "Addressable Market Definition": "Patients with a diagnosis of atrial fibrillation or atrial flutter who are older adults (age ≥65 with high CHA2DS2-VASc score), are not on oral anticoagulation due to unsuitability or unwillingness, have bleeding risk factors, and are unsuitable for LAA closure. We will exclude patients with reversible AF causes, recent use of anticoagulants, recent intracranial bleeding, recent stroke/TIA, mechanical heart valves, or patients on dialysis.",
  "Addressable Market Criteria by Each of the following Patient Attribute": {{
    "Age": "65-74 with CHA2DS2VASc ≥4 OR ≥75 with CHA2DS2VASc ≥3 (proxy using age ≥65)",
    "Gender": "Both",
    "Additional ICD codes that must also be present": {{
      "Group1": ["Z13.6"],
      "Group2": ["I50.x"],
      "Group3": ["I10", "I11.x", "I12.x", "I13.x", "I15.x"],
      "Group4": ["E11.x"],
      "Group5": ["I63.x", "G45.x"],
      "Group6": ["N18.4", "N18.5"],
      "Group7": ["D62", "D68.32", "D68.4"],
      "Group8": ["Z79.02", "Z79.01"],
      "Group9": ["R26.2"],
      "Group10": ["M05.x", "M06.x", "M15.x-M19.x"]
    }},
    "ICD codes that must not be present": {{
      "Group1": ["I97.89", "I26.x", "E05.x", "F10.x"],
      "Group2": ["S06.x"],
      "Group3": ["H45.x"],
      "Group4": ["I63.x", "G45.x"],
      "Group5": ["Z95.2"],
      "Group6": ["N18.6", "Z99.2"]
    }}
  }}
}}
"""

    prompt_input = st.text_area("Prompt Template", value=default_prompt, height=650)

    if st.button("\U0001F680 Generate Final Insights"):
        with st.spinner("Calling Azure OpenAI..."):
            llm = AzureChatOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                api_version="2024-05-01-preview",
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                deployment_name=AZURE_DEPLOYMENT_NAME,
                temperature=0
            )
            chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_input))
            final_insights = chain.run(
                nct_id=nct_id,
                brief_title=brief_title,
                conditions=conditions,
                interventions=interventions,
                target_population=target_population,
                scenario_name=scenario_name,
                indication=indication,
                product=product,
                drugs=drugs,
                exclusion_only=exclusion_only,
                inclusion_only=inclusion_only
            )
            
            st.success("✅ Final Insights:")
            st.markdown(final_insights)
            st.session_state['final_insights'] = final_insights

if 'trial_json' in st.session_state and 'clean_data' in st.session_state:
    st.markdown("### ✏️ Post-processing Insight Refinement")

    user_query = st.text_input("Ask the assistant to change something about the insights")

    if user_query:
        llm_refine = AzureChatOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version="2024-05-01-preview",
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            deployment_name=AZURE_DEPLOYMENT_NAME,
            temperature=0.5
        )

        

        refinement_prompt = PromptTemplate.from_template("""Below is the original insight:

"
{insight}
"

User asked:
"
{query}
"

Update the original insight accordingly and return ONLY the revised insight. Do not include extra explanation.
""")

        # Display previous insight box
        if 'final_insights' in st.session_state:
            

            chain_refine = LLMChain(llm=llm_refine, prompt=refinement_prompt)
            revised = chain_refine.run(insight=st.session_state['final_insights'], query=user_query)
            st.success("🔁 Updated Insight:")
            st.markdown(revised)

            with st.expander("Generated Insight", expanded=True):
                st.markdown(st.session_state['final_insights'])
        else:
            st.info("Please generate the insights above first before refining them.")

st.markdown("### 🏥 FDA Competitor Product Extraction")
if indication:
    user = st.button("click on this to get product")
    if user:
        result = fetch_and_extract_fda_products(indication)
        if result and result.strip():
            st.success("✅ Extracted Product Names from FDA:")
            lines = result.strip().split("\n")
            formatted = "\n".join([f"- {line.strip()}" for line in lines if line.strip()])
            st.markdown(formatted)
        else:
            st.warning("⚠️ No products found or output was empty.")

