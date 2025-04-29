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

# --- Get LLM-Suggested NCT IDs ---
def suggest_closest_nct_ids(nct_id, llm):
    prompt = f"""
The user entered an invalid NCT ID: {nct_id}.

Please suggest the top 3 most likely valid NCT IDs that the user may have intended. Only return the list in this format:
[
  "NCTxxxxxxx1",
  "NCTxxxxxxx2",
  "NCTxxxxxxx3"
]
"""
    response = llm.invoke(prompt)
    return response.content

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
    llm_validator = AzureChatOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2024-05-01-preview",
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        deployment_name=AZURE_DEPLOYMENT_NAME,
        temperature=0.2
    )

    if not is_valid_nct_format(trial_id) or not does_nct_id_exist(trial_id):
        st.error("❌ The NCT ID seems incorrect or not found. Showing best guesses below...")
        suggestions = suggest_closest_nct_ids(trial_id, llm_validator)
        st.warning(f"🔍 Did you mean one of the following?\n\n{suggestions}")
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
    default_prompt = """You are a pharmaceutical commercial strategist and market access analyst. Use the following clinical trial data to generate concise, structured, and real-world insights tailored for life sciences commercial teams.

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

Return the following insights in bullet points and numbers only, no narrative:

🔹 ICD-10 CODES
- Top 5 ICD-10 codes relevant to the condition, with brief justification.

🔹 DRUG PROFILE + COMPETITIVE ANALYSIS
- List each drug in {drugs} with class/mechanism of action
- Is any drug considered standard of care (SoC)?
- Known side effects that may impact uptake
- Any relevant biosimilars or generics in pipeline

🔹 EPIDEMIOLOGY SNAPSHOT
- Global prevalence (%)
- US prevalence (%)
- EU prevalence (%)
- Age distribution (%): <18, 18-40, 41-60, 61+
- Gender ratio: Male (%), Female (%)
- 5-year trend: CAGR (%)

🔹 MARKET LANDSCAPE
- Global market size (USD)
- US market size (USD)
- Annual market growth rate (%)
- Top 3 competitor treatments with global market share (%)
- Differentiation potential of {product} vs competitors

🔹 PATIENT ACCESS + TARGETING
- % of eligible patient population (based on criteria)
- Diagnosed patients globally (count)
- Avg. age at diagnosis
- % of patients requiring intervention/treatment
- Primary channels of care (e.g. oncology clinics, hospitals)

🔹 COMMERCIAL RECOMMENDATIONS
- Target segment(s) with highest commercial viability
- Suggested launch positioning or indication extension
- Risk factors or data gaps that could impact reimbursement/access
- Key commercial opportunities for {drugs}
- Top competitor therapies or trials
- Demographics of eligible patients
- Implications for launch strategy or targeting

Be extremely concise, structure your output with clear headers and bullet points."""

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
