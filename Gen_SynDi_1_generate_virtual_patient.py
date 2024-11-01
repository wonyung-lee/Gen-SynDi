# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 13:47:42 2024

@author: wonyu
"""
## prerequisite
# openai (1.47.0), pandas, numpy 



import pandas as pd
import random
import numpy as np
from openai import OpenAI
import os
import json
from datetime import datetime



# OpenAI API key set
openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=openai_api_key   # Use your environment variable or directly insert your API key here
)

fatigue_disease = pd.read_csv('dataset_fatigue_disease.csv')
fatigue_pattern = pd.read_csv('dataset_fatigue_syndrome.csv')

non_conflicting_combinations = [
    ('Chronic Fatigue Syndrome', 'Qi and Blood Deficiency'),
    ('Chronic Fatigue Syndrome', 'Liver and Kidney Yin Deficiency'),
    ('Chronic Fatigue Syndrome', 'Spleen Deficiency with Dampness'),
    ('Chronic Fatigue Syndrome', 'Liver Qi Stagnation and Spleen Deficiency'),
    ('Sleep Disorder', 'Qi and Blood Deficiency'),
    ('Sleep Disorder', 'Spleen Qi Deficiency'),
    ('Sleep Disorder', 'Liver and Kidney Yin Deficiency'),
    ('Sleep Disorder', 'Heart and Spleen Yang Deficiency'),
    ('Sleep Disorder', 'Liver Qi Stagnation and Spleen Deficiency'),
    ('Fibromyalgia', 'Qi and Blood Deficiency'),
    ('Fibromyalgia', 'Spleen Qi Deficiency'),
    ('Fibromyalgia', 'Spleen Deficiency with Dampness'),
    ('Fibromyalgia', 'Heart and Spleen Yang Deficiency'),
    ('Fibromyalgia', 'Liver Qi Stagnation and Spleen Deficiency'),
    ('Fibromyalgia', 'Shaoyang Half-Exterior, Half-Interior'),
    ('Depression', 'Qi and Blood Deficiency'),
    ('Depression', 'Spleen Qi Deficiency'),
    ('Depression', 'Liver and Kidney Yin Deficiency'),
    ('Depression', 'Spleen Deficiency with Dampness'),
    ('Depression', 'Heart and Spleen Yang Deficiency'),
    ('Depression', 'Liver Qi Stagnation and Spleen Deficiency'),
    ('Depression', 'Shaoyang Half-Exterior, Half-Interior'),
    ('Hyperthyroidism', 'Qi and Blood Deficiency'),
    ('Hyperthyroidism', 'Spleen Qi Deficiency'),
    ('Hyperthyroidism', 'Liver and Kidney Yin Deficiency'),
    ('Hyperthyroidism', 'Spleen Deficiency with Dampness'),
    ('Hyperthyroidism', 'Heart and Spleen Yang Deficiency'),
    ('Hyperthyroidism', 'Liver Qi Stagnation and Spleen Deficiency')
]




def combine_columns(col1, col2):
    # 두 개의 컬럼을 결합하는 함수
    if pd.isna(col1) and pd.isna(col2):
        return ""
    elif pd.isna(col1):
        return col2
    elif pd.isna(col2):
        return col1
    else:
        return f"{col1}, {col2}"

def generate_virtual_patient_data(clinical_phenotype, target_language, selected_disease=None, selected_pattern=None):
    
    # If disease and pattern are not provided, randomly select from non_conflicting_combinations
    if selected_disease is None or selected_pattern is None:
        selected_disease_pattern = random.choice(non_conflicting_combinations)
        selected_disease = selected_disease_pattern[0]
        selected_pattern = selected_disease_pattern[1]

    # Combine disease and pattern for further data manipulation
    combined_sim = selected_disease + '_' + selected_pattern

    # Preparing the data
    subset_disease = fatigue_disease[['History Category', 'Question', selected_disease]]
    subset_pattern = fatigue_pattern[['History Category', 'Question', selected_pattern]]

    subset_combined = subset_disease.copy()
    subset_combined[selected_pattern] = subset_pattern[selected_pattern]
    subset_combined[combined_sim] = np.vectorize(combine_columns)(subset_disease[selected_disease], subset_pattern[selected_pattern])

    subset_QnA = subset_combined[['Question', combined_sim]].dropna(subset=[combined_sim])
    subset_QnA[combined_sim].replace('', pd.NA, inplace=True)
    subset_QnA_cleaned = subset_QnA.dropna(subset=[combined_sim])
    subset_str = '\n'.join([f"{row['Question']} : {row[combined_sim]}" for index, row in subset_QnA_cleaned.iterrows()])

    # dialogue_list = [row['Question'] + ' : ' + row[combined_sim] for index, row in subset_QnA_cleaned.iterrows()]

    # Generating patient response
    messages_simul = [{
        'role': 'system',
        'content': f"""
        You are a medical expert tasked with generating a virtual patient scenario representing the disease {selected_disease} and the traditional medicine pattern {selected_pattern}.
        Your goal is to naturally and specifically write out the conversation between the doctor and the patient.
        The given patient's responses are in the form of keywords.
        Based on these keywords, transform the patient's response into a natural sentence.
        If there is not enough information to respond, do not generate a response for that question and move on to the next one.
        The questions and responses must strictly follow the format below, and do not change the format.
        The responses must be specific and natural.

        Additionally, translate both the questions and responses into {target_language} and output only the translated result.
        Even if the questions and responses are in another language, make sure to write them in {target_language}.

        ### Format
        'Translated to "{target_language}": {{"Natural response in "{target_language}"}}'
        
        ### Example
        input: 'Since when have you been feeling fatigued? : More than 3 months' 
        output: 'Since when have you been feeling fatigued? : I started feeling fatigued about 3 months ago, and it has persisted since then.'
        """
    }]

    # Adding initial example
    messages_simul.extend([
        {'role': 'user', 'content': 'Could you explain exactly what symptoms you mean by fatigue? : Memory problems, difficulty concentrating, drowsiness, lack of focus, dizziness'},
        {'role': 'assistant', 'content': 'Could you explain exactly what symptoms you mean by fatigue? : I feel drowsy, my energy levels drop, my concentration decreases, and sometimes I feel dizzy or forget things.'}
    ])
    
    messages_simul.extend([{'role':'user', 'content': subset_str}])

    # Generating response
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages_simul,
        temperature=0
    )

    response_str = response.choices[0].message.content

    # Preparing chat log
    lines = response_str.strip().split('\n\n')
    chat_log = []
    for line in lines:
        if ' : ' in line:
            question, answer = line.split(' : ')
            if answer != '<NA>':  # Skip '<NA>' values
                chat_log.append({'role': 'user', 'content': question})
                chat_log.append({'role': 'assistant', 'content': answer})
                
    # Generating patient summary and emotional state
    patient_summary_personality = generate_patient_summary(selected_disease, selected_pattern, target_language, response_str)

    # Generating situation guidelines
    patient_condition, extracted_info = generate_situation_guidelines(clinical_phenotype, selected_disease, selected_pattern, target_language)

    # Collecting necessary data
    patient_data = {
        'selected_disease': selected_disease,
        'selected_pattern': selected_pattern,
        'chat_log': chat_log,
        'patient_summary_personality': patient_summary_personality,
        'patient_condition': patient_condition,
        'extracted_info': extracted_info,
        'subset_disease': subset_disease.to_json(orient='split'),
        'subset_pattern': subset_pattern.to_json(orient='split'),
        'target_language': target_language
    }

    return patient_data, selected_disease, selected_pattern

def generate_patient_summary(selected_disease, selected_pattern, target_language, response_str):
    # Example patient summary
    patient_example = """- Case Summary -
I started feeling tired easily and lost my appetite about a year ago.
Around 7 months ago, I also began feeling a sense of weakness and had trouble concentrating.
In the last 2 months, it has worsened to the point where it’s difficult to manage work and childcare, and I came to the clinic because I was worried.

- Personality and Emotional State -
○ Usual Personality:
Generally, I’m introverted and quiet, but I tend to be positive.
○ Current Emotional State:
Due to fatigue, I feel unmotivated and get physically exhausted easily. I am anxious that this condition will persist.
○ Current Situation:
Even though it's the start of the semester, my condition has worsened so much that the vice-principal, who was treated at this clinic, suggested I take sick leave and visit here.
If the cause of my illness is found and requires aggressive treatment, I am considering taking time off work.
○ Current Biggest Worry:
I am worried that if we can’t identify the cause, the symptoms will worsen, and it will affect my ability to care for my children.
○ Current Expectation:
I hope to find out what’s causing this and whether it’s treatable.
"""

    patient_prompt = [{
        'role': 'system',
        'content': f"""
        You are a medical expert tasked with generating a virtual patient scenario. 
        Your goal is to write a detailed case for a patient suffering from fatigue, and to describe the patient's personality and emotional state in detail. 
        The underlying disease is {selected_disease}, and the traditional medicine pattern is {selected_pattern}.

        Each virtual patient scenario must follow this structure:

        1. Case Summary
        2. Personality and Emotional State

        The content of each section is as follows:

        ### Case Summary
        - When the patient started experiencing fatigue
        - How the fatigue has progressed
        - The current state of fatigue and how it affects daily life
        - The reason why the patient visited the clinic

        ### Personality and Emotional State
        - Usual Personality: The patient’s typical personality traits (e.g., introverted, extroverted, positive, etc.)
        - Current Emotional State: How the patient currently feels due to the fatigue (e.g., lethargy, anxiety, etc.)
        - Current Situation: The patient’s current situation and how it affects their fatigue (e.g., work, family, etc.)
        - Current Biggest Worry: The problem the patient is most concerned about
        - Current Expectation: What the patient hopes to achieve through treatment

        Here is an example of a case summary and personality and emotional state for a virtual patient complaining of fatigue:
        {patient_example}

        Based on this format, write a case summary and the personality and emotional state for a virtual patient complaining of fatigue.
        Ensure that the content aligns with the example Q&A responses below: 
        {response_str}
        The response language should be "{target_language}". Even if the input is in a different language, write the response in {target_language}.
        """
    }]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=patient_prompt,
        temperature=0.5
    )

    return response.choices[0].message.content


def generate_situation_guidelines(clinical_phenotype, selected_disease, selected_pattern, target_language):
    # Example situation guideline

    situation_guide_prompt = [{
        'role': 'system',
        'content': f"""
        You are tasked with explaining a virtual patient scenario to a traditional medicine student.
        The virtual patient has been assigned the disease {selected_disease}, the pattern {selected_pattern}, and is presenting with the chief complaint of {clinical_phenotype}.
        While you will not provide the exact disease, pattern, or chief complaint in the answer, create a situation guideline for the patient that does not conflict with the assigned disease, pattern, or chief complaint.
        The entire response should be in {target_language}, and ensure that the patient’s name and cultural details are appropriate for a {target_language} context.
        Below is an example of a situation guideline. Keep the format the same, but adjust the content appropriately to avoid conflict with the disease, pattern, and chief complaint:
    
        - Situation Guidelines - 
        A (29)-year-old (female) (Minji Kim) came to the hospital complaining of ({clinical_phenotype}).

        [Vital Signs]
        ● Blood Pressure: 120/76 mmHg
        ● Pulse: 72 beats/min
        ● Respiration: 14 breaths/min
        ● Temperature: 36.2°C
        ● Weight: 54kg
        ● Height: 160cm

        The applicant should take the patient's medical history for the diagnosis of the disease and pattern and conduct appropriate examinations. 
        ※ Note: After entering "end," the disease and pattern must be inferred.
        """
    }]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=situation_guide_prompt,
        temperature=0.5
    )

    patient_condition = response.choices[0].message.content

    # Extract patient info
    extract_info_prompt = [{
        'role': 'system',
        'content': """
        You are an expert in analyzing text and extracting relevant information.
        I will provide you with a scenario guide for a virtual patient.
        Your task is to extract the patient's name, age, and gender from the provided text.
        Please return the extracted information in the following format:

        Name: [Name]
        Age: [Age]
        Gender: [Gender]

        Here is the scenario guide:
        """
    }, {
        'role': 'user',
        'content': patient_condition
    }]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=extract_info_prompt,
        temperature=0
    )

    extracted_info = response.choices[0].message.content

    return patient_condition, extracted_info

def save_patient_data(executor_name, patient_data):
    # Create folder name using current time and executor's name
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{current_time}_{executor_name}"

    # Create a new folder within the data folder
    data_folder = os.path.join("data", folder_name)
    os.makedirs(data_folder, exist_ok=True)

    # Save patient_data.json file
    patient_data_file_path = os.path.join(data_folder, "patient_data.json")
    with open(patient_data_file_path, 'w', encoding='utf-8') as patient_file:
        json.dump(patient_data, patient_file, ensure_ascii=False, indent=4)

    # Save the latest folder path in data/latest_folder_path.txt
    with open('data/latest_folder_path.txt', 'w', encoding='utf-8') as latest_file:
        latest_file.write(data_folder)

    print(f"Patient data saved in folder: {data_folder}")
    
    return data_folder  # Return folder path


def main():

    # Get user input
    clinical_phenotype = 'Fatigue'
    target_language = input("Target Language (e.g., English): ")
    executor_name = input("Executor name: ")

    # List of patterns and diseases
    valid_patterns = [
        "Qi and Blood Deficiency", "Spleen Qi Deficiency", "Liver and Kidney Yin Deficiency",
        "Spleen Deficiency with Dampness", "Heart and Spleen Yang Deficiency",
        "Liver Qi Stagnation and Spleen Deficiency", "Shaoyang Half-Exterior, Half-Interior"
    ]

    valid_diseases = [
        "Chronic Fatigue Syndrome", "Sleep Disorder", "Fibromyalgia",
        "Depression", "Hyperthyroidism"
    ]

    use_custom_selection = input("Would you like to select a specific disease and pattern? (y/n): ").strip().lower()

    if use_custom_selection == 'y':
        # Input for disease and pattern with validation loop
        while True:
            selected_disease = input("Enter the disease (e.g., Fibromyalgia): ").strip().lower()
            matched_disease = next((disease for disease in valid_diseases if disease.lower() == selected_disease), None)
            if not matched_disease:
                print(f"Invalid disease. Please select from: {', '.join(valid_diseases)}")
                continue

            selected_pattern = input("Enter the pattern (e.g., Spleen Qi Deficiency): ").strip().lower()
            matched_pattern = next((pattern for pattern in valid_patterns if pattern.lower() == selected_pattern), None)
            if not matched_pattern:
                print(f"Invalid pattern. Please select from: {', '.join(valid_patterns)}")
                continue

            # Check if the combination is valid
            if (matched_disease, matched_pattern) not in non_conflicting_combinations:
                print("Invalid combination of disease and pattern. Please try again.")
                # Display all valid combinations
                print("\nHere are the valid combinations of disease and pattern:")
                for disease, pattern in non_conflicting_combinations:
                    print(f"  - {disease} with {pattern}")
            else:
                selected_disease, selected_pattern = matched_disease, matched_pattern
                break

    else:
        selected_disease = None
        selected_pattern = None


    # Generate virtual patient data
    patient_data, selected_disease, selected_pattern = generate_virtual_patient_data(
        clinical_phenotype, target_language,
        selected_disease=selected_disease, selected_pattern=selected_pattern
    )

    # Save patient data to a JSON file and return the folder path
    folder_path = save_patient_data(executor_name, patient_data)

    # Add folder path to patient_data
    patient_data['folder_path'] = folder_path

    # Save the updated patient_data.json file
    with open(os.path.join(folder_path, 'patient_data.json'), 'w', encoding='utf-8') as f:
        json.dump(patient_data, f, ensure_ascii=False, indent=4)

    # Virtual patient data generation message
    print("""\nA virtual patient was created, which represents a clinical phenotype of %s
          with the underlying disease and syndrome being %s and %s, respectively."""
          % (clinical_phenotype, selected_pattern, selected_disease))    

    print("\nPatient situation guidelines:")
    print(patient_data['patient_condition'])


if __name__ == "__main__":
    main()
