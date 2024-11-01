# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 13:51:12 2024

@author: wonyu
"""

from openai import OpenAI
import os
import json
import pandas as pd
import time
import openai
from io import StringIO  # StringIO를 import합니다.

# OpenAI API KEY set

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=openai_api_key   # Use your environment variable or directly insert your API key here
)


def evaluate_student_performance(folder_path):
    # Load conversation content
    conversation_file_path = os.path.join(folder_path, 'conversation.json')
    with open(conversation_file_path, 'r', encoding='utf-8') as f:
        question_dict = json.load(f)

    # Load patient data
    patient_data_file_path = os.path.join(folder_path, 'patient_data.json')
    with open(patient_data_file_path, 'r', encoding='utf-8') as f:
        patient_data = json.load(f)

    target_language = patient_data['target_language']
    selected_disease = patient_data['selected_disease']
    selected_pattern = patient_data['selected_pattern']

    # Restore DataFrame (using StringIO)
    subset_disease = pd.read_json(StringIO(patient_data['subset_disease']), orient='split')
    subset_pattern = pd.read_json(StringIO(patient_data['subset_pattern']), orient='split')

    # Get student's inference input
    if target_language.lower() == "korean":
        inference_disease = input("""Based on the given history taking,
    please infer the likely diagnosis and provide its reasoning:\n""")
        inference_pattern = input("""\n\nBased on the given history taking,
    please deduce the likely syndrome pattern and provide its reasoning:\n""")
        print("\n\nYour answer has been submitted. Please wait a moment.")
    else:
        inference_disease = input("""Based on the history taking,
    infer the likely disease and the basis for your conclusion:\n""")
        inference_pattern = input("""Based on the history taking,
    deduce the likely pattern (證) and the basis for its judgment:\n""")
        print("\n\nYour answer is received. Please wait a moment for your evaluation.")
        
    # Load evaluation data
    fatigue_evaluation = pd.read_csv('dataset_fatigue_reasoning.csv')

    evaluation_answer_disease = fatigue_evaluation['Explanation'].loc[fatigue_evaluation['Name'] == selected_disease].values[0]
    evaluation_answer_pattern = fatigue_evaluation['Explanation'].loc[fatigue_evaluation['Name'] == selected_pattern].values[0]

    # Inference evaluation
    score_and_feedback_disease = evaluate_inference(evaluation_answer_disease, inference_disease, selected_disease, target_language)
    score_and_feedback_pattern = evaluate_inference(evaluation_answer_pattern, inference_pattern, selected_pattern, target_language)
    
    print("\n\n")
    print("Qualitative Evaluation on Disease Diagnosis:")
    print(score_and_feedback_disease)
    print("\n\n")
    print("Qualitative Evaluation on Pattern Identification (辨證):")
    print(score_and_feedback_pattern)
    
    # Perform quantitative evaluation
    evaluation_result = perform_quantitative_evaluation(question_dict, subset_disease, subset_pattern, target_language, selected_disease, selected_pattern)

    # Save evaluation log
    evaluation_log = {
        "disease_evaluation": score_and_feedback_disease,
        "pattern_evaluation": score_and_feedback_pattern,
        "quantitative_evaluation": evaluation_result
    }
    
    evaluation_log_file_path = os.path.join(folder_path, 'evaluation_log.json')
    with open(evaluation_log_file_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_log, f, ensure_ascii=False, indent=4)

    print(f"Evaluation log saved in folder: {folder_path}")

def evaluate_inference(evaluation_contents, student_answer, answer, language):
    messages = [
        {"role": "system", "content": "You are an evaluator for academic reports on traditional Korean medicine."},
        {"role": "user", "content": f"""As an expert in Korean medicine, you are evaluating a student's response based on the interview results of a simulated patient.
The patient’s disease/syndrome is {answer}, and an example answer for inference is as follows:

{evaluation_contents}

The student’s submitted response is as follows:
{student_answer}

Evaluation criteria are as follows:
- Points awarded if disease/syndrome content aligns with or is similar to the correct answer based on Korean medicine principles.
- Points awarded if symptoms that support the disease/syndrome are described accurately.
- Points awarded if the disease/syndrome is correctly deduced and reasoning is provided.
- Deductions if logical connections are weak.
- Deductions for redundant, formalistic, or inaccurate explanations.

Based on these criteria, provide a score between 0 and 5, written as an integer along with a brief explanation in "{language}".
Regardless of the prompt language, respond in "{language}".

Indicate the score out of 5, e.g., 3/5.
"""}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=4000,
        n=1,
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()

def perform_quantitative_evaluation(question_dict, subset_disease, subset_pattern, target_language, selected_disease, selected_pattern):
    # Prepare keywords
    keyword_disease = subset_disease[['Question', selected_disease]].rename(columns={selected_disease: 'answer keyword'}).dropna()
    keyword_disease_dict = keyword_disease.set_index('Question')['answer keyword'].to_dict()

    keyword_pattern = subset_pattern[['Question', selected_pattern]].rename(columns={selected_pattern: 'answer keyword'}).dropna()
    keyword_pattern_dict = keyword_pattern.set_index('Question')['answer keyword'].to_dict()

    Quant_evaluation_prompt = f"""
You are an expert in traditional Korean medicine (TKM). 
Your task is to evaluate the completeness and accuracy of the following conversation between a simulated patient and a student TKM practitioner. 
The goal is to determine if the practitioner asked all necessary questions to accurately identify the underlying disease and syndrome differentiation based on the patient's symptoms.

Conversation:
{question_dict}

Ideal question and answer set for disease:
{keyword_disease_dict}

Criteria for disease evaluation:
1. Overall score: Mark the number of correct answers relative to the total (e.g., 2/3)

2. Performed vs Missed questions: Briefly summarize which ideal questions were asked and which were missed, focusing on keywords. 
    (Example: Performed Q&A: drowsiness, depression inquiry / Improvement of symptoms after sleeping
            Missed Q&A: worsening of depression)

Ideal question and answer set for pattern differentiation:
{keyword_pattern_dict}

Criteria for pattern differentiation evaluation:
1. Overall score: Mark the number of correct answers relative to the total (e.g., 2/3)

2. Performed vs Missed questions: Briefly summarize which ideal questions were asked and which were missed, focusing on keywords. 
    (Example: Performed Q&A: drowsiness, depression inquiry / Improvement of symptoms after sleeping
            Missed Q&A: worsening of depression)

The response language is {target_language}. Regardless of prompt language, respond only in {target_language}. 

"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": Quant_evaluation_prompt}],
        temperature=0.2,
    )

    evaluation = response.choices[0].message.content
    print("\nEvaluation of qualitative diag:")
    print(evaluation)
    return evaluation  # Returns evaluation result


def main():
    # 폴더 경로를 자동으로 patient_data.json에서 가져오기
    with open('data/latest_folder_path.txt', 'r', encoding='utf-8') as f:
        folder_path = f.read().strip()

    evaluate_student_performance(folder_path)

if __name__ == "__main__":
    main()
