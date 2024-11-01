# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 13:50:31 2024

@author: wonyu
"""

# dialogue_execution.py


from openai import OpenAI
import os
import json
from datetime import datetime

# OpenAI API key set
openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=openai_api_key   # Use your environment variable or directly insert your API key here
)



def save_conversation(folder_path, question_dict):
    # conversation.json 파일을 folder_path에 저장
    conversation_file_path = os.path.join(folder_path, "conversation.json")
    with open(conversation_file_path, 'w', encoding='utf-8') as f:
        json.dump(question_dict, f, ensure_ascii=False, indent=4)

    print(f"Conversation saved in folder: {folder_path}")


def simulate_dialogue(patient_data):
    # 필요한 데이터 로드
    chat_log = patient_data['chat_log']
    patient_summary_personality = patient_data['patient_summary_personality']
    patient_condition = patient_data['patient_condition']
    extracted_info = patient_data['extracted_info']
    target_language = patient_data['target_language']
    selected_disease = patient_data['selected_disease']
    folder_path = patient_data['folder_path']  # 폴더 경로 불러오기
    

    # 시뮬레이션 메시지 준비
    message_sim_patient = [{
        'role': 'system',
        'content': f"""
        You are a patient visiting a Korean medicine clinic with the condition of {selected_disease}. Please play only the role of the patient and refrain from asking or offering help to the questioner.
Specifically, if the questioner greets you with something like “Hello,” do not respond with something like “Hello, what seems to be the problem?” Instead, respond purely from the patient’s perspective.
If you encounter a question that is very odd for a patient to answer, simply respond that you don’t know.

Please keep your responses short, concise, and clear. Avoid providing detailed answers to information that hasn’t been asked. Do not proactively explain your information or answer unasked questions, especially about the specified condition or diagnosis.
If you are asked about the condition or diagnosis, say that you don’t know and that you’ve come here to find out.
For questions unrelated to the provided patient information, answer appropriately as the patient would, but avoid answers that could lead to misdiagnosis.

The language of response is {target_language}. Regardless of the language used by the questioner, respond exclusively in {target_language}.
Your name, age, and gender are as follows: {extracted_info}
A summary of your case, personality, and emotional state is as follows: {patient_summary_personality}
        """
    }]

    message_sim_patient.extend(chat_log)

    question_dict = {}

    # 대화 시뮬레이션
    print("""\n=== Start conversation with patients. ===
           (Type 'End' to end the conversation)
                    \n""")

    while True:
        question = input("Q: ")
        if question.strip().lower() == "end":
            break

        message_sim_patient.append({'role': 'user', 'content': question})

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=message_sim_patient,
            temperature=0.1
        )

        answer_original = response.choices[0].message.content
        print('A: ' + answer_original)

        message_sim_patient.append({'role': 'assistant', 'content': answer_original})
        question_dict[question] = answer_original

    # 대화 내용 저장
    save_conversation(folder_path, question_dict)


    print("\nThe conversation has ended. Proceed to the next step..")

def main():
    # 최신 폴더 경로 불러오기
    with open('data/latest_folder_path.txt', 'r', encoding='utf-8') as f:
        folder_path = f.read().strip()

    patient_data_file_path = os.path.join(folder_path, 'patient_data.json')

    # 환자 데이터 로드
    with open(patient_data_file_path, 'r', encoding='utf-8') as f:
        patient_data = json.load(f)

    simulate_dialogue(patient_data)

if __name__ == "__main__":
    main()
