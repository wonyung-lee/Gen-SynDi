Gen-SynDi: A Framework for Fatigue Diagnosis and Syndrome Differentiation in Traditional Medicine

Welcome to Gen-SynDi, an open-source framework for diagnosing fatigue-related conditions and their syndrome differentiation in Traditional Asian Medicine. Gen-SynDi leverages generative AI to create virtual patients, simulate history-taking, and provide clinical reasoning feedback. This repository is designed to assist both educators and students in learning and exploring syndrome differentiation through interactive patient scenarios.
Overview of Gen-SynDi
Gen-SynDi consists of three primary modules:

Virtual Patient Generation: Using Gen_SynDi_1_generate_virtual_patient.py, you can generate a simulated patient profile. This includes both disease and syndrome differentiation that reflect realistic cases based on common fatigue-associated conditions such as Chronic Fatigue Syndrome, Sleep Disorder, Fibromyalgia, and more. The module constructs the patient's clinical scenario by combining symptoms with traditional medicine syndromes, helping students understand complex patient presentations.
Dialogue Execution (History Taking): The Gen_SynDi_2_dialogue_execution.py script allows users to interact with the generated virtual patient. The script simulates a realistic history-taking session where the AI-driven patient responds naturally, adhering strictly to the generated disease and syndrome context. It aims to train students to gather relevant information effectively, focusing on clear and concise questions and patient responses.
Clinical Reasoning Evaluation: With Gen_SynDi_3_evaluation.py, students' clinical reasoning is evaluated based on their history-taking performance. This module provides both qualitative feedback on the accuracy and coherence of their differential diagnosis and syndrome differentiation, as well as a quantitative assessment that highlights missed opportunities and potential areas for improvement.

Getting Started
To get started, clone this repository and follow these instructions:
Install the required dependencies:
pip install openai==1.47.0 pandas numpy

The specific version of the openai module required is 1.47.0 to ensure compatibility with the scripts.
2. Make sure you have your OpenAI API key set up as an environment variable (OPENAI_API_KEY).

Modules in Detail

1. Virtual Patient Generation
File: Gen_SynDi_1_generate_virtual_patient.py
Description: This script generates a virtual patient scenario by selecting combinations of diseases and traditional medicine patterns. The generated patient includes detailed symptom descriptions and a personality profile to enhance the realism of the interaction.

2. History Taking with Dialogue Execution
File: Gen_SynDi_2_dialogue_execution.py
Description: This script allows users to conduct a history-taking interview with the virtual patient. The patient responds naturally based on predefined guidelines, ensuring the interaction remains consistent with the simulated disease and syndrome differentiation.

3. Evaluation Module
File: Gen_SynDi_3_evaluation.py
Description: The evaluation script provides comprehensive feedback on the student's performance. It assesses the relevance and accuracy of their questions, their ability to extract key information, and the quality of their inferred diagnosis and syndrome differentiation.

Running the Entire Process
To execute the entire Gen-SynDi process on Ubuntu, use the following commands (note that Python is pre-installed and the 'python' prefix is not necessary):
set OPENAI_API_KEY="Your OpenAI API KEY"
Gen_SynDi_1_generate_virtual_patient.py
Gen_SynDi_2_dialogue_execution.py
Gen_SynDi_3_evaluation.py
pause

If you are using Windows and Python is installed, you can execute the process by running the following commands:
set OPENAI_API_KEY="Your OpenAI API KEY"
python Gen_SynDi_1_generate_virtual_patient.py
python Gen_SynDi_2_dialogue_execution.py
python Gen_SynDi_3_evaluation.py
pause

This sequence will guide you through generating a virtual patient, conducting a history-taking session, and evaluating the diagnostic skills. Each step is designed to help users understand and practice syndrome differentiation in a systematic manner.

Acknowledgments
Gen-SynDi was developed by Wonyung Lee to improve syndrome differentiation education in traditional medicine using cutting-edge generative AI techniques. Special thanks to all contributors and collaborators who provided invaluable support.
