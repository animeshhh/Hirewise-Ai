# Hirewise-Ai
AI-powered Resume Parser &amp; Performance Predictor – A Flask-based web application that extracts structured data from PDF resumes using NLP and Groq API, and predicts candidate performance scores using multiple ML regression models. Includes resume quality insights and a Tailwind CSS frontend.

### Objective

Creating a Resume Parser Web App using Flask is an innovative way to help job seekers evaluate the ATS (Applicant Tracking System) compatibility of their resumes. This application allows users to upload resumes in PDF format, which are then parsed to extract key information such as:

-Full name

-Email address

-GitHub and LinkedIn profiles

-Employment history

-Technical and soft skills

The extracted data is displayed in structured JSON format, offering users actionable insights into how well their resumes align with automated recruitment systems.

The app leverages powerful tools and libraries, including Python, Flask, pdfminer.six, docx2txt, Pyresparser, and NLP frameworks like NLTK and spaCy, to process and extract content from PDF and DOCX resumes efficiently.

With recruitment processes increasingly relying on automation, this app provides a valuable solution by enabling job seekers to analyze and improve their resumes for better ATS performance, increasing their chances of getting noticed by employers.

### Sneak Peak of the App
![image](https://github.com/pik1989/Resume-Parser-OpenAI/assets/34673684/5d206207-1b25-4dbe-8e11-add701b632e7)

#### Overview: 
This App is created for job seekers to test whether their resumes are ATS friendly or not, if our App is able to parse your details and show it, then assume that everything is good.

#### Features: 
Ability to extract specific information from resumes, the use of JSON format for presenting the extracted data, and the integration of various libraries and tools for parsing resumes.

#### Installation: 
Run the pip install requirements.txt to install and set up the app, including any dependencies and prerequisites.

#### Usage: 
Just upload your resume in pdf format, and see for yourself :)


##### Running the program

1. Clone the repository to your local machine
2. Navigate to the project directory
3. Install all the required libraries
4. Provide your Open AI API key in the .yaml file
5. Run the following command to start the chatbot -

    ```
    python app.py
    ```

    ```
    Go to: https://localhost:8000
    ```
    
Overall, the development of a resume parser app using Flask represents a significant advancement in leveraging technology to support job seekers in optimizing their resumes for the modern recruitment landscape. This app aligns with the increasing demand for efficient and technology-driven solutions in the job application process, ultimately benefiting both job seekers and recruiters.
