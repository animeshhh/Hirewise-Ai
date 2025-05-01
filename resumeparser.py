import requests
import yaml

# Load Groq API key from config.yaml
CONFIG_PATH = r"config.yaml"
with open(CONFIG_PATH) as file:
    data = yaml.load(file, Loader=yaml.FullLoader)
    grok_api_key = data['GROK_API_KEY']

def ats_extractor(resume_data):
    # Define the Groq API endpoint
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    # Set request headers
    headers = {
        "Authorization": f"Bearer {grok_api_key}",
        "Content-Type": "application/json"
    }

    # ✅ Updated Prompt to Match ML Model's Feature Expectations
    prompt = '''
You are an AI assistant designed to extract structured information from resumes.
Parse the following resume text and return the result in **valid JSON** with the following fields only:

{
  "education": [list of degrees, institutions, or education-related phrases],
  "experience": [list of job roles, companies, or internship references],
  "skills": [list of technical and soft skills combined],
  "total_experience": estimated number of years of total experience (as a number),
  "certifications": [list of certifications if mentioned]
}

⚠️ Important: Only respond with valid JSON. Do not add comments, text, or explanations outside the JSON.
'''

    # Construct the conversation
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": resume_data}
    ]

    # Groq API request payload
    data = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 1500
    }

    # Call the Groq API
    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        try:
            content = response.json()['choices'][0]['message']['content']
            return content
        except Exception as e:
            print("❌ Error parsing response:", e)
            return "{}"
    else:
        print("❌ Groq API call failed")
        print("Status code:", response.status_code)
        print("Details:", response.text)
        return "{}"
