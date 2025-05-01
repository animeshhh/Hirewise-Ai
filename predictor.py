import pickle
import numpy as np

# Load pre-trained models
def load_models():
    global gb, lr, rf, svm, ensemble_model
    with open('gb_model.pkl', 'rb') as f:
        gb = pickle.load(f)
    with open('lr_model.pkl', 'rb') as f:
        lr = pickle.load(f)
    with open('rf_model.pkl', 'rb') as f:
        rf = pickle.load(f)
    with open('svm_model.pkl', 'rb') as f:
        svm = pickle.load(f)
    with open('ensemble_model.pkl', 'rb') as f:
        ensemble_model = pickle.load(f)

load_models()

def extract_features(resume_data):
    """
    Extracts 7 features from the parsed resume data.
    These features must match what the model was trained on.
    """
    features = []

    # 1. Number of education entries
    edu_count = len(resume_data.get("education", []))
    features.append(edu_count)

    # 2. Number of experience entries
    exp_count = len(resume_data.get("experience", []))
    features.append(exp_count)

    # 3. Number of technical skills
    tech_keywords = ['python', 'java', 'machine learning', 'sql', 'tableau']
    resume_skills = [skill.lower() for skill in resume_data.get("skills", [])]
    tech_skill_count = sum(1 for skill in resume_skills if skill in tech_keywords)
    features.append(tech_skill_count)

    # 4. Number of soft skills
    soft_keywords = ['communication', 'leadership', 'teamwork', 'adaptability']
    soft_skill_count = sum(1 for skill in resume_skills if skill in soft_keywords)
    features.append(soft_skill_count)

    # 5. Total skills (tech + soft)
    total_skills = tech_skill_count + soft_skill_count
    features.append(total_skills)

    # 6. Total experience (in years)
    total_exp = resume_data.get("total_experience", 0)
    try:
        total_exp = float(total_exp)
    except (TypeError, ValueError):
        total_exp = 0.0
    features.append(total_exp)

    # 7. Certification flag
    certs_flag = 1 if resume_data.get("certifications") else 0
    features.append(certs_flag)

    return np.array(features).reshape(1, -1)

def predict_performance(features):
    """
    Predicts performance score using all models and returns results.
    """
    gb_score = gb.predict(features)[0]
    lr_score = lr.predict(features)[0]
    rf_score = rf.predict(features)[0]
    svm_score = svm.predict(features)[0]
    ensemble_score = ensemble_model.predict(features)[0]

    scores = {
        "GradientBoosting": round(float(gb_score)),
        "LogisticRegression": round(float(lr_score)),
        "RandomForest": round(float(rf_score)),
        "SVM": round(float(svm_score)),
        "EnsembleModel": round(float(ensemble_score))
    }

    results = {
        "scores": scores,
        "best_model": max(scores, key=lambda k: scores[k])
    }

    return results
