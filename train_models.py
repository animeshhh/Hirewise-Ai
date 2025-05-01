import pandas as pd
import numpy as np
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# ✅ Load your dataset
df = pd.read_csv('/Users/animeshpandey/Downloads/archive/Resume/Resume.csv')


np.random.seed(42)
df['performance_score'] = np.random.randint(1, 101, size=len(df))

# ✅ Feature extraction (7 features as per predictor.py)
def extract_features(resume):
    text = resume['Resume_str'].lower()

    edu_keywords = ['bachelor', 'master', 'b.e', 'mca', 'btech', 'mba']
    exp_keywords = ['experience', 'worked', 'project', 'internship']
    tech_skill_keywords = ['python', 'java', 'machine learning', 'sql', 'tableau']
    soft_skill_keywords = ['communication', 'leadership', 'teamwork', 'adaptability']
    cert_keywords = ['certified', 'certification', 'certificate']

    edu_count = sum(1 for word in edu_keywords if word in text)
    exp_count = sum(1 for word in exp_keywords if word in text)
    tech_skill_count = sum(1 for word in tech_skill_keywords if word in text)
    soft_skill_count = sum(1 for word in soft_skill_keywords if word in text)
    total_skills = tech_skill_count + soft_skill_count
    certs_flag = 1 if any(word in text for word in cert_keywords) else 0

    # Total experience (mocked from text)
    exp_match = re.search(r'(\d+)\+?\s+years?', text)
    total_exp = int(exp_match.group(1)) if exp_match else 0

    return [
        edu_count,
        exp_count,
        tech_skill_count,
        soft_skill_count,
        total_skills,
        total_exp,
        certs_flag
    ]

# ✅ Extract features and target
X = df.apply(extract_features, axis=1).tolist()
y = df['performance_score'].values

X = np.array(X)
y = np.array(y)

# ✅ Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train models
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

gb_model = GradientBoostingRegressor()
gb_model.fit(X_train, y_train)
with open('gb_model.pkl', 'wb') as f:
    pickle.dump(gb_model, f)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
with open('lr_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

svm_model = SVR()
svm_model.fit(X_train, y_train)
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)

# ✅ Ensemble model
ensemble_model = VotingRegressor(
    estimators=[
        ('rf', rf_model),
        ('gb', gb_model),
        ('lr', lr_model),
        ('svm', svm_model)
    ]
)
ensemble_model.fit(X_train, y_train)
with open('ensemble_model.pkl', 'wb') as f:
    pickle.dump(ensemble_model, f)

# ✅ Evaluation
y_pred = ensemble_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Training complete!")
print(f"🔍 MSE: {mse:.2f}")
print(f"📈 R² Score: {r2:.2f}")
