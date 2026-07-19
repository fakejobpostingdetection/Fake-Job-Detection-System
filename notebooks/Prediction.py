import joblib

model = joblib.load("models/best_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

print("="*50)
print("FAKE JOB DETECTION")
print("="*50)

title = input("Job Title: ")
company = input("Company Profile: ")
description = input("Job Description: ")
requirements = input("Requirements: ")
benefits = input("Benefits: ")

combined_text = (
    title + " " +
    company + " " +
    description + " " +
    requirements + " " +
    benefits
)

job_vector = vectorizer.transform([combined_text])

prediction = model.predict(job_vector)[0]

print("="*50)

if prediction == 1:
    print("Prediction : FAKE JOB")
else:
    print("Prediction : GENUINE JOB")

print("="*50)