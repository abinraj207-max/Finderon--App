import pickle

model = pickle.load(open("model/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

def predict_news(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]

    confidence = max(probability) * 100

    if prediction == 1:
        result = "Real"
    else:
        result = "Fake"

    return {
        "result": result,
        "confidence": round(confidence, 2)
    }