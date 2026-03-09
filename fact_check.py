import requests

API_KEY = "AIzaSyDrUzhbT6CjFG6GA7Hh2G5FNledMb4xW68"

def verify_fact(claim_text):
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

    params = {
        "query": claim_text,
        "key": API_KEY,
        "languageCode": "en"
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "claims" not in data:
        return None

    claim = data["claims"][0]
    review = claim["claimReview"][0]

    return {
        "claim": claim["text"],
        "publisher": review["publisher"]["name"],
        "rating": review["textualRating"],
        "review_url": review["url"]
    }