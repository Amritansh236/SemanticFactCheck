from flask import Flask, render_template, request
import requests
from fuzzywuzzy import fuzz
from transformers import pipeline

HUGGINGFACE_API_KEY = "your_huggingface_api_key_here"
MODEL_NAME = "google/flan-t5-large"
GOOGLE_API_KEY = "your_google_api_key_here"
SEARCH_ENGINE_ID = "your_search_engine_id_here"

entailment_pipeline = pipeline("text-classification", model="roberta-large-mnli")

def get_google_search_results(topic):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"q": topic, "cx": SEARCH_ENGINE_ID, "key": GOOGLE_API_KEY, "num": 7}
    response = requests.get(url, params=params)
    return response.json().get("items", [])

def filter_relevant_results(search_results, topic):
    return [r for r in search_results if fuzz.partial_ratio(topic.lower(), r['title'].lower()) > 60]

def analyze_entailment(topic, articles):
    results = []
    for text in articles:
        if not isinstance(text, str) or not isinstance(topic, str):
            print("Error: Invalid input type. Expected string but got:", type(text), type(topic))
            return []

        result = entailment_pipeline(text, text_pair=topic)

        if not result:
            print("No entailment result returned")
            continue

        label = result[0].get('label', '').lower()
        score = result[0].get('score', 0)

        results.append((label, score))

    return results

def determine_final_verdict(scores):
    if not scores or not isinstance(scores, list):
        print(f"Error: Expected list but got: {type(scores)}")
        return "Error: Invalid score format"

    entailment_score = 0
    contradiction_score = 0

    for entry in scores:
        if isinstance(entry, tuple) and len(entry) == 2: 
            label, score = entry
            if label == "entailment":
                entailment_score += score
            elif label == "contradiction":
                contradiction_score += score

    if entailment_score > contradiction_score:
        return "Likely True"
    elif contradiction_score > entailment_score:
        return "Likely False"

    return "Insufficient Data"

def generate_correction(topic, snippets):
    prompt = (
        f"The following statement has been circulating: '{topic}'.\n\n"
        "Respond thoughtfully by first asking clarifying questions. Then, explain in simple and clear terms why the statement is misleading or incorrect. "
        "Back your correction with evidence from reliable sources, and finally, rewrite the claim accurately. Your explanation should be around 200 words. "
        "Please cite the sources you reference.\n\n"
        "Here are some related article snippets for context:\n"
        + "\n".join(f"- {s}" for s in snippets)
    )


    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "inputs": prompt,
        "parameters": {"max_new_tokens":250}
    }

    url = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
    response = requests.post(url, headers=headers, json=data)

    try:
        result = response.json()
        if isinstance(result, list) and result and "generated_text" in result[0]:
            return result[0]["generated_text"]
        else:
            print("⚠️ Unexpected HuggingFace response:", result)
            return "Could not generate correction. Try again later."
    except Exception as e:
        print("❌ Error parsing response from HuggingFace:", e)
        return "Correction generation failed."


def extract_spread_info(articles):
    platforms = []
    countries = []

    for article in articles:
        link = article["link"]
        snippet = article.get("snippet", "").lower()

        # Check domain for country
        if ".co.uk" in link: countries.append("UK")
        elif ".in" in link: countries.append("India")
        elif ".com.au" in link or ".au" in link: countries.append("Australia")
        elif ".ca" in link: countries.append("Canada")
        elif ".co.nz" in link or ".nz" in link: countries.append("New Zealand")
        elif ".ie" in link: countries.append("Ireland")
        elif ".za" in link: countries.append("South Africa")
        elif ".sg" in link: countries.append("Singapore")
        elif ".de" in link: countries.append("Germany")
        elif ".fr" in link: countries.append("France")
        elif ".it" in link: countries.append("Italy")
        elif ".es" in link: countries.append("Spain")
        elif ".nl" in link: countries.append("Netherlands")
        elif ".se" in link: countries.append("Sweden")
        elif ".no" in link: countries.append("Norway")
        elif ".fi" in link: countries.append("Finland")
        elif ".dk" in link: countries.append("Denmark")
        elif ".ch" in link: countries.append("Switzerland")
        elif ".be" in link: countries.append("Belgium")
        elif ".pl" in link: countries.append("Poland")
        elif ".cz" in link: countries.append("Czech Republic")
        elif ".at" in link: countries.append("Austria")
        elif ".gr" in link: countries.append("Greece")
        elif ".pt" in link: countries.append("Portugal")
        elif ".ru" in link: countries.append("Russia")
        elif ".jp" in link: countries.append("Japan")
        elif ".kr" in link: countries.append("South Korea")
        elif ".br" in link: countries.append("Brazil")
        elif ".mx" in link: countries.append("Mexico")
        elif ".ar" in link: countries.append("Argentina")
        elif ".cl" in link: countries.append("Chile")
        elif ".tr" in link: countries.append("Turkey")
        elif ".sa" in link: countries.append("Saudi Arabia")
        elif ".ae" in link: countries.append("UAE")
        elif ".eg" in link: countries.append("Egypt")
        elif ".ng" in link: countries.append("Nigeria")
        elif ".pk" in link: countries.append("Pakistan")
        elif ".bd" in link: countries.append("Bangladesh")
        elif ".my" in link: countries.append("Malaysia")
        elif ".ph" in link: countries.append("Philippines")
        elif ".vn" in link: countries.append("Vietnam")
        elif ".th" in link: countries.append("Thailand")
        elif ".hk" in link: countries.append("Hong Kong")
        elif ".tw" in link: countries.append("Taiwan")
        elif ".il" in link: countries.append("Israel")
        elif ".ir" in link: countries.append("Iran")
        elif ".ua" in link: countries.append("Ukraine")
        elif ".ro" in link: countries.append("Romania")
        elif ".hu" in link: countries.append("Hungary")
        elif ".sk" in link: countries.append("Slovakia")
        elif ".si" in link: countries.append("Slovenia")
        elif ".bg" in link: countries.append("Bulgaria")
        elif ".hr" in link: countries.append("Croatia")
        elif ".lt" in link: countries.append("Lithuania")
        elif ".lv" in link: countries.append("Latvia")
        elif ".ee" in link: countries.append("Estonia")
        elif ".by" in link: countries.append("Belarus")
        elif ".kz" in link: countries.append("Kazakhstan")
        elif ".am" in link: countries.append("Armenia")
        elif ".ge" in link: countries.append("Georgia")
        elif ".az" in link: countries.append("Azerbaijan")


        # Check snippet for platform mentions
        if "facebook" in snippet: platforms.append("Facebook")
        if "twitter" in snippet or "x.com" in snippet: platforms.append("Twitter")
        if "whatsapp" in snippet: platforms.append("WhatsApp")

    return list(set(countries)), list(set(platforms))



# Main fact-checking function
def fact_check(topic):
    articles = get_google_search_results(topic)
    
    if not articles:
        print("❌ No articles found. Check your search query or API.")
        return [], "No articles found."

    relevant_articles = filter_relevant_results(articles, topic)

    if not relevant_articles:
        print("❌ No relevant sources found.")
        return [], "No relevant articles found."

    # Use article snippets for entailment analysis
    snippets = [article['snippet'] for article in relevant_articles]
    scores = analyze_entailment(topic, snippets)
    verdict = determine_final_verdict(scores)

    # Prepare articles and verdict to return
    articles_info = [{"title": article["title"], "link": article["link"], "snippet": article.get("snippet", "")} for article in relevant_articles]
    return articles_info, verdict

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        topic = request.form["topic"]
        articles, verdict = fact_check(topic)

        snippets = [a.get('snippet', '') for a in articles]
        correction = generate_correction(topic, snippets)
        countries, platforms = extract_spread_info(articles)

        return render_template(
            "index.html",
            topic=topic,
            articles=articles,
            verdict=verdict,
            correction=correction,
            countries=countries,
            platforms=platforms
        )

    return render_template("index.html", topic=None, articles=None, verdict=None, correction=None, countries=None, platforms=None)

if __name__ == "__main__":
    app.run(debug=True)
