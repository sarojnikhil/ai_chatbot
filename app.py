from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from gtts import gTTS
import os
import tempfile
import sqlite3
from deep_translator import GoogleTranslator

# Set up Google API Key

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    google_api_key=GOOGLE_API_KEY
)
app = Flask(__name__)



# Set up SQLite Database
conn = sqlite3.connect("mistakes.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("DROP TABLE IF EXISTS mistakes")
cursor.execute('''
CREATE TABLE IF NOT EXISTS mistakes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_sentence TEXT,
    mistake TEXT,
    correction TEXT
)
''')
conn.commit()

# Language Codes
LANGUAGE_CODES = {
    "afrikaans": "af", "arabic": "ar", "bulgarian": "bg", "bengali": "bn",
    "catalan": "ca", "czech": "cs", "danish": "da", "german": "de",
    "greek": "el", "english": "en", "spanish": "es", "french": "fr",
    "gujarati": "gu", "hindi": "hi", "croatian": "hr", "hungarian": "hu",
    "indonesian": "id", "italian": "it", "japanese": "ja", "kannada": "kn",
    "korean": "ko", "latin": "la", "lithuanian": "lt", "malayalam": "ml",
    "marathi": "mr", "malay": "ms", "nepali": "ne", "dutch": "nl",
    "norwegian": "no", "polish": "pl", "portuguese": "pt", "romanian": "ro",
    "russian": "ru", "slovak": "sk", "albanian": "sq", "serbian": "sr",
    "swedish": "sv", "swahili": "sw", "tamil": "ta", "telugu": "te",
    "thai": "th", "turkish": "tr", "ukrainian": "uk", "urdu": "ur",
    "vietnamese": "vi", "chinese (simplified)": "zh-CN"
}

def get_language_code(language_name):
    return LANGUAGE_CODES.get(language_name.lower(), "en")

@app.route("/")
def home():
    return render_template("index.html")

# Chatbot API
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")
    proficiency = data.get("proficiency", "Beginner")
    language = data.get("language", "English")

    user_language_code = get_language_code(language)
    
    prompt = f"Teach a {proficiency} learner a {language} conversation. Respond clearly: {user_input}"
    ai_response = llm.predict(prompt)

    translated_text = GoogleTranslator(source=user_language_code, target="en").translate(ai_response)

    return jsonify({"response": ai_response, "translation": translated_text})

def clean_text(text):
    """Cleans text by removing unwanted characters before text-to-speech processing."""
    return text.replace("*", "").replace("_", "").replace("\n", " ")  # Remove formatting

@app.route("/tts", methods=["POST"])
def text_to_speech():
    data = request.json
    text = clean_text(data.get("text", ""))
    lang = data.get("lang", "en")
    slow = data.get("slow", False)

    try:
        tts = gTTS(text=text, lang=lang, slow=slow)
        filename = "static/audio/tts_output.mp3"  # Save in a static folder
        tts.save(filename)
        return jsonify({"audio": filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# Mistake Analysis API
@app.route("/mistakes", methods=["POST"])
def analyze_mistakes():
    data = request.json
    user_sentence = data.get("sentence", "")

    correction_prompt = f"Check for grammar and vocabulary mistakes in this sentence: '{user_sentence}'. Provide corrections."
    corrected_sentence = llm.predict(correction_prompt)

    cursor.execute("INSERT INTO mistakes (user_sentence, mistake, correction) VALUES (?, ?, ?)",
                   (user_sentence, user_sentence, corrected_sentence))
    conn.commit()

    return jsonify({"correction": corrected_sentence})

# Generate Feedback API
@app.route("/feedback", methods=["GET"])
def generate_feedback():
    cursor.execute("SELECT mistake, correction FROM mistakes")
    mistakes_data = cursor.fetchall()

    if not mistakes_data:
        return jsonify({"feedback": "No mistakes recorded yet."})

    feedback_prompt = f"""
    Analyze these language mistakes and provide feedback in this format:
    
    1️⃣ **Categorize mistakes** (e.g., verb tense, article usage, gender agreement)
    2️⃣ **Provide example sentences** to clarify corrections
    3️⃣ **Include relevant grammar explanation links** for further learning
    
    Mistakes and corrections:
    {mistakes_data}
    
    Format response as:
    📌 **Mistake Type:** <Category>
    ❌ **Incorrect Sentence:** <Original mistake>
    ✅ **Correction:** <Corrected sentence>
    📝 **Example Sentence:** <Extra example>
    📚 **Learn More:** <Link to grammar guide>
    """
    
    feedback = llm.predict(feedback_prompt)

    return jsonify({"feedback": feedback})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

