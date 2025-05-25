from flask import Flask, render_template, request, jsonify
from bookbot import get_openai_response, analyze_image, recommend_books, recognize_speech, synthesize_speech

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    response = None
    if request.method == "POST":
        text_input = request.form.get("text_input")
        image_file = request.files.get("image_input")

        img_text = analyze_image(image_file) if image_file else ""
        full_input = f"{text_input} {img_text}".strip()

        matches = recommend_books(full_input)
        prompt = f"The user input is: {full_input}\nfind similar books:\n{matches}\nSuggest top 3 books."
        response = get_openai_response(prompt)
        synthesize_speech(response)

    return render_template("index.html", response=response)

@app.route("/speech_to_text", methods=["POST"])
def speech_to_text():
    text = recognize_speech()
    return jsonify({"text": text})

if __name__ == "__main__":
    app.run(debug=True)
