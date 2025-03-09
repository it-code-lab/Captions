from flask import Flask, request, jsonify
import json

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load word timestamps
@app.route('/get_word_timestamps', methods=['GET'])
def get_word_timestamps():
    try:
        with open("temp/word_timestamps.json", "r") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Save updated word timestamps
@app.route('/save_word_timestamps', methods=['POST'])
def save_word_timestamps():
    try:
        data = request.json
        with open("temp/word_timestamps.json", "w") as f:
            json.dump(data, f, indent=4)
        return jsonify({"message": "✅ Word timestamps updated successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Load structured_output.json (for headings & list items)
@app.route('/get_structured_output', methods=['GET'])
def get_structured_output():
    try:
        with open("temp/structured_output.json", "r") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Run the server on port 5000
