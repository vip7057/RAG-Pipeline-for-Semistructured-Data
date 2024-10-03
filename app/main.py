from flask import Flask, request, jsonify, render_template

from app.chatbot import get_response

app = Flask(__name__)

# Define the root route
@app.route('/')
def home():
    return render_template('index.html')  # Ensure you have this HTML file in the templates folder

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    response = get_response(user_input)  # Ensure this function is defined correctly
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
