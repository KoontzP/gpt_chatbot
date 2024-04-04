import os
import subprocess

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


def send_prompt(prompt):
    # Modify the command to run the script
    cmd = 'python gpt_chatbot.py -batch_size 128'

    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,
                               text=True, env=env)

    # Print the prompt
    print("Prompt:", prompt)

    # Communicate with subprocess
    try:
        stdout, stderr = process.communicate(input=prompt + "\n")
    except EOFError:
        stdout, stderr = "", "EOFError: EOF when reading a line\n"

    # Check if stdout is not None
    stdout = stdout.strip() if stdout is not None else ""

    # Extract text after "Chatbot:" line
    chatbot_index = stdout.find("Chatbot:")
    captured_output = stdout[chatbot_index + len("Chatbot:"):] if chatbot_index != -1 else ""

    print("Captured output:", captured_output)  # Print the captured output
    return captured_output.strip()  # Return the extracted text, stripped of leading/trailing whitespace



@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        prompt = request.form['prompt']
        response = send_prompt(prompt)
        return jsonify({'response': response})
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
