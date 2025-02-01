import os
from flask import Flask, render_template, request, jsonify
from t3 import ChurnPredictor  

app = Flask(__name__)


UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        
        predictor = ChurnPredictor()
        churn_rate = predictor.analyze_dataset(file_path)


        
        
        churn_users_count = 415  
        churn_user_ids = [2164, 2178, 2181, 2182]  
        
        return jsonify({
            'churn_rate': f"{churn_rate:.2f}%",
            'churn_users_count': churn_users_count,
            'churn_user_ids': churn_user_ids,
            'message': 'Analysis complete! You can now view the results.'
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
