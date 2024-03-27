from flask import Flask, render_template, request, jsonify
import joblib
from flask_cors import CORS


# print(model.predictFromComment('Almost good, but the wheel is so bad, it stopped just 2 months after buying.'))
# Khởi tạo ứng dụng Flask
app = Flask(__name__)
CORS(app)  # Tạo đối tượng CORS cho ứng dụng Flask


# Định nghĩa route cho form
@app.route('/server', methods=['POST', 'GET'])
def process():
    model = joblib.load('model/model.pkl')
    data = request.get_json()
    
    if data:
        message = data['message']
        reply = model.predictFromComment(message)
        print(message)
        print(reply)
        return jsonify({'message': f'{reply}'})  
    


if __name__ == '__main__':
    # Chạy ứng dụng trên cổng 5000
    app.run(debug=True)