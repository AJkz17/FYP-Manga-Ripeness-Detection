import os
import base64
import csv
import cv2
import joblib
import numpy as np
import google.generativeai as genai
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.utils import secure_filename
from models import db, DetectionRecord, User
from flask import session
from io import StringIO
from flask import make_response
from ml_integration import predict_ripeness



app = Flask(__name__)

# --- Configuration ---
app.config['SECRET_KEY'] = 'your_secret_key_here'


# 1. Get the base directory
basedir = os.path.abspath(os.path.dirname(__file__))

# --- Database Setup ---
instance_path = os.path.join(basedir, 'instance')
if not os.path.exists(instance_path):
    os.makedirs(instance_path)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(instance_path, 'mango.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- Upload Folder Setup ---
UPLOAD_FOLDER = os.path.join(basedir, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize DB connection
db.init_app(app)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- LOAD HARVEST MODEL ---
harvest_model_path = os.path.join(basedir, 'harvest_model_simple.pkl')
try:
    harvest_model = joblib.load(harvest_model_path)
    print("Harvest Regression Model loaded successfully!")
except FileNotFoundError:
    print("WARNING: 'harvest_model_simple.pkl' not found. Prediction will fail.")
    harvest_model = None

# --- Helper Functions ---

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_handling_suggestion(label):
    label = str(label).lower()
    if "unripe" in label:
        return "Keep at room temperature to allow it to ripen."
    elif "overripe" in label:
        return "Best used for smoothies or baking. Consume immediately."
    elif "ripe" in label:
        return "Ready to eat! Store in refrigerator."
    else:
        return "No specific instructions."

# --- Routes ---

# @app.route('/')
# def index():
#     return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    result = None
    filename = None
    detections_list = []
    
    user_temp = 28.0
    user_humidity = 75.0

    if request.method == 'POST':
        try:
            user_temp = float(request.form.get('temperature', 28.0))
            user_humidity = float(request.form.get('humidity', 75.0))
        except ValueError:
            user_temp = 28.0
            user_humidity = 75.0

        file = request.files.get('file')
        existing_filename = request.form.get('existing_filename')

        if file and file.filename != '' and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = cv2.imread(filepath)
            
            # Call your updated model function which now returns a LIST of dictionaries
            raw_detections = predict_ripeness(filepath) 

            # Loop through every isolated mango coordinate profile returned
            for index, det in enumerate(raw_detections):
                pred_class = det.get('class', 'Unknown')
                conf_score = det.get('confidence', 0.0)
                box_data = det.get('box', None)

                p_class_str = pred_class.lower()
                if "overripe" in p_class_str:
                    days_remaining = "0"
                    harvest_msg = "Past harvest time. Process immediately."
                elif "unripe" in p_class_str:
                    if harvest_model:
                        pred = harvest_model.predict([[0, user_temp, user_humidity]])
                        days_remaining = str(int(round(pred[0])))
                        harvest_msg = f"Estimated {days_remaining} days until optimal harvest/process."
                    else:
                        days_remaining = "N/A"
                        harvest_msg = "Model missing."
                elif "ripe" in p_class_str:
                    days_remaining = "0"
                    harvest_msg = "Optimal Harvest Time! Pick now."
                else:
                    days_remaining = "Error"
                    harvest_msg = "Could not determine status."

                suggestion_text = get_handling_suggestion(pred_class)

                detections_list.append({
                    'id': index + 1,
                    'class': pred_class,
                    'confidence': conf_score,
                    'suggestion': suggestion_text,
                    'days_remaining': days_remaining,
                    'harvest_msg': harvest_msg
                })

                # Layer bounding rectangles sequentially on top of the same image matrix canvas
                if box_data:
                    x1, y1 = int(box_data.get('x1', 0)), int(box_data.get('y1', 0))
                    x2, y2 = int(box_data.get('x2', 0)), int(box_data.get('y2', 0))
                    
                    # Match bounding outline color schemes explicitly to specific classes
                    if "unripe" in p_class_str:
                        box_color = (0, 255, 255) # Yellow bboxes for Unripe mangoes
                    elif "overripe" in p_class_str:
                        box_color = (0, 0, 255)   # Red bboxes for Overripe mangoes
                    else:
                        box_color = (0, 255, 0)   # Green bboxes for Ripe mangoes

                    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 3)
                    
                    label = f"#{index + 1}: {pred_class} ({conf_score}%)"
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

            # Write the single image with ALL bounding annotations back to your static folder
            cv2.imwrite(filepath, img)

            if detections_list:
                result = {
                    'temp': user_temp,
                    'humidity': user_humidity,
                    'detections': detections_list
                }
                
                # Log individual records safely to history database tables
                for d in detections_list:
                    new_record = DetectionRecord(
                        filename=filename,
                        prediction=d['class'],
                        confidence=d['confidence'],
                        temperature=user_temp,
                        humidity=user_humidity,
                        days_remaining=str(d['days_remaining']),
                        harvest_msg=d['harvest_msg']
                    )
                    db.session.add(new_record)
                db.session.commit()

        elif existing_filename:
            filename = existing_filename

    return render_template('detect.html', result=result, filename=filename)


@app.route('/realtime_detect', methods=['POST'])
def realtime_detect():
    try:
        data = request.json or {}
        image_data = data.get('image')
        user_temp = float(data.get('temperature', 28.0))
        user_humidity = float(data.get('humidity', 75.0))
        
        if not image_data:
            return jsonify({'error': 'No image data captured'}), 400

        # Decode base64 frame string received from frontend
        header, encoded = image_data.split(",", 1)
        decoded = base64.b64decode(encoded)
        np_arr = np.frombuffer(decoded, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Cache frame temporarily for the prediction engine pipeline execution
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'live_stream_cache.jpg')
        cv2.imwrite(temp_path, frame)

        # Run inference pipeline using your native prediction engine
        prediction_class, confidence_score, box_data = predict_ripeness(temp_path)

        # Calculate shelf-life projections dynamically for automated outputs
        days_remaining = "0"
        if "unripe" in prediction_class.lower() and harvest_model:
            pred = harvest_model.predict([[0, user_temp, user_humidity]])
            days_remaining = str(int(round(pred[0])))

        # Construct a list structure to support flexible multi-object listing extensions
        detections = []
        if prediction_class:
            detections.append({
                "label": prediction_class,
                "confidence": confidence_score,
                "days_remaining": days_remaining,
                # Safe dimensions fallbacks to ensure bounding box metrics exist
                "box": box_data if box_data else {"x1": 50, "y1": 50, "x2": 250, "y2": 250}
            })

        # Return original resolution metrics for accurate viewport relative scaling
        h, w, _ = frame.shape
        return jsonify({
            "detections": detections,
            "source_width": w,
            "source_height": h
        })
    except Exception as e:
        print(f"Real-time pipeline tracking error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/export_history')
def export_history():
    # 1. Security Check: Must be logged in
    if 'user_id' not in session:
        flash('Please login to export data.')
        return redirect(url_for('login'))
    
    # 2. Query all records from the database
    records = DetectionRecord.query.order_by(DetectionRecord.timestamp.desc()).all()
    
    # 3. Create a CSV in memory (RAM)
    si = StringIO()
    cw = csv.writer(si)
    
    # Write the Header Row
    cw.writerow(['ID', 'Date/Time', 'Filename', 'Prediction', 'Confidence', 'Temp(C)', 'Humidity(%)', 'Days Remaining'])
    
    # Write the Data Rows
    for r in records:
        cw.writerow([
            r.id, 
            r.timestamp.strftime('%Y-%m-%d %H:%M:%S'), 
            r.filename, 
            r.prediction, 
            f"{r.confidence}%", 
            r.temperature, 
            r.humidity, 
            r.days_remaining
        ])
        
    # 4. Prepare the response as a file download
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=mango_history.csv"
    output.headers["Content-type"] = "text/csv"
    
    return output   


@app.route('/')
def index():
    # 1. If user is NOT logged in, show the public landing page (Option B)
    if 'user_id' not in session:
        return render_template('index.html', logged_in=False)
    index
    # 2. If logged in, calculate statistics for the Dashboard (Option A)
    total_scans = DetectionRecord.query.count()
    ripe_count = DetectionRecord.query.filter_by(prediction='Ripe').count()
    unripe_count = DetectionRecord.query.filter_by(prediction='Unripe').count()
    overripe_count = DetectionRecord.query.filter_by(prediction='Overripe').count()
    
    return render_template('index.html', 
                           logged_in=True,
                           total=total_scans,
                           ripe=ripe_count,
                           unripe=unripe_count,
                           overripe=overripe_count)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check if user exists in DB
        user = User.query.filter_by(username=username, password=password).first()
        
        if user:
            session['user_id'] = user.id
            session['role'] = user.role  # Remember if they are 'manager'
            flash(f'Welcome back, {user.username}!')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password')
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.')
    return redirect(url_for('login'))

# --- MANAGER ONLY ROUTES ---

@app.route('/delete_multiple_records', methods=['POST'])
def delete_multiple_records():
    if 'user_id' not in session:
        flash('Please login to perform this action.')
        return redirect(url_for('login'))

    # Extract checked checkbox identifier values array sent from the HTML page form
    selected_ids = request.form.getlist('selected_ids')
    
    if not selected_ids:
        flash('No items were selected for deletion.')
        return redirect(url_for('history'))

    files_to_check = set()

    try:
        # 1. Loop through database IDs to clean records and gather active filename contexts
        for record_id in selected_ids:
            record = DetectionRecord.query.get(int(record_id))
            if record:
                files_to_check.add(record.filename)
                db.session.delete(record)
        
        # Flush deleted database row states first
        db.session.commit()

        # 2. Check remaining copies for image disk cleanup safety 
        for filename in files_to_check:
            remaining_copies = DetectionRecord.query.filter_by(filename=filename).count()
            if remaining_copies == 0:
                try:
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        print(f"File {filename} cleared safely from system storage disk.")
                except Exception as disk_err:
                    print(f"Non-blocking file clean mistake context context: {disk_err}")

        flash(f'Successfully cleared {len(selected_ids)} analysis records from your history log.')
    except Exception as e:
        db.session.rollback()
        flash(f'An error occurred during bulk clear execution tracking: {str(e)}')

    return redirect(url_for('history'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']       
        phone = request.form['phone']       
        password = request.form['password']
        
        existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
        if existing_user:
            flash('Username or Email already exists. Please choose another.')
            return redirect(url_for('register'))
        
        # Create new user with all fields
        new_user = User(username=username, email=email, phone=phone, password=password, role='user')
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route('/history')
def history():
    records = DetectionRecord.query.order_by(DetectionRecord.timestamp.desc()).all()
    return render_template('history.html', records=records)



# --- CHATBOT CONFIGURATION ---
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 1024,
}

# 1. Use 'gemini-pro' (Stable Version)
# 2. Removed 'system_instruction' from here to avoid errors

chat_model = genai.GenerativeModel(
    model_name="gemini-2.5-flash", 
    generation_config=generation_config
)

@app.route('/chat_api', methods=['POST'])
def chat_api():
    try:
        user_message = request.json.get('message')
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        # Updated with explicit summarization and short-answer constraints
        system_instruction = """
        You are 'MangoBot', an expert agricultural assistant for the MangoDetect application.
        Your goal is to assist users with mango ripeness, shelf life, and storage conditions.

        CRITICAL REQUIREMENT:
        Provide ultra-short, concise, and summarized answers. Use bullet points where possible. 
        Keep your total response under 3 sentences unless listing specific metrics. Avoid long paragraphs.

        ENVIRONMENTAL STANDARDS (MALAYSIA CONTEXT):
        1. Standard Room (No AC): 27°C - 31°C | 70% - 85% Humidity (Warm/Humid kitchen/warehouse)[cite: 38].
        2. AC Room: 20°C - 24°C | 45% - 55% Humidity (Slows ripening, extends shelf life)[cite: 39].
        3. Refrigerator: 4°C - 10°C | 85% - 95% Humidity (Best for long-term storage of ripe mangoes)[cite: 40].
        4. Outdoor Farm (Daytime): 32°C - 36°C | 60% - 75% Humidity (Hot condition)[cite: 41].

        If asked about room temperature, briefly summarize Fan-only vs. AC room settings using the metrics above[cite: 41].
        """
        
        # Combine instruction + user message 
        full_prompt = f"{system_instruction}\n\nUser Question: {user_message}"

        # Start chat (stateless for this simple version) 
        chat = chat_model.start_chat(history=[]) 
        response = chat.send_message(full_prompt) 
        
        return jsonify({'response': response.text}) 
    
    except Exception as e:
        print(f"Chat Error: {e}") 
        return jsonify({'response': "Sorry, I am having trouble connecting to the satellite. Please try again later."}) 

with app.app_context():
    db.create_all()
    # Check if manager exists, if not, create one
    if not User.query.filter_by(username='admin').first():
        manager = User(username='admin', password='password123', role='manager')
        db.session.add(manager)
        db.session.commit()
        print("Manager Account Created: User='admin', Pass='password123'")

if __name__ == '__main__':
    with app.app_context():
        db.create_all()

        
    app.run(debug=True, port=7777)