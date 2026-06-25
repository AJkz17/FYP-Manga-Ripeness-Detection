import os
import base64
import csv
import cv2
import joblib
import numpy as np
import secrets
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.utils import secure_filename
from models import db, DetectionRecord, User
from io import StringIO
from flask import make_response
from ml_integration import predict_ripeness
from dotenv import load_dotenv
from google import genai
from google.genai import types

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', secrets.token_hex(32))

basedir = os.path.abspath(os.path.dirname(__file__))

# init Database 
instance_path = os.path.join(basedir, 'instance')
if not os.path.exists(instance_path):
    os.makedirs(instance_path)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(instance_path, 'mango.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
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

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Prepare the unified configuration mapping block
generation_config = types.GenerateContentConfig(
    temperature=0.7,
    top_p=0.95,
    top_k=64,
    max_output_tokens=1024
)

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

            # Save a pristine raw clone backup BEFORE cv2 draws on it for your logistics view
            raw_filename = "raw_" + filename
            raw_filepath = os.path.join(app.config['UPLOAD_FOLDER'], raw_filename)
            import shutil
            shutil.copyfile(filepath, raw_filepath)

            img = cv2.imread(filepath)
            
            # Call your model function which returns a LIST of dictionaries
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
                
                # --- UPDATED UNRIPE LOGIC SECTION ---
                elif "unripe" in p_class_str:
                    if harvest_model:
                        # 1. Predict raw baseline days using your environment metrics
                        pred = harvest_model.predict([[0, user_temp, user_humidity]])
                        calculated_days = int(round(pred[0]))
                        
                        # 2. Enforce the minimum 5-day clamp standard
                        final_days = max(5, calculated_days)
                        
                        days_remaining = str(final_days)
                        harvest_msg = f"Estimated {days_remaining} days until optimal harvest/process."
                    else:
                        # Baseline fallback standard if the simple .pkl file is missing
                        days_remaining = "5"
                        harvest_msg = "Model missing. Default baseline applied."
                
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

        detections = []
        if prediction_class:
            detections.append({
                "label": prediction_class,
                "confidence": confidence_score,
                "days_remaining": days_remaining,
                "box": box_data if box_data else {"x1": 50, "y1": 50, "x2": 250, "y2": 250}
            })

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
    if 'user_id' not in session:
        flash('Please login to export data.')
        return redirect(url_for('login'))
    records = DetectionRecord.query.order_by(DetectionRecord.timestamp.desc()).all()
    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(['ID', 'Date/Time', 'Filename', 'Prediction', 'Confidence', 'Temp(C)', 'Humidity(%)', 'Days Remaining'])
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
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=mango_history.csv"
    output.headers["Content-type"] = "text/csv"
    
    return output   


@app.route('/')
def index():
    if 'user_id' not in session:
        return render_template('index.html', logged_in=False)
    
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
        user = User.query.filter_by(username=username, password=password).first()
        
        if user:
            session['user_id'] = user.id
            session['role'] = user.role 
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

# --- Manager Functionality ---
@app.route('/delete_multiple_records', methods=['POST'])
def delete_multiple_records():
    if 'user_id' not in session:
        flash('Please login to perform this action.')
        return redirect(url_for('login'))
    selected_ids = request.form.getlist('selected_ids')
    
    if not selected_ids:
        flash('No items were selected for deletion.')
        return redirect(url_for('history'))

    files_to_check = set()

    try:
        for record_id in selected_ids:
            record = DetectionRecord.query.get(int(record_id))
            if record:
                files_to_check.add(record.filename)
                db.session.delete(record)
        
        db.session.commit()

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

@app.route('/logistics', methods=['GET', 'POST'])
def logistics():
    if 'user_id' not in session:
        flash('Please login to view logistics mapping suites.')
        return redirect(url_for('login'))

    records = DetectionRecord.query.order_by(DetectionRecord.timestamp.desc()).all()

    selected_record = None
    selected_route = None
    transit_time = 0
    shelf_life = 0
    condition_margin = 0

    if request.method == 'POST':
        record_id = request.form.get('record_id')
        selected_route = request.form.get('route_id')

        if record_id and selected_route:
            selected_record = DetectionRecord.query.get(int(record_id))

            if selected_record:
                route_transit_matrix = {
                    'route_a': 1,  # Kuala Lumpur (1 Day Transit) 
                    'route_b': 3,  # Singapore (3 Days Transit) 
                    'route_c': 5   # East Asia Export (5 Days Transit)
                }
                transit_time = route_transit_matrix.get(selected_route, 0)

                try:
                    shelf_life = int(selected_record.days_remaining)
                except ValueError:
                    shelf_life = 0  

                condition_margin = shelf_life - transit_time 

    return render_template(
        'logistics.html',
        records=records,
        selected_record=selected_record,
        selected_route=selected_route,
        transit_time=transit_time,
        shelf_life=shelf_life,
        condition_margin=condition_margin
    )

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


# --- UPGRADED: Chatbot endpoint utilizing the google-genai Client architecture ---
@app.route('/chat_api', methods=['POST'])
def chat_api():
    try:
        user_message = request.json.get('message')
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
            
        system_instruction = """
        You are 'MangoBot', an expert agricultural assistant for the MangoDetect application.
        Your goal is to assist users with mango ripeness, shelf life, and storage conditions.

        CRITICAL REQUIREMENT:
        Provide ultra-short, concise, and summarized answers. Use bullet points where possible. 
        Keep your total response under 3 sentences unless listing specific metrics. Avoid long paragraphs.

        ENVIRONMENTAL STANDARDS (MALAYSIA CONTEXT):
        1. Standard Room (No AC): 27°C - 31°C | 70% - 85% Humidity (Warm/Humid kitchen/warehouse).
        2. AC Room: 20°C - 24°C | 45% - 55% Humidity (Slows ripening, extends shelf life).
        3. Refrigerator: 4°C - 10°C | 85% - 95% Humidity (Best for long-term storage of ripe mangoes).
        4. Outdoor Farm (Daytime): 32°C - 36°C | 60% - 75% Humidity (Hot condition).

        If asked about room temperature, briefly summarize Fan-only vs. AC room settings using the metrics above.
        """

        full_prompt = f"{system_instruction}\n\nUser Question: {user_message}"
        
        # UPGRADED: Generate content statelessly using models.generate_content with our loaded config setup
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
            config=generation_config
        )
        
        return jsonify({'response': response.text}) 
    
    except Exception as e:
        print(f"Chat Error: {e}") 
        return jsonify({'response': "Sorry, I am having trouble connecting to the satellite. Please try again later."}) 

with app.app_context():
    db.create_all()
    if not User.query.filter_by(username='admin').first():
        manager = User(username='admin', password='password123', role='manager')
        db.session.add(manager)
        db.session.commit()
        print("Manager Account Created: User='admin', Pass='password123'")

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        
    app.run(debug=True, port=7777)