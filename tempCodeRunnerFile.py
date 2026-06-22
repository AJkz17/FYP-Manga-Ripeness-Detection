import os
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.utils import secure_filename
from models import db, DetectionRecord, User
from flask import session
import csv
import cv2
from io import StringIO
from flask import make_response


from ml_integration import predict_ripeness
import joblib
import google.generativeai as genai

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
    prediction_class = None
    confidence_score = 0.0
    days_remaining = "N/A"
    harvest_msg = ""
    
    # Default environmental values
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

        # ==========================================
        # SCENARIO A: NEW IMAGE UPLOAD (VIA API)
        # ==========================================
        if file and file.filename != '' and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # 1. Call your API Function
            # Note: Now it returns THREE values (Class, Conf, Box)
            prediction_class, confidence_score, box_data = predict_ripeness(filepath)

            # 2. Draw the Box if data exists
            if box_data:
                # Load image with OpenCV
                img = cv2.imread(filepath)
                
                # Extract coordinates (API usually returns x1, y1, x2, y2)
                x1 = int(box_data.get('x1', 0))
                y1 = int(box_data.get('y1', 0))
                x2 = int(box_data.get('x2', 0))
                y2 = int(box_data.get('y2', 0))
                
                # Draw Rectangle (Green, thickness=3)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Draw Label
                label = f"{prediction_class} {confidence_score}%"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Save it back
                cv2.imwrite(filepath, img)

        # ==========================================
        # SCENARIO B: HISTORY / EXISTING IMAGE
        # ==========================================
        elif existing_filename:
            filename = existing_filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            last_record = DetectionRecord.query.filter_by(filename=filename).order_by(DetectionRecord.timestamp.desc()).first()

            if last_record:
                prediction_class = last_record.prediction
                confidence_score = last_record.confidence
            else:
                # If history missing, re-run API
                if os.path.exists(filepath):
                    prediction_class, confidence_score, box_data = predict_ripeness(filepath)
                    # Optional: Re-draw box here if needed, same logic as above
                else:
                    flash("Error: Original file not found.")
                    return redirect(request.url)
        
        else:
            flash('No selected file')
            return redirect(request.url)


        # ==========================================
        # HARVEST & SHELF LIFE PREDICTION
        # ==========================================
        # HARVEST PREDICTION
        if filename and prediction_class:
            p_class_str = prediction_class.lower()

            if "overripe" in p_class_str:
                days_remaining = "-1"
                harvest_msg = "Past harvest time. Process immediately."

            elif "unripe" in p_class_str:
                if harvest_model:
                    # Original logic: Predicts based on Class 0 (Unripe), Temp, Humidity
                    pred = harvest_model.predict([[0, user_temp, user_humidity]])
                    days_remaining = str(int(round(pred[0])))
                    harvest_msg = f"Estimated {days_remaining} days until optimal harvest."

            elif "ripe" in p_class_str:
                # Original logic: Hardcoded to 0 days for ripe
                days_remaining = "0"
                harvest_msg = "Optimal Harvest Time! Pick now."

            else:
                days_remaining = "Error"
                harvest_msg = "Could not determine harvest time."

            suggestion_text = get_handling_suggestion(prediction_class)

            # Result Dict
            result = {
                'class': prediction_class,
                'confidence': confidence_score,
                'suggestion': suggestion_text,
                'temp': user_temp,
                'humidity': user_humidity,
                'days_remaining': days_remaining,
                'shelf_life': days_remaining,
                'harvest_msg': harvest_msg
            }

            # Save to DB
            new_record = DetectionRecord(
                filename=filename,
                prediction=prediction_class,
                confidence=confidence_score,
                temperature=user_temp,
                humidity=user_humidity,
                days_remaining=str(days_remaining),
                harvest_msg=harvest_msg
            )
            db.session.add(new_record)
            db.session.commit()

    return render_template(
        'detect.html',
        result=result,
        filename=filename
    )

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

@app.route('/delete_record/<int:id>', methods=['POST'])
def delete_record(id):
    if 'user_id' not in session:
        flash('Please login to perform this action.')
        return redirect(url_for('login'))

    record = DetectionRecord.query.get_or_404(id)
    filename_to_check = record.filename
    
    # 1. Delete the Database Record first
    db.session.delete(record)
    db.session.commit()
    
    # 2. Safety Check: Are there any OTHER records still using this filename?
    # We query the database to see if this filename still exists in any remaining records.
    remaining_copies = DetectionRecord.query.filter_by(filename=filename_to_check).count()
    
    if remaining_copies == 0:
        # 3. No one else is using this image, so it is SAFE to delete the file.
        try:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename_to_check)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"File {filename_to_check} deleted from disk.")
        except Exception as e:
            print(f"Error deleting file: {e}")
    else:
        # 4. Someone else is using it, so we KEEP the file.
        print(f"File {filename_to_check} kept because {remaining_copies} other records use it.")

    flash('Record deleted successfully.')
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

        # Define the system instruction here
        system_instruction = """
        You are 'MangoBot', an expert agricultural assistant for the MangoDetect application.
        Your goal is to assist users with mango ripeness, shelf life, and storage conditions.

        IMPORTANT: GUIDELINES FOR TEMPERATURE & HUMIDITY (MALAYSIA CONTEXT):
        Users often ask about environmental settings. Use these standards to answer them:

        1. **Standard Room (No Air-Cond):**
        - Temp: 27°C - 31°C (Warm/Humid)
        - Humidity: 70% - 85%
        - Note: This is typical for an open warehouse or kitchen in Malaysia.

        2. **Air-Conditioned Room:**
        - Temp: 20°C - 24°C (Cool/Dry)
        - Humidity: 45% - 55%
        - Note: This significantly slows down ripening (extends shelf life).

        3. **Refrigerator (Chiller):**
        - Temp: 4°C - 10°C
        - Humidity: 85% - 95%
        - Note: Best for long-term storage of ripe mangoes.

        4. **Outdoor Farm (Daytime):**
        - Temp: 32°C - 36°C (Hot)
        - Humidity: 60% - 75%

        If a user asks "What is the room temperature?", explain the difference between a Fan-only room vs. an AC room using the data above.
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