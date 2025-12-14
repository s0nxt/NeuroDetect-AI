from flask import render_template, request, redirect, url_for, session, flash, send_file
from app import app, db, users_col, history_col, brain_model, eye_model, lung_model, class_labels, EYE_LABELS, LUNG_LABELS, MONGODB_AVAILABLE
from app.utils import (
    validate_email, validate_password, validate_username, is_valid_mri_file,
    convert_dicom_to_image, convert_nifti_to_image, process_nifti_slices, preprocess_image,
    get_prediction_confidence, generate_pdf_report, TUMOR_INFO, EYE_INFO, LUNG_INFO
)
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import os
import uuid
import numpy as np
from datetime import datetime

# Auth Decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            if not MONGODB_AVAILABLE:
                if 'user' not in session:
                     session['user'] = 'demo_user'
            else:
                flash('Please login to access this page.', 'error')
                return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    if not MONGODB_AVAILABLE:
        flash('Database not available. Please install and start MongoDB.', 'error')
        return render_template('register.html')
    if request.method == 'POST':
        username = request.form['username'].strip()
        email = request.form['email'].strip()
        password = request.form['password']

        if not validate_username(username):
            flash('Username must be 3-20 characters and contain only letters, numbers, and underscores.', 'error')
            return render_template('register.html')
            
        if not validate_email(email):
            flash('Invalid email address format.', 'error')
            return render_template('register.html')
            
        if not validate_password(password):
            flash('Password must be at least 8 characters long and contain both letters and numbers.', 'error')
            return render_template('register.html')

        if users_col.find_one({'username': username}):
            flash('Username already exists', 'error')
            return redirect(url_for('register'))

        if users_col.find_one({'email': email}):
            flash('Email already registered. Please login.', 'error')
            return redirect(url_for('register'))

        hashed_pw = generate_password_hash(password)
        users_col.insert_one({
            'username': username,
            'email': email,
            'password': hashed_pw
        })
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    if not MONGODB_AVAILABLE:
        flash('Database not available. Please install and start MongoDB.', 'error')
        return render_template('login.html')
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']

        user = users_col.find_one({'username': username})
        if user and check_password_hash(user['password'], password):
            session['user'] = username
            flash('Logged in successfully!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('index'))

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    if request.method == 'POST':
        patient_name = request.form.get('patient_name', '').strip()
        analysis_type = request.form.get('analysis_type', 'brain') # Default to brain
        
        if not patient_name:
            flash('Patient name is required.', 'error')
            return redirect(url_for('dashboard'))

        if 'file' not in request.files:
            flash('No file uploaded! Please select a medical image.', 'error')
            return redirect(url_for('dashboard'))

        file = request.files['file']
        if file.filename == '':
            flash('No file selected! Please choose a medical image.', 'error')
            return redirect(url_for('dashboard'))

        # Reuse MRI validation for now as it checks for image formats too
        is_valid, message = is_valid_mri_file(file, analysis_type)
        if not is_valid:
            flash(message, 'error')
            return redirect(url_for('dashboard'))

        file_extension = os.path.splitext(file.filename)[1].lower()
        filename = str(uuid.uuid4()) + file_extension
        # Save to app/static
        upload_path = os.path.join('app', 'static', filename)
        os.makedirs(os.path.join('app', 'static'), exist_ok=True)
        
        file.save(upload_path)
        
        processed_image_path = upload_path
        slice_paths = [] # Initialize slice_paths

        if file_extension in ['.dcm', '.dicom']:
            processed_filename = str(uuid.uuid4()) + '.jpg'
            processed_image_path = os.path.join('app', 'static', processed_filename)
            
            if convert_dicom_to_image(upload_path, processed_image_path):
                filename = processed_filename
            else:
                flash('Error processing DICOM file. Please try a different image.', 'error')
                return redirect(url_for('dashboard'))
        
        elif file_extension in ['.nii', '.nii.gz']:
            processed_filename = str(uuid.uuid4()) + '.jpg'
            processed_image_path = os.path.join('app', 'static', processed_filename)
            
            # Process single slice for prediction
            if convert_nifti_to_image(upload_path, processed_image_path):
                filename = processed_filename
            else:
                flash('Error processing NIfTI file. Please try a different image.', 'error')
                return redirect(url_for('dashboard'))
                
            # Process all slices for 3D viewer (only for brain MRI)
            if analysis_type == 'brain':
                slices_dir_name = f"slices_{uuid.uuid4()}"
                slices_output_dir = os.path.join('app', 'static', slices_dir_name)
                slice_paths = process_nifti_slices(upload_path, slices_output_dir)

        try:
            # All models (Brain, Eye, Lung) now use EfficientNetB0 which expects 224x224
            target_size = (224, 224)
            img = preprocess_image(processed_image_path, target_size=target_size)
            
            # Select Model and Labels based on Analysis Type
            active_model = None
            active_labels = []
            info_dict = {}
            
            if analysis_type == 'eye':
                active_model = eye_model
                active_labels = EYE_LABELS
                info_dict = EYE_INFO
            elif analysis_type == 'lung':
                active_model = lung_model
                active_labels = LUNG_LABELS
                info_dict = LUNG_INFO
            else: # Default to brain
                active_model = brain_model
                active_labels = class_labels
                info_dict = TUMOR_INFO

            if active_model:
                prediction_array = active_model.predict(img)
            else:
                import random
                # Simulate prediction if model not loaded
                num_classes = len(active_labels)
                prediction_array = np.zeros((1, num_classes))
                prediction_array[0][random.randint(0, num_classes - 1)] = 0.9
                flash(f'Running in DEMO mode ({analysis_type} model not loaded). Result is simulated.', 'warning')
            
            predicted_index = np.argmax(prediction_array)
            predicted_class = active_labels[predicted_index]
            
            # Softmax normalization if needed (usually model output is already softmax)
            if np.sum(prediction_array) < 0.9 or np.sum(prediction_array) > 1.1:
                e_x = np.exp(prediction_array - np.max(prediction_array))
                prediction_array = e_x / e_x.sum(axis=1, keepdims=True)

            confidence_scores = get_prediction_confidence(prediction_array, active_labels)
            
            # Check for low confidence (potential wrong modality or ambiguous image)
            top_confidence = confidence_scores[predicted_class]
            
            # Dynamic threshold based on number of classes/difficulty
            warning_threshold = 50.0
            if analysis_type == 'eye': # 5 classes, harder task
                warning_threshold = 40.0
            elif analysis_type == 'lung': # 4 classes
                warning_threshold = 45.0
            elif analysis_type == 'brain': # 2 classes (Binary), should be high
                warning_threshold = 60.0

            if top_confidence < warning_threshold:
                flash(f"Note: The AI model is uncertain ({top_confidence:.1f}%). This can happen with complex cases or unclear images.", "warning")
            
            heatmap_filename = None
            # Grad-CAM generation
            try:
                import app.gradcam as gradcam
                print(f"Generating Grad-CAM heatmap for {analysis_type} model...")
                
                target_layer = 'block5_conv3' if analysis_type == 'brain' else 'out_relu'
                if analysis_type == 'lung':
                    target_layer = 'block5_conv3' # Assuming VGG-like architecture for lung model too
                
                heatmap = gradcam.get_gradcam_heatmap(active_model, img, target_layer_name=target_layer)
                
                if heatmap is not None:
                    heatmap_filename = "heatmap_" + filename
                    heatmap_path = os.path.join('app', 'static', heatmap_filename)
                    gradcam.save_gradcam(processed_image_path, heatmap, heatmap_path)
                    print(f"Grad-CAM saved to {heatmap_path}")
            except Exception as e:
                print(f"Grad-CAM failed: {e}")
                import traceback
                traceback.print_exc()

            if MONGODB_AVAILABLE:
                history_record = {
                    'username': session['user'],
                    'patient_name': patient_name,
                    'analysis_type': analysis_type,
                    'image_path': filename,
                    'heatmap_path': heatmap_filename,
                    'prediction': predicted_class,
                    'confidence_scores': confidence_scores,
                    'date': datetime.now()
                }
                result = history_col.insert_one(history_record)
                session['last_analysis_id'] = str(result.inserted_id)
            else:
                session['last_analysis'] = {
                    'patient_name': patient_name,
                    'analysis_type': analysis_type,
                    'image_path': filename,
                    'heatmap_path': heatmap_filename,
                    'prediction': predicted_class,
                    'confidence_scores': confidence_scores,
                    'date': datetime.now()
                }

            # For template rendering, we pass the filename relative to static folder
            # The template uses {{ url_for('static', filename=...) }}
            return render_template('result.html', 
                                 image_path=filename, # Just filename for url_for
                                 heatmap_path=heatmap_filename,
                                 slice_paths=slice_paths, # Pass slice paths
                                 prediction=predicted_class, 
                                 patient_name=patient_name,
                                 confidence_scores=confidence_scores,
                                 tumor_info=info_dict.get(predicted_class, {}))
        
        except Exception as e:
            flash(f'Error processing medical image: {str(e)}', 'error')
            return redirect(url_for('dashboard'))

    return render_template('dashboard.html', username=session['user'])

@app.route('/download_report')
@login_required
def download_report():
    try:
        if MONGODB_AVAILABLE and session.get('last_analysis_id'):
            from bson import ObjectId
            analysis_id = session.get('last_analysis_id')
            record = history_col.find_one({'_id': ObjectId(analysis_id)})
            if not record:
                flash('Analysis record not found.', 'error')
                return redirect(url_for('dashboard'))
            
            patient_name = record['patient_name']
            image_path = os.path.join('app', 'static', record['image_path'])
            heatmap_path = os.path.join('app', 'static', record['heatmap_path']) if record.get('heatmap_path') else None
            prediction = record['prediction']
            confidence_scores = record.get('confidence_scores', {})
            analysis_date = record['date']
        elif session.get('last_analysis'):
            record = session['last_analysis']
            patient_name = record['patient_name']
            image_path = os.path.join('app', 'static', record['image_path'])
            heatmap_path = os.path.join('app', 'static', record['heatmap_path']) if record.get('heatmap_path') else None
            prediction = record['prediction']
            confidence_scores = record.get('confidence_scores', {})
            analysis_date = record['date']
        else:
            flash('No recent analysis found. Please perform an analysis first.', 'error')
            return redirect(url_for('dashboard'))
        
        # Determine info_dict based on record content or default
        # Since record doesn't store info_dict, we need to infer it or store analysis_type
        # Fortunately, record has 'analysis_type' if it came from DB, or we can check session
        
        current_analysis_type = record.get('analysis_type', 'brain')
        current_info_dict = TUMOR_INFO
        if current_analysis_type == 'eye':
            current_info_dict = EYE_INFO
        elif current_analysis_type == 'lung':
            current_info_dict = LUNG_INFO
            
        pdf_buffer = generate_pdf_report(
            patient_name=patient_name,
            image_path=image_path,
            prediction=prediction,
            confidence_scores=confidence_scores,
            analysis_date=analysis_date,
            heatmap_path=heatmap_path,
            info_dict=current_info_dict,
            analysis_type=current_analysis_type
        )
        
        safe_patient_name = "".join(c for c in patient_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"Analysis_Report_{safe_patient_name}_{analysis_date.strftime('%Y%m%d_%H%M%S')}.pdf"
        
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )
        
    except Exception as e:
        flash(f'Error generating report: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

@app.route('/download_report/<record_id>')
@login_required
def download_historical_report(record_id):
    try:
        from bson import ObjectId
        record = history_col.find_one({'_id': ObjectId(record_id), 'username': session['user']})
        if not record:
            flash('Analysis record not found or access denied.', 'error')
            return redirect(url_for('history'))
        
        image_path = os.path.join('app', 'static', record['image_path'])
        heatmap_path = os.path.join('app', 'static', record['heatmap_path']) if record.get('heatmap_path') else None
        
        current_analysis_type = record.get('analysis_type', 'brain')
        current_info_dict = TUMOR_INFO
        if current_analysis_type == 'eye':
            current_info_dict = EYE_INFO
        elif current_analysis_type == 'lung':
            current_info_dict = LUNG_INFO

        pdf_buffer = generate_pdf_report(
            patient_name=record['patient_name'],
            image_path=image_path,
            prediction=record['prediction'],
            confidence_scores=record.get('confidence_scores', {}),
            analysis_date=record['date'],
            heatmap_path=heatmap_path,
            info_dict=current_info_dict,
            analysis_type=current_analysis_type
        )
        
        safe_patient_name = "".join(c for c in record['patient_name'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"Analysis_Report_{safe_patient_name}_{record['date'].strftime('%Y%m%d_%H%M%S')}.pdf"
        
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )
        
    except Exception as e:
        flash(f'Error generating report: {str(e)}', 'error')
        return redirect(url_for('history'))

@app.route('/delete_history/<record_id>')
@login_required
def delete_history(record_id):
    if not MONGODB_AVAILABLE:
        flash('History deletion is not available in demo mode.', 'error')
        return redirect(url_for('history'))
        
    try:
        from bson import ObjectId
        result = history_col.delete_one({
            '_id': ObjectId(record_id),
            'username': session['user']
        })
        
        if result.deleted_count > 0:
            flash('Record deleted successfully.', 'success')
        else:
            flash('Record not found or access denied.', 'error')
            
    except Exception as e:
        flash(f'Error deleting record: {str(e)}', 'error')
        
    return redirect(url_for('history'))

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    try:
        data = request.get_json()
        user_question = data.get('message')
        
        # Get context from session or request
        context = {}
        record = None
        if MONGODB_AVAILABLE and session.get('last_analysis_id'):
            from bson import ObjectId
            record = history_col.find_one({'_id': ObjectId(session['last_analysis_id'])})
        elif session.get('last_analysis'):
            record = session['last_analysis']
            
        if record:
            a_type = record.get('analysis_type', 'brain')
            if a_type == 'eye':
                info_dict = EYE_INFO
            elif a_type == 'lung':
                info_dict = LUNG_INFO
            else:
                info_dict = TUMOR_INFO
                
            context = {
                'diagnosis': record['prediction'],
                'confidence': record.get('confidence_scores', {}).get(record['prediction'], 'N/A'),
                'description': info_dict.get(record['prediction'], {}).get('description', '')
            }
            
        from app.chat_service import get_ai_response
        ai_response = get_ai_response(user_question, context)
        
        return {'response': ai_response}
        
    except Exception as e:
        print(f"Chat error: {e}")
        return {'response': "Sorry, an error occurred while processing your request."}, 500

@app.route('/history')
@login_required
def history():
    if not MONGODB_AVAILABLE:
        flash('History feature requires MongoDB. Please install and start MongoDB.', 'error')
        return redirect(url_for('dashboard'))

    user_history = list(history_col.find({'username': session['user']}).sort('date', -1))
    return render_template('history.html', history=user_history, tumor_info=TUMOR_INFO, eye_info=EYE_INFO, lung_info=LUNG_INFO)

@app.route('/feedback', methods=['POST'])
@login_required
def feedback():
    if not MONGODB_AVAILABLE:
        flash('Feedback feature requires MongoDB.', 'error')
        return redirect(url_for('dashboard'))
        
    try:
        analysis_id = request.form.get('analysis_id')
        correct_diagnosis = request.form.get('correct_diagnosis')
        comments = request.form.get('comments')
        
        from app import db
        feedback_col = db['feedback']
        
        feedback_record = {
            'analysis_id': analysis_id,
            'username': session['user'],
            'correct_diagnosis': correct_diagnosis,
            'comments': comments,
            'date': datetime.now()
        }
        
        feedback_col.insert_one(feedback_record)
        flash('Thank you for your feedback! This helps improve our AI.', 'success')
        
    except Exception as e:
        flash(f'Error submitting feedback: {str(e)}', 'error')
        
    return redirect(url_for('history'))

@app.route('/email_report/<record_id>')
@login_required
def email_report(record_id):
    try:
        from bson import ObjectId
        from app.utils import send_email_report
        
        record = history_col.find_one({'_id': ObjectId(record_id), 'username': session['user']})
        if not record:
            flash('Analysis record not found or access denied.', 'error')
            return redirect(url_for('history'))
        
        # Get user email
        user = users_col.find_one({'username': session['user']})
        if not user or 'email' not in user:
            flash('User email not found.', 'error')
            return redirect(url_for('history'))
            
        image_path = os.path.join('app', 'static', record['image_path'])
        heatmap_path = os.path.join('app', 'static', record['heatmap_path']) if record.get('heatmap_path') else None
        
        current_analysis_type = record.get('analysis_type', 'brain')
        current_info_dict = TUMOR_INFO
        if current_analysis_type == 'eye':
            current_info_dict = EYE_INFO
        elif current_analysis_type == 'lung':
            current_info_dict = LUNG_INFO

        pdf_buffer = generate_pdf_report(
            patient_name=record['patient_name'],
            image_path=image_path,
            prediction=record['prediction'],
            confidence_scores=record.get('confidence_scores', {}),
            analysis_date=record['date'],
            heatmap_path=heatmap_path,
            info_dict=current_info_dict
        )
        
        safe_patient_name = "".join(c for c in record['patient_name'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"Analysis_Report_{safe_patient_name}_{record['date'].strftime('%Y%m%d_%H%M%S')}.pdf"
        
        if send_email_report(user['email'], record['patient_name'], pdf_buffer, filename):
            flash(f'Report emailed successfully to {user["email"]}', 'success')
        else:
            flash('Failed to send email. Please check server configuration.', 'error')
            
    except Exception as e:
        flash(f'Error emailing report: {str(e)}', 'error')
        
    return redirect(url_for('history'))
