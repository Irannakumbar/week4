# Student Performance Prediction API

A machine-learning demo estimating a score from study hours, attendance, previous score, sleep, assignment completion, and extracurricular participation. The data is synthetic, not real student records; do not use for real grading.

## Run locally
1. Install Python 3.10+.
2. `pip install -r requirements.txt`
3. `python train_model.py`
4. `python app.py`
5. Open `http://127.0.0.1:5000`.

API: `GET /health`; `POST /predict` with JSON keys `study_hours`, `attendance`, `previous_score`, `sleep_hours`, `assignment_completion`, `extracurricular`. The training script reports MAE, RMSE, and R². Metrics demonstrate the workflow only because data is synthetic.
