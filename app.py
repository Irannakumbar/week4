from flask import Flask, request, jsonify, render_template_string
import joblib, numpy as np
app=Flask(__name__); model=joblib.load('student_performance_model.joblib')
FEATURES=['study_hours','attendance','previous_score','sleep_hours','assignment_completion','extracurricular']
LIMITS={'study_hours':(0,12),'attendance':(0,100),'previous_score':(0,100),'sleep_hours':(0,16),'assignment_completion':(0,100),'extracurricular':(0,1)}
HTML="""<!doctype html><html><head><meta name='viewport' content='width=device-width,initial-scale=1'><title>Student Performance Predictor</title><style>body{font-family:Arial;background:#f2f4f7;padding:24px;color:#172033}main{max-width:760px;margin:20px auto;background:white;padding:28px;border-radius:14px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}input,select{box-sizing:border-box;width:100%;padding:11px;margin-top:7px;border:1px solid #ccd3dc;border-radius:8px}button{margin-top:20px;padding:12px 18px;background:#155eef;color:white;border:0;border-radius:8px}#result{padding:14px;background:#eef5ff;margin-top:16px;white-space:pre-wrap}@media(max-width:560px){.grid{grid-template-columns:1fr}}</style></head><body><main><h1>Student Performance Predictor</h1><p>Enter details to estimate a score (demo only).</p><form id='f'><div class='grid'><label>Study hours<input name='study_hours' type='number' min='0' max='12' step='.5' value='3' required></label><label>Attendance (%)<input name='attendance' type='number' min='0' max='100' value='85' required></label><label>Previous score<input name='previous_score' type='number' min='0' max='100' value='70' required></label><label>Sleep hours<input name='sleep_hours' type='number' min='0' max='16' step='.5' value='7' required></label><label>Assignments completed (%)<input name='assignment_completion' type='number' min='0' max='100' value='80' required></label><label>Extracurricular<select name='extracurricular'><option value='0'>No</option><option value='1'>Yes</option></select></label></div><button>Predict score</button></form><div id='result'>Prediction appears here.</div><small>Synthetic demo data only. Do not use for real academic decisions.</small></main><script>document.getElementById('f').addEventListener('submit',async e=>{e.preventDefault();let d={};for(let [k,v] of new FormData(e.target).entries())d[k]=Number(v);let b=document.getElementById('result');b.textContent='Predicting...';try{let r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)}),j=await r.json();b.textContent=r.ok?`Estimated score: ${j.predicted_score}/100\nPerformance band: ${j.performance_band}`:'Error: '+j.error}catch(_){b.textContent='Could not reach API'}});</script></body></html>"""
@app.get('/')
def home(): return render_template_string(HTML)
@app.get('/health')
def health(): return jsonify({'status':'ok','model_loaded':True})
@app.post('/predict')
def predict():
 d=request.get_json(silent=True)
 if not isinstance(d,dict): return jsonify({'error':'Send a JSON object.'}),400
 try: values=[float(d[k]) for k in FEATURES]
 except (KeyError,TypeError,ValueError): return jsonify({'error':'Required fields: '+', '.join(FEATURES)}),400
 for k,v in zip(FEATURES,values):
  lo,hi=LIMITS[k]
  if not np.isfinite(v) or not lo<=v<=hi: return jsonify({'error':f'{k} must be between {lo} and {hi}.'}),400
 score=float(np.clip(model.predict([values])[0],0,100)); band='Needs support' if score<50 else ('Developing' if score<70 else 'Strong')
 return jsonify({'predicted_score':round(score,2),'performance_band':band})
if __name__=='__main__': app.run(host='127.0.0.1',port=5000,debug=False)
