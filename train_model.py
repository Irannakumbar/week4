import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Synthetic data for demonstration only; not real student records.
rng=np.random.default_rng(42); n=1200
df=pd.DataFrame({'study_hours':rng.uniform(0,12,n),'attendance':rng.uniform(40,100,n),'previous_score':rng.uniform(20,100,n),'sleep_hours':rng.uniform(3,10,n),'assignment_completion':rng.uniform(0,100,n),'extracurricular':rng.integers(0,2,n)})
df['final_score']=(8+2.8*df.study_hours+.16*df.attendance+.30*df.previous_score+.75*df.sleep_hours+.12*df.assignment_completion+1.5*df.extracurricular+rng.normal(0,7,n)).clip(0,100)
features=['study_hours','attendance','previous_score','sleep_hours','assignment_completion','extracurricular']
Xtr,Xte,ytr,yte=train_test_split(df[features],df.final_score,test_size=.2,random_state=42)
model=Pipeline([('scaler',StandardScaler()),('regressor',RandomForestRegressor(n_estimators=200,random_state=42,min_samples_leaf=2))]); model.fit(Xtr,ytr); pred=model.predict(Xte)
mae=mean_absolute_error(yte,pred); rmse=float(np.sqrt(mean_squared_error(yte,pred))); r2=r2_score(yte,pred)
joblib.dump(model,'student_performance_model.joblib'); df.to_csv('student_performance_demo_dataset.csv',index=False); pd.DataFrame({'actual_score':yte,'predicted_score':pred}).to_csv('test_predictions.csv',index=False)
with open('evaluation_results.txt','w') as f: f.write(f'Rows: {n}\nTraining rows: {len(Xtr)}\nTesting rows: {len(Xte)}\nMAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR2: {r2:.3f}\n')
print('Training complete'); print(f'Rows: {n} | Training: {len(Xtr)} | Testing: {len(Xte)}'); print(f'MAE: {mae:.3f}'); print(f'RMSE: {rmse:.3f}'); print(f'R²: {r2:.3f}'); print('Saved model, dataset, predictions, and evaluation_results.txt')
