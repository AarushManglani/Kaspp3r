from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

app = Flask(__name__)
def load_model_and_data():
    global xgb_reg, X_train, df2
    df_main = pd.read_csv('medical_charges.csv')
    X = df_main.drop('Charges', axis=1).copy()
    y = df_main['Charges'].copy()
    X_encoded = pd.get_dummies(X, columns=['Sex', 'Physical Impairments', 'Mental Disabilities', 'Smoker', 'Region'])
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, 
                             gamma=0.005, learning_rate=0.1, max_depth=3, n_estimators=100)
    xgb_reg.fit(X_train, y_train)
    df2 = pd.read_csv('updated_insurance_plans.csv')
    
    return "Model and data loaded successfully!"

load_model_and_data()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        age = int(request.form['age'])
        children = int(request.form['children'])
        bmi = float(request.form['bmi'])
        sex = request.form['sex'].lower()
        smoker = request.form['smoker'].lower()
        region = request.form['region'].lower()
        impairments = request.form['impairments'].lower()
        mental_disability = request.form['mental_disability'].lower()
        aorp = request.form['aorp'].lower()
        
        budget = None
        if aorp == 'affordability' and 'budget' in request.form and request.form['budget']:
            budget = int(request.form['budget'])
        
        input_data = {
            "Age": age,
            "Number of Children": children,
            "BMI": bmi,
            "Sex_Female": 1 if sex == "female" else 0,
            "Sex_Male": 1 if sex == "male" else 0,
            "Smoker_Yes": 1 if smoker == "yes" else 0,
            "Smoker_No": 1 if smoker == "no" else 0,
            "Region_Urban": 1 if region == "urban" else 0,
            "Region_Suburban": 1 if region == "suburban" else 0,
            "Region_Rural": 1 if region == "rural" else 0,
            "Physical Impairments_Hearing": 1 if impairments == "hearing" else 0,
            "Physical Impairments_Mobility": 1 if impairments == "mobility" else 0,
            "Physical Impairments_Visual": 1 if impairments == "visual" else 0,
            "Mental Disabilities_Anxiety": 1 if mental_disability == "anxiety" else 0,
            "Mental Disabilities_Depression": 1 if mental_disability == "depression" else 0,
            "Mental Disabilities_Schizophrenia": 1 if mental_disability == "schizophrenia" else 0,
        }
        
        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(columns=X_train.columns, fill_value=0)
        predicted_charges = xgb_reg.predict(input_df)[0]
        
        recommended_plans = None
        plan_message = ""
        
        if aorp == 'affordability':
            if budget is not None:
                affordable_plans = df2[df2['Monthly Premium'] * 12 <= budget]
                if not affordable_plans.empty:
                    recommended_plans = affordable_plans[['Plan Name', 'Plan ID', 'Monthly Premium']].head(10).to_dict('records')
                    plan_message = "Affordable Plans within your Budget:"
                else:
                    plan_message = "No insurance plans fit within your budget."
            else:
                plan_message = "Budget not provided!"
        elif aorp == 'risk protection':
            best_plan = df2[df2['Max Coverage'] >= predicted_charges].sort_values(by='Max Coverage', ascending=True).head(5)
            if not best_plan.empty:
                recommended_plans = best_plan[['Plan Name', 'Plan ID', 'Monthly Premium', 'Max Coverage']].to_dict('records')
                plan_message = "Best Risk Protection Plans:"
            else:
                plan_message = "No suitable plan found with high enough coverage."
        
        return render_template('result.html', 
                               predicted_charges=predicted_charges, 
                               recommended_plans=recommended_plans,
                               plan_message=plan_message,
                               user_data=request.form)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
