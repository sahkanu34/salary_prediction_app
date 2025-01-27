import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import  r2_score
import plotly.express as px
import random

# Set random seed
np.random.seed(42)

def generate_data():
    sectors = ['Technology', 'Healthcare', 'Finance', 'Manufacturing', 'Retail', 'Education', 
               'Construction', 'Energy', 'Consulting', 'Real Estate']

    job_titles = {
        'Technology': ['Software Engineer', 'Data Scientist', 'Product Manager', 'IT Manager', 'DevOps Engineer'],
        'Healthcare': ['Doctor', 'Surgeon', 'Hospital Administrator', 'Medical Director', 'Nurse Practitioner'],
        'Finance': ['Investment Banker', 'Financial Analyst', 'Portfolio Manager', 'CFO', 'Account Manager'],
        'Manufacturing': ['Operations Manager', 'Plant Manager', 'Quality Control Manager', 'Production Supervisor', 'Industrial Engineer'],
        'Retail': ['Store Manager', 'Regional Manager', 'Sales Director', 'Merchandising Manager', 'Retail Operations Manager'],
        'Education': ['Professor', 'Principal', 'Dean', 'Department Head', 'Research Director'],
        'Construction': ['Project Manager', 'Site Engineer', 'Construction Manager', 'Architect', 'Civil Engineer'],
        'Energy': ['Energy Analyst', 'Plant Operator', 'Environmental Engineer', 'Project Developer', 'Operations Manager'],
        'Consulting': ['Management Consultant', 'Strategy Consultant', 'Business Analyst', 'Senior Consultant', 'Partner'],
        'Real Estate': ['Real Estate Agent', 'Property Manager', 'Development Manager', 'Investment Analyst', 'Broker']
    }

    data = []
    for _ in range(100):
        sector = random.choice(sectors)
        job_title = random.choice(job_titles[sector])
        experience = random.randint(3, 25)
        age = experience + random.randint(21, 30)
        
        base_salary = 50000 + (experience * 5000) + random.randint(-10000, 10000)
        if sector in ['Technology', 'Finance', 'Consulting']:
            base_salary *= 1.3
        elif sector in ['Healthcare', 'Energy']:
            base_salary *= 1.2
        
        monthly_salary = round(base_salary / 12, 2)
        annual_package = base_salary
        
        data.append({
            'Age': age,
            'Sector': sector,
            'Job_Title': job_title,
            'Experience': experience,
            'Monthly_Salary': monthly_salary,
            'Annual_Package': annual_package,
            'Final': 1 if annual_package >= 100000 else 0
        })
    
    return pd.DataFrame(data)

def train_model(df):
    # Feature Engineering
    df['Experience_Squared'] = df['Experience'] ** 2
    df['Age_Experience_Ratio'] = df['Age'] / (df['Experience'] + 1)
    
    # Prepare the data
    le_sector = LabelEncoder()
    le_job = LabelEncoder()
    
    X = df[['Age', 'Experience', 'Experience_Squared', 'Age_Experience_Ratio']]
    X['Sector_encoded'] = le_sector.fit_transform(df['Sector'])
    X['Job_Title_encoded'] = le_job.fit_transform(df['Job_Title'])
    
    y = df['Annual_Package']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring ='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    return rf_model, scaler, le_sector, le_job, X_test, y_test, y_pred,  r2, cv_rmse

def main():
    st.title('Salary Prediction App 💰')
    
    # Generate data
    df = generate_data()
    
    # Train model
    model, scaler, le_sector, le_job, X_test, y_test, y_pred,  r2, cv_rmse = train_model(df)
    
    # Sidebar for navigation
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Go to', ['Data Explorer', 'Model Performance', 'Salary Predictor'])
    
    if page == 'Data Explorer':
        st.header('Data Explorer')
        st.write('Sample of the dataset:')
        st.dataframe(df.head())
        
        st.subheader('Data Distribution')
        fig = px.box(df, y='Annual_Package', x='Sector', title='Salary Distribution by Sector')
        st.plotly_chart(fig)
        
        fig2 = px.scatter(df, x='Experience', y='Annual_Package', color='Sector',
                         title='Experience vs Annual Package')
        st.plotly_chart(fig2)
        
        fig3 = px.scatter(df, x='Age', y='Annual_Package', color='Sector',
                         title='Age vs Annual Package')
        st.plotly_chart(fig3)
        
    elif page == 'Model Performance':
        st.header('Model Performance')
        
        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
    
        with col2:
            st.metric("R² Score", f"{r2:.3f}")
        with col3:
            st.metric("CV RMSE", f"{cv_rmse:,.2f}")
        
        # Predictions vs Actual plot
        fig = px.scatter(x=y_test, y=y_pred, 
                        title='Predicted vs Actual Values',
                        labels={'x': 'Actual Values', 'y': 'Predicted Values'})
        fig.add_shape(type='line', x0=y_test.min(), y0=y_test.min(),
                     x1=y_test.max(), y1=y_test.max(),
                     line=dict(color='red', dash='dash'))
        st.plotly_chart(fig)
        
    else:  # Salary Predictor
        st.header('Salary Predictor')
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input('Age', min_value=21, max_value=70, value=30)
            experience = st.number_input('Years of Experience', min_value=0, max_value=40, value=5)
        
        with col2:
            sector = st.selectbox('Sector', sorted(df['Sector'].unique()))
            job_title = st.selectbox('Job Title', sorted(df[df['Sector'] == sector]['Job_Title'].unique()))
        
        if st.button('Predict Salary'):
            # Calculate derived features
            experience_squared = experience ** 2
            age_experience_ratio = age / (experience + 1)
            
            # Prepare input data
            input_data = np.array([[
                age, 
                experience,
                experience_squared,
                age_experience_ratio,
                le_sector.transform([sector])[0],
                le_job.transform([job_title])[0]
            ]])
            
            # Scale the input
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            
            # Display prediction with styling
            st.success(f'Predicted Annual Package: ${prediction:,.2f}')
            st.info(f'Predicted Monthly Salary: ${prediction/12:,.2f}')
            
            # Show confidence interval
            confidence = 0.95
            predictions = []
            X_test_array = X_test.values  # Convert DataFrame to numpy array
            for _ in range(100):
                # Generate random indices
                idx = np.random.randint(0, len(X_test_array), size=10)
                # Use the numpy array for prediction
                pred = model.predict(scaler.transform(X_test_array[idx])).mean()
                predictions.append(pred)
            
            # lower = np.percentile(predictions, (1-confidence)*100/2)
            # upper = np.percentile(predictions, 100-(1-confidence)*100/2)
            # st.write(f'95% Confidence Interval: ${lower:,.2f} to ${upper:,.2f}')

if __name__ == '__main__':
    main()