# app.py - SIMPLE VERSION (FIXED with Confidence)
import streamlit as st
import joblib
import pandas as pd

from employee_perf_prediction_dataset.csv import some_function

# ============================================================================
# SETUP
# ============================================================================
st.set_page_config(page_title="Performance Predictor", page_icon="🎯")

# Load trained model
model = joblib.load("employee_perf_model.pkl")

# Load sample data template
template = pd.read_csv("template_row.csv")

# Load full employee dataset (for searching)
try:
    full_data = pd.read_csv(r"data\employee_perf_prediction_dataset.csv", 
                            on_bad_lines='skip', engine='python')
    if 'perf_band_next' in full_data.columns:
        full_data = full_data.drop(columns=['perf_band_next'])
except:
    full_data = None

# ============================================================================
# HEADER
# ============================================================================
st.title("🎯 Employee Performance Predictor")
st.markdown("---")

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3 = st.tabs(["🔍 Search Employee", "📝 New Employee", "📊 Upload CSV"])

# ----------------------------------------------------------------------------
# TAB 1: SEARCH EXISTING EMPLOYEE
# ----------------------------------------------------------------------------
with tab1:
    st.subheader("Search for Employee")
    
    if full_data is not None:
        # SEARCH METHOD
        search_by = st.radio("Search by:", ["Employee ID", "Row Number"])
        
        # Option 1: Search by Employee ID
        if search_by == "Employee ID" and 'employee_id' in full_data.columns:
            emp_id = st.selectbox("Select Employee ID:", full_data['employee_id'].unique())
            employee_data = full_data[full_data['employee_id'] == emp_id].iloc[0:1].copy()
            if 'employee_id' in employee_data.columns:
                employee_data = employee_data.drop(columns=['employee_id'])
        
        # Option 2: Search by Row Number
        else:
            row = st.number_input("Enter Row Number:", 0, len(full_data)-1, 0)
            employee_data = full_data.iloc[row:row+1].copy()
            if 'employee_id' in employee_data.columns:
                employee_data = employee_data.drop(columns=['employee_id'])
        
        # SHOW DATA (editable)
        st.write("**Employee Details (you can edit):**")
        edited = st.data_editor(employee_data, hide_index=True)
        
        # PREDICT BUTTON
        if st.button("🔮 Predict", type="primary"):
            prediction = model.predict(edited)[0]
            confidence = model.predict_proba(edited).max() if hasattr(model, 'predict_proba') else 0.85
            
            # SHOW RESULT
            col1, col2 = st.columns(2)
            col1.metric("Predicted Band", prediction)
            col2.metric("Confidence", f"{confidence:.1%}")
    
    else:
        st.warning("⚠️ Dataset not found. Use 'New Employee' tab.")

# ----------------------------------------------------------------------------
# TAB 2: MANUAL ENTRY FOR NEW EMPLOYEE
# ----------------------------------------------------------------------------
with tab2:
    st.subheader("Enter New Employee Details")
    
    # SHOW EMPTY TEMPLATE (editable)
    edited = st.data_editor(template, hide_index=True)
    
    # PREDICT BUTTON
    if st.button("🔮 Predict", type="primary", key="manual"):
        prediction = model.predict(edited)[0]
        confidence = model.predict_proba(edited).max() if hasattr(model, 'predict_proba') else 0.85
        
        # SHOW RESULT
        col1, col2 = st.columns(2)
        col1.metric("Predicted Band", prediction)
        col2.metric("Confidence", f"{confidence:.1%}")

# ----------------------------------------------------------------------------
# TAB 3: BATCH UPLOAD (FIXED - NOW WITH CONFIDENCE!)
# ----------------------------------------------------------------------------
with tab3:
    st.subheader("Upload CSV File")
    
    # FILE UPLOADER
    file = st.file_uploader("Choose CSV file", type=["csv"])
    
    if file:
        df = pd.read_csv(file)
        st.write(f"✅ Loaded {len(df)} employees")
        st.dataframe(df.head())
        
        # PREDICT BUTTON
        if st.button("🚀 Predict All"):
            # Remove employee_id if exists
            df_pred = df.drop(columns=['employee_id']) if 'employee_id' in df.columns else df
            
            # Make predictions
            predictions = model.predict(df_pred)
            df['predicted_band'] = predictions
            
            # ADD CONFIDENCE (FIXED!)
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(df_pred)
                df['confidence'] = [f"{p.max():.1%}" for p in probabilities]
            
            # Summary stats
            st.write("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total", len(df))
            col2.metric("High (H)", (df['predicted_band'] == 'H').sum())
            col3.metric("Medium (M)", (df['predicted_band'] == 'M').sum())
            
            # Show results
            st.dataframe(df)
            
            # DOWNLOAD BUTTON
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Download", csv, "results.csv")

# ============================================================================
# FOOTER
# ============================================================================
st.caption("HR Analytics | Model v1.0")
