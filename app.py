import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os

# Set page config
st.set_page_config(
    page_title="Self-Optimizing Workflow Agent",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = []
if 'current_run' not in st.session_state:
    st.session_state.current_run = 1

# Title and description
st.title("🤖 Self-Optimizing Workflow Agent")
st.markdown("Monitor and optimize your workflow performance in real-time")

# Sidebar for controls
st.sidebar.header("Control Panel")

# Add some sample data if none exists
if not st.session_state.data:
    sample_data = [
        {
            'run': 1,
            'timestamp': datetime.now() - timedelta(hours=2),
            'accuracy': 0.85,
            'processing_time': 120,
            'memory_usage': 75,
            'status': 'completed'
        },
        {
            'run': 2,
            'timestamp': datetime.now() - timedelta(hours=1),
            'accuracy': 0.88,
            'processing_time': 115,
            'memory_usage': 72,
            'status': 'completed'
        },
        {
            'run': 3,
            'timestamp': datetime.now() - timedelta(minutes=30),
            'accuracy': 0.91,
            'processing_time': 108,
            'memory_usage': 68,
            'status': 'completed'
        }
    ]
    st.session_state.data = sample_data
    st.session_state.current_run = len(sample_data)

# Calculate max_run safely
max_run = len(st.session_state.data) if st.session_state.data else 1

# Fix the slider issue with proper bounds checking
st.sidebar.subheader("Run Selection")
if max_run >= 1:
    # Ensure min <= max for slider
    min_run = 1
    default_run = min(max_run, st.session_state.current_run)
    run_selected = st.sidebar.slider(
        "Select Run", 
        min_value=min_run, 
        max_value=max_run, 
        value=default_run,
        help=f"Choose from {min_run} to {max_run} available runs"
    )
else:
    st.sidebar.warning("No data available")
    run_selected = 1

# Add data button
if st.sidebar.button("Add New Run"):
    new_run = {
        'run': max_run + 1,
        'timestamp': datetime.now(),
        'accuracy': np.random.uniform(0.80, 0.95),
        'processing_time': np.random.randint(90, 150),
        'memory_usage': np.random.randint(60, 80),
        'status': 'completed'
    }
    st.session_state.data.append(new_run)
    st.session_state.current_run = max_run + 1
    st.rerun()

# Clear data button
if st.sidebar.button("Clear All Data"):
    st.session_state.data = []
    st.session_state.current_run = 1
    st.rerun()

# Main content area
if st.session_state.data:
    df = pd.DataFrame(st.session_state.data)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    latest_data = df.iloc[-1] if not df.empty else {}
    
    with col1:
        st.metric(
            label="Current Accuracy",
            value=f"{latest_data.get('accuracy', 0):.2%}",
            delta=f"{(latest_data.get('accuracy', 0) - df.iloc[-2]['accuracy']):.2%}" if len(df) > 1 else None
        )
    
    with col2:
        st.metric(
            label="Processing Time",
            value=f"{latest_data.get('processing_time', 0):.0f}s",
            delta=f"{(latest_data.get('processing_time', 0) - df.iloc[-2]['processing_time']):.0f}s" if len(df) > 1 else None,
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="Memory Usage",
            value=f"{latest_data.get('memory_usage', 0):.0f}%",
            delta=f"{(latest_data.get('memory_usage', 0) - df.iloc[-2]['memory_usage']):.0f}%" if len(df) > 1 else None,
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            label="Total Runs",
            value=len(df),
            delta=1 if len(df) > 0 else 0
        )
    
    # Charts section
    st.subheader("Performance Trends")
    
    # Create two columns for charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Accuracy trend
        fig_acc = px.line(
            df, 
            x='run', 
            y='accuracy', 
            title='Accuracy Over Time',
            markers=True
        )
        fig_acc.update_layout(
            yaxis=dict(tickformat='.0%'),
            height=400
        )
        st.plotly_chart(fig_acc, use_container_width=True)
    
    with chart_col2:
        # Processing time trend
        fig_time = px.line(
            df, 
            x='run', 
            y='processing_time', 
            title='Processing Time Trend',
            markers=True,
            color_discrete_sequence=['#ff7f0e']
        )
        fig_time.update_layout(height=400)
        st.plotly_chart(fig_time, use_container_width=True)
    
    # Memory usage chart
    fig_memory = px.bar(
        df, 
        x='run', 
        y='memory_usage', 
        title='Memory Usage by Run',
        color='memory_usage',
        color_continuous_scale='Viridis'
    )
    fig_memory.update_layout(height=400)
    st.plotly_chart(fig_memory, use_container_width=True)
    
    # Selected run details
    st.subheader(f"Run {run_selected} Details")
    
    if run_selected <= len(df):
        selected_run_data = df[df['run'] == run_selected].iloc[0]
        
        detail_col1, detail_col2 = st.columns(2)
        
        with detail_col1:
            st.write("**Run Information:**")
            st.write(f"- Run Number: {selected_run_data['run']}")
            st.write(f"- Timestamp: {selected_run_data['timestamp']}")
            st.write(f"- Status: {selected_run_data['status'].title()}")
        
        with detail_col2:
            st.write("**Performance Metrics:**")
            st.write(f"- Accuracy: {selected_run_data['accuracy']:.2%}")
            st.write(f"- Processing Time: {selected_run_data['processing_time']:.0f} seconds")
            st.write(f"- Memory Usage: {selected_run_data['memory_usage']:.0f}%")
    
    # Data table
    st.subheader("All Runs Data")
    st.dataframe(
        df.style.format({
            'accuracy': '{:.2%}',
            'processing_time': '{:.0f}s',
            'memory_usage': '{:.0f}%'
        }),
        use_container_width=True
    )
    
    # Export functionality
    st.subheader("Export Data")
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"workflow_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        json_data = df.to_json(orient='records', date_format='iso')
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name=f"workflow_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

else:
    # Empty state
    st.info("No data available. Click 'Add New Run' in the sidebar to get started!")
    
    # Show a placeholder chart
    st.subheader("Sample Dashboard Preview")
    sample_x = list(range(1, 11))
    sample_y = [0.8 + 0.02 * i + np.random.uniform(-0.01, 0.01) for i in sample_x]
    
    fig_preview = px.line(
        x=sample_x, 
        y=sample_y, 
        title='Accuracy Improvement Over Time (Sample)',
        labels={'x': 'Run', 'y': 'Accuracy'}
    )
    fig_preview.update_layout(
        yaxis=dict(tickformat='.0%'),
        height=400
    )
    st.plotly_chart(fig_preview, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")
