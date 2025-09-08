import streamlit as st
import random
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Self-Optimizing Workflow Agent", layout="wide")

# --------------------------
# Session State Initialization
# --------------------------
if "run" not in st.session_state:
    st.session_state.run = 0
if "decision_log" not in st.session_state:
    st.session_state.decision_log = []

# --------------------------
# Dashboard Header
# --------------------------
st.title("🤖 Self-Optimizing Workflow Orchestration Agent")
st.write("Prototype simulation of autonomous multi-agent workflow optimization")

# --------------------------
# Control Buttons
# --------------------------
col1, col2, col3 = st.columns([1, 1, 8])

with col1:
    if st.button("🚀 Run Optimization Cycle"):
        st.session_state.run += 1

        # Simulate agent proposals
        latency_after = random.randint(80, 200)  # ms
        cost_after = random.randint(100, 500)    # $
        error_after = round(random.uniform(0.5, 5), 2)  # %
        throughput_after = random.randint(800, 1200)    # req/s

        # Conflict resolution (pick trade-off strategy)
        conflict_strategy = random.choice([
            "Latency-Biased",
            "Cost-Biased",
            "Balanced",
            "Error-Minimizing",
            "Throughput-Maximizing"
        ])

        # Confidence weights (simulate negotiation influence)
        weights = {
            "Latency": np.random.uniform(0.2, 1.0),
            "Cost": np.random.uniform(0.2, 1.0),
            "Error": np.random.uniform(0.2, 1.0),
            "Throughput": np.random.uniform(0.2, 1.0),
        }
        total = sum(weights.values())
        for k in weights:
            weights[k] = round(weights[k] / total, 2)

        # Log decision
        decision_entry = {
            "Run": st.session_state.run,
            "Strategy": conflict_strategy,
            "Latency_Proposal": latency_after,
            "Cost_Proposal": cost_after,
            "Error_Proposal": error_after,
            "Throughput_Proposal": throughput_after,
            "Final_Decision": f"Applied '{conflict_strategy}' trade-off",
            "Latency_Weight": weights["Latency"],
            "Cost_Weight": weights["Cost"],
            "Error_Weight": weights["Error"],
            "Throughput_Weight": weights["Throughput"],
        }
        st.session_state.decision_log.append(decision_entry)
        st.rerun()

with col2:
    if st.button("🔄 Reset Simulation"):
        st.session_state.run = 0
        st.session_state.decision_log = []
        st.rerun()

# --------------------------
# Replay Mode
# --------------------------
st.subheader("🎬 Replay Mode: Step Through Runs")

if st.session_state.decision_log:
    max_run = len(st.session_state.decision_log)
    run_selected = st.slider("Select Run", 1, max_run, max_run)

    replay_df = pd.DataFrame(st.session_state.decision_log)
    current_run = replay_df[replay_df["Run"] == run_selected].iloc[0]

    st.markdown(f"**Showing Run {run_selected}:** Strategy → `{current_run['Strategy']}`, Decision → `{current_run['Final_Decision']}`")

    # Metrics Snapshot in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Latency", f"{current_run['Latency_Proposal']} ms")
    with col2:
        st.metric("Cost", f"${current_run['Cost_Proposal']}")
    with col3:
        st.metric("Error Rate", f"{current_run['Error_Proposal']}%")
    with col4:
        st.metric("Throughput", f"{current_run['Throughput_Proposal']} req/s")

    # Two column layout for visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Sankey for reasoning
        st.markdown("**Decision Flow Visualization**")
        fig = go.Figure(go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=["Latency", "Cost", "Error", "Throughput", "Conflict Resolution", "Final Decision"],
                color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DFE6E9"]
            ),
            link=dict(
                source=[0, 1, 2, 3, 4],
                target=[4, 4, 4, 4, 5],
                value=[
                    current_run["Latency_Weight"],
                    current_run["Cost_Weight"],
                    current_run["Error_Weight"],
                    current_run["Throughput_Weight"],
                    1
                ],
                color=["rgba(255,107,107,0.4)", "rgba(78,205,196,0.4)", 
                       "rgba(69,183,209,0.4)", "rgba(150,206,180,0.4)", "rgba(223,230,233,0.4)"]
            )
        ))
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Agent Weight Distribution
        st.markdown("**Agent Weight Distribution**")
        conf_df = pd.DataFrame({
            "Agent": ["Latency", "Cost", "Error", "Throughput"],
            "Weight": [
                current_run["Latency_Weight"],
                current_run["Cost_Weight"],
                current_run["Error_Weight"],
                current_run["Throughput_Weight"]
            ]
        })
        
        fig = go.Figure(go.Bar(
            x=conf_df["Agent"],
            y=conf_df["Weight"],
            marker_color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"],
            text=conf_df["Weight"].round(2),
            textposition='auto',
        ))
        fig.update_layout(
            yaxis_title="Weight",
            xaxis_title="Agent",
            height=400,
            margin=dict(l=0, r=0, t=20, b=0),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Click 'Run Optimization Cycle' to start the simulation")

# --------------------------
# Time-Series Evolution
# --------------------------
st.subheader("📈 Metrics Evolution Across Runs")

if st.session_state.decision_log:
    df = pd.DataFrame(st.session_state.decision_log)
    
    # Create interactive Plotly time series
    fig = go.Figure()
    
    # Add traces for each metric
    fig.add_trace(go.Scatter(
        x=df["Run"], 
        y=df["Latency_Proposal"], 
        mode='lines+markers',
        name='Latency (ms)',
        line=dict(color='#FF6B6B', width=2),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=df["Run"], 
        y=df["Cost_Proposal"], 
        mode='lines+markers',
        name='Cost ($)',
        line=dict(color='#4ECDC4', width=2),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=df["Run"], 
        y=df["Error_Proposal"], 
        mode='lines+markers',
        name='Error Rate (%)',
        line=dict(color='#45B7D1', width=2),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=df["Run"], 
        y=df["Throughput_Proposal"], 
        mode='lines+markers',
        name='Throughput (req/s)',
        line=dict(color='#96CEB4', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Workflow Metrics Over Time",
        xaxis_title="Optimization Runs",
        yaxis_title="Value",
        hovermode='x unified',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(gridcolor='lightgray', showgrid=True)
    fig.update_yaxes(gridcolor='lightgray', showgrid=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # --------------------------
    # Statistics Summary
    # --------------------------
    st.subheader("📊 Statistical Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Calculate statistics
        stats_df = pd.DataFrame({
            "Metric": ["Latency (ms)", "Cost ($)", "Error Rate (%)", "Throughput (req/s)"],
            "Mean": [
                df["Latency_Proposal"].mean().round(2),
                df["Cost_Proposal"].mean().round(2),
                df["Error_Proposal"].mean().round(2),
                df["Throughput_Proposal"].mean().round(2)
            ],
            "Std Dev": [
                df["Latency_Proposal"].std().round(2),
                df["Cost_Proposal"].std().round(2),
                df["Error_Proposal"].std().round(2),
                df["Throughput_Proposal"].std().round(2)
            ],
            "Min": [
                df["Latency_Proposal"].min(),
                df["Cost_Proposal"].min(),
                df["Error_Proposal"].min(),
                df["Throughput_Proposal"].min()
            ],
            "Max": [
                df["Latency_Proposal"].max(),
                df["Cost_Proposal"].max(),
                df["Error_Proposal"].max(),
                df["Throughput_Proposal"].max()
            ]
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    with col2:
        # Strategy distribution pie chart
        strategy_counts = df["Strategy"].value_counts()
        fig = go.Figure(go.Pie(
            labels=strategy_counts.index,
            values=strategy_counts.values,
            hole=0.3,
            marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        ))
        fig.update_layout(
            title="Strategy Distribution",
            height=300,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # --------------------------
    # Data Export
    # --------------------------
    st.subheader("💾 Export Data")
    
    col1, col2, col3 = st.columns([1, 1, 8])
    
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="workflow_optimization_log.csv",
            mime="text/csv"
        )
    
    with col2:
        # Create a detailed report
        report = f"""Workflow Optimization Report
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Runs: {len(df)}

SUMMARY STATISTICS:
{stats_df.to_string()}

STRATEGY DISTRIBUTION:
{strategy_counts.to_string()}

DETAILED LOG:
{df.to_string()}
"""
        st.download_button(
            label="📄 Download Report",
            data=report,
            file_name="workflow_optimization_report.txt",
            mime="text/plain"
        )

else:
    st.info("Run at least one optimization to see metrics evolution and statistics.")

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <small>Self-Optimizing Workflow Agent v1.0 | Simulation Mode</small>
    </div>
    """,
    unsafe_allow_html=True
)
