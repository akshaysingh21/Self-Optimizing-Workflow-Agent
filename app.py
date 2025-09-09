import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import random
import time
import matplotlib

# Set page config
st.set_page_config(
    page_title="Self-Optimizing AI Platform",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state with AI agent memory
if 'business_data' not in st.session_state:
    st.session_state.business_data = []
if 'ai_decisions' not in st.session_state:
    st.session_state.ai_decisions = []
if 'optimization_history' not in st.session_state:
    st.session_state.optimization_history = []
if 'current_week' not in st.session_state:
    st.session_state.current_week = 1
if 'ai_enabled' not in st.session_state:
    st.session_state.ai_enabled = True
if 'learning_rate' not in st.session_state:
    st.session_state.learning_rate = 0.85  # AI gets better over time
if 'auto_optimize' not in st.session_state:
    st.session_state.auto_optimize = False
if 'last_optimization' not in st.session_state:
    st.session_state.last_optimization = datetime.now()

# AI Agent Configuration
AI_STRATEGIES = {
    'conservative': {'risk_tolerance': 0.3, 'learning_speed': 0.1, 'description': 'Low risk, steady improvements'},
    'balanced': {'risk_tolerance': 0.6, 'learning_speed': 0.15, 'description': 'Moderate risk, balanced growth'},
    'aggressive': {'risk_tolerance': 0.9, 'learning_speed': 0.25, 'description': 'High risk, rapid optimization'}
}

# Business scenario configurations
PRODUCTS = {
    'Fresh Milk': {'category': 'Dairy', 'shelf_life': 7, 'base_price': 4.99, 'seasonality': 1.0, 'ai_priority': 'high'},
    'Organic Bread': {'category': 'Bakery', 'shelf_life': 5, 'base_price': 6.49, 'seasonality': 1.1, 'ai_priority': 'medium'},
    'Premium Coffee': {'category': 'Beverages', 'shelf_life': 365, 'base_price': 12.99, 'seasonality': 1.2, 'ai_priority': 'low'},
    'Seasonal Fruits': {'category': 'Produce', 'shelf_life': 4, 'base_price': 8.99, 'seasonality': 1.5, 'ai_priority': 'high'},
    'Winter Jackets': {'category': 'Apparel', 'shelf_life': 180, 'base_price': 89.99, 'seasonality': 2.0, 'ai_priority': 'medium'}
}

CAMPAIGNS = ['Holiday Sale', 'Flash Friday', 'Loyalty Rewards', 'Clearance', 'New Product Launch', 'AI-Optimized', 'Dynamic Pricing']

class AIOptimizationAgent:
    def __init__(self, strategy='balanced'):
        self.strategy = AI_STRATEGIES[strategy]
        self.learning_memory = {}
        self.performance_history = []
    
    def learn_from_results(self, week_data, previous_decisions):
        """AI learns from previous decisions and their outcomes"""
        if not previous_decisions:
            return
        
        # Calculate decision effectiveness
        for decision in previous_decisions:
            actual_performance = week_data[week_data['product'] == decision['product']]
            if not actual_performance.empty:
                actual_revenue = actual_performance['revenue'].iloc[0]
                expected_revenue = decision.get('expected_impact', actual_revenue)
                effectiveness = actual_revenue / expected_revenue if expected_revenue > 0 else 1.0
                
                # Store learning
                decision_type = decision['type']
                if decision_type not in self.learning_memory:
                    self.learning_memory[decision_type] = []
                
                self.learning_memory[decision_type].append({
                    'effectiveness': effectiveness,
                    'context': decision.get('context', {}),
                    'week': decision.get('week', 0)
                })
    
    def generate_optimizations(self, current_data, week):
        """AI generates optimization recommendations"""
        optimizations = []
        
        for _, item in current_data.iterrows():
            product = item['product']
            
            # Dynamic Pricing Optimization
            if item['days_to_expiry'] <= 3:
                discount = self._calculate_optimal_discount(item, week)
                optimizations.append({
                    'type': 'dynamic_pricing',
                    'product': product,
                    'action': f'Apply {discount:.0f}% discount',
                    'expected_impact': item['revenue'] * (1 + discount/100 * 0.5),
                    'confidence': min(0.95, st.session_state.learning_rate + 0.1),
                    'reasoning': f'AI predicts {discount:.0f}% discount will maximize revenue before expiry',
                    'week': week
                })
            
            # Inventory Rebalancing
            if item['current_stock'] < item['forecasted_demand'] * 0.7:
                reorder_qty = self._calculate_optimal_reorder(item, week)
                optimizations.append({
                    'type': 'inventory_reorder',
                    'product': product,
                    'action': f'Reorder {reorder_qty:.0f} units',
                    'expected_impact': reorder_qty * item['current_price'] * 0.4,
                    'confidence': st.session_state.learning_rate,
                    'reasoning': f'Prevent stockout based on {item["forecasted_demand"]:.0f} demand forecast',
                    'week': week
                })
            
            # Campaign Optimization
            if item['actual_sales'] > item['forecasted_demand'] * 1.2:
                optimizations.append({
                    'type': 'campaign_scaling',
                    'product': product,
                    'action': 'Scale up marketing campaign',
                    'expected_impact': item['revenue'] * 0.3,
                    'confidence': st.session_state.learning_rate - 0.1,
                    'reasoning': f'High demand detected - scale campaign for {product}',
                    'week': week
                })
        
        return optimizations
    
    def _calculate_optimal_discount(self, item, week):
        """AI calculates optimal discount based on learning"""
        base_discount = (4 - item['days_to_expiry']) * 12  # Base logic
        
        # AI adjustment based on learning
        if 'dynamic_pricing' in self.learning_memory:
            avg_effectiveness = np.mean([d['effectiveness'] for d in self.learning_memory['dynamic_pricing']])
            adjustment = (avg_effectiveness - 1.0) * 10
            base_discount = max(5, min(50, base_discount + adjustment))
        
        return base_discount
    
    def _calculate_optimal_reorder(self, item, week):
        """AI calculates optimal reorder quantity"""
        base_reorder = item['forecasted_demand'] * 1.5 - item['current_stock']
        
        # AI adjustment based on historical accuracy
        if week > 3:
            seasonal_factor = 1.0 + (week % 12) * 0.05  # Seasonal learning
            base_reorder *= seasonal_factor
        
        return max(0, base_reorder)

# Initialize AI Agent
if 'ai_agent' not in st.session_state:
    st.session_state.ai_agent = AIOptimizationAgent('balanced')

def generate_realistic_business_data(week, ai_optimizations=None):
    """Generate realistic business data with AI optimization effects"""
    data = []
    
    for product, details in PRODUCTS.items():
        # Base business logic
        days_to_expiry = random.randint(1, details['shelf_life'])
        current_stock = random.randint(50, 500)
        base_demand = random.randint(80, 200)
        seasonal_demand = int(base_demand * details['seasonality'])
        
        # Apply AI optimizations
        ai_impact_multiplier = 1.0
        applied_optimizations = []
        
        if ai_optimizations and st.session_state.ai_enabled:
            for opt in ai_optimizations:
                if opt['product'] == product:
                    if opt['type'] == 'dynamic_pricing':
                        ai_impact_multiplier *= (1.0 + opt['confidence'] * 0.3)
                        applied_optimizations.append(opt['action'])
                    elif opt['type'] == 'campaign_scaling':
                        ai_impact_multiplier *= (1.0 + opt['confidence'] * 0.4)
                        applied_optimizations.append(opt['action'])
        
        # Dynamic pricing logic with AI enhancement
        urgency_multiplier = 1.0
        if days_to_expiry <= 2:
            urgency_multiplier = 0.7 * (1 + st.session_state.learning_rate * 0.1)
        elif days_to_expiry <= 5:
            urgency_multiplier = 0.85 * (1 + st.session_state.learning_rate * 0.05)
        
        current_price = details['base_price'] * urgency_multiplier
        
        # Campaign with AI optimization
        campaign = random.choice(CAMPAIGNS)
        if applied_optimizations and st.session_state.ai_enabled:
            campaign = 'AI-Optimized'
        
        campaign_lift = {
            'Holiday Sale': 1.4, 'Flash Friday': 1.6, 'Loyalty Rewards': 1.2,
            'Clearance': 0.9, 'New Product Launch': 1.1,
            'AI-Optimized': 1.5 + st.session_state.learning_rate * 0.3,
            'Dynamic Pricing': 1.3 + st.session_state.learning_rate * 0.2
        }
        
        # Calculate final metrics
        sales_volume = int(seasonal_demand * campaign_lift[campaign] * ai_impact_multiplier * random.uniform(0.8, 1.2))
        revenue = sales_volume * current_price
        waste_percentage = max(0, (current_stock - sales_volume) / current_stock * 100) if current_stock > 0 else 0
        
        # Reduce waste with AI optimization
        if st.session_state.ai_enabled and details['ai_priority'] == 'high':
            waste_percentage *= (1 - st.session_state.learning_rate * 0.2)
        
        cost_per_unit = details['base_price'] * 0.6
        profit_margin = ((current_price - cost_per_unit) / current_price) * 100
        
        data.append({
            'week': week,
            'product': product,
            'category': details['category'],
            'current_stock': current_stock,
            'days_to_expiry': days_to_expiry,
            'forecasted_demand': seasonal_demand,
            'actual_sales': sales_volume,
            'base_price': details['base_price'],
            'current_price': current_price,
            'revenue': revenue,
            'campaign': campaign,
            'waste_percentage': min(waste_percentage, 100),
            'profit_margin': profit_margin,
            'inventory_turnover': sales_volume / current_stock if current_stock > 0 else 0,
            'ai_optimized': len(applied_optimizations) > 0,
            'ai_actions': ', '.join(applied_optimizations) if applied_optimizations else 'None',
            'ai_priority': details['ai_priority']
        })
    
    return data

# Title with AI branding
st.title("🤖 Self-Optimizing AI Business Platform")
st.markdown("""
### Autonomous Intelligence for Retail Excellence
*Real-time learning • Dynamic optimization • Predictive insights*
""")

# AI Control Panel
with st.container():
    ai_col1, ai_col2, ai_col3, ai_col4 = st.columns(4)
    
    with ai_col1:
        ai_status = "🟢 ACTIVE" if st.session_state.ai_enabled else "🔴 INACTIVE"
        st.metric("🤖 AI Agent Status", ai_status)
    
    with ai_col2:
        learning_progress = st.session_state.learning_rate
        st.metric("📈 Learning Progress", f"{learning_progress:.1%}", delta=f"+{(learning_progress-0.8)*100:.1f}%")
    
    with ai_col3:
        total_optimizations = len([d for d in st.session_state.ai_decisions if d.get('week', 0) > 0])
        st.metric("⚡ AI Decisions Made", total_optimizations, delta="+Auto")
    
    with ai_col4:
        if st.session_state.optimization_history:
            avg_improvement = np.mean([h['improvement'] for h in st.session_state.optimization_history])
            st.metric("📊 Avg Performance Lift", f"+{avg_improvement:.1%}", delta="AI Impact")
        else:
            st.metric("📊 Performance Impact", "Initializing...", delta="Learning")

# AI Configuration Sidebar
st.sidebar.header("🤖 AI Agent Controls")

# AI Strategy Selection
ai_strategy = st.sidebar.selectbox(
    "AI Optimization Strategy",
    list(AI_STRATEGIES.keys()),
    format_func=lambda x: f"{x.title()} - {AI_STRATEGIES[x]['description']}"
)

if ai_strategy != getattr(st.session_state.ai_agent, 'current_strategy', 'balanced'):
    st.session_state.ai_agent = AIOptimizationAgent(ai_strategy)
    st.session_state.ai_agent.current_strategy = ai_strategy

# AI Controls
st.session_state.ai_enabled = st.sidebar.toggle("🔄 Enable AI Optimization", value=st.session_state.ai_enabled)
st.session_state.auto_optimize = st.sidebar.toggle("⚡ Auto-Execute Decisions", value=st.session_state.auto_optimize)

if st.sidebar.button("🧠 Trigger AI Learning Cycle"):
    if st.session_state.business_data:
        df = pd.DataFrame(st.session_state.business_data)
        latest_week_data = df[df['week'] == df['week'].max()]
        st.session_state.ai_agent.learn_from_results(latest_week_data, st.session_state.ai_decisions)
        st.session_state.learning_rate = min(0.95, st.session_state.learning_rate + 0.05)
        st.sidebar.success("🎯 AI Learning Complete!")
        st.rerun()

# Business Week Controls
st.sidebar.header("📅 Simulation Controls")
max_week = max([d['week'] for d in st.session_state.business_data]) if st.session_state.business_data else 1

if max_week > 1:
    selected_week = st.sidebar.slider(
        "Business Week", 
        min_value=1, 
        max_value=max_week, 
        value=max_week
    )
else:
    selected_week = 1
    st.sidebar.write(f"Business Week: {selected_week}")

# Initialize sample data
if not st.session_state.business_data:
    for week in range(1, 5):
        week_data = generate_realistic_business_data(week)
        st.session_state.business_data.extend(week_data)
        
        # Generate initial AI optimizations
        if week > 1 and st.session_state.ai_enabled:
            current_data = pd.DataFrame([d for d in st.session_state.business_data if d['week'] == week])
            optimizations = st.session_state.ai_agent.generate_optimizations(current_data, week)
            st.session_state.ai_decisions.extend(optimizations)

# Simulate Next Week with AI
if st.sidebar.button("▶️ Run AI Simulation"):
    new_week = max_week + 1
    
    # Generate AI optimizations for the new week
    current_data = pd.DataFrame([d for d in st.session_state.business_data if d['week'] == max_week])
    ai_optimizations = st.session_state.ai_agent.generate_optimizations(current_data, new_week) if st.session_state.ai_enabled else []
    
    # Generate new week data with AI optimizations applied
    new_data = generate_realistic_business_data(new_week, ai_optimizations)
    st.session_state.business_data.extend(new_data)
    
    # Store AI decisions
    if ai_optimizations:
        st.session_state.ai_decisions.extend(ai_optimizations)
    
    # Calculate performance improvement
    if max_week > 1:
        prev_week_data = pd.DataFrame([d for d in st.session_state.business_data if d['week'] == max_week])
        new_week_data = pd.DataFrame(new_data)
        
        prev_revenue = prev_week_data['revenue'].sum()
        new_revenue = new_week_data['revenue'].sum()
        improvement = (new_revenue - prev_revenue) / prev_revenue if prev_revenue > 0 else 0
        
        st.session_state.optimization_history.append({
            'week': new_week,
            'improvement': improvement,
            'ai_decisions': len(ai_optimizations),
            'timestamp': datetime.now()
        })
    
    # AI Learning - improve over time
    st.session_state.learning_rate = min(0.95, st.session_state.learning_rate + random.uniform(0.01, 0.03))
    st.rerun()

# Process and display data
if st.session_state.business_data:
    df = pd.DataFrame(st.session_state.business_data)
    current_week_data = df[df['week'] == selected_week]
    
    # Enhanced Executive KPIs with AI impact
    st.subheader("📊 AI-Enhanced Business Intelligence")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_revenue = current_week_data['revenue'].sum()
        ai_optimized_revenue = current_week_data[current_week_data['ai_optimized'] == True]['revenue'].sum()
        ai_lift = (ai_optimized_revenue / total_revenue * 100) if total_revenue > 0 else 0
        st.metric(
            "💰 Total Revenue", 
            f"${total_revenue:,.0f}",
            delta=f"AI Lift: {ai_lift:.1f}%"
        )
    
    with col2:
        avg_margin = current_week_data['profit_margin'].mean()
        ai_margin_boost = current_week_data[current_week_data['ai_optimized'] == True]['profit_margin'].mean() - current_week_data[current_week_data['ai_optimized'] == False]['profit_margin'].mean()
        st.metric(
            "📈 Profit Margin", 
            f"{avg_margin:.1f}%",
            delta=f"AI Boost: +{ai_margin_boost:.1f}%" if not pd.isna(ai_margin_boost) else "Optimizing"
        )
    
    with col3:
        avg_waste = current_week_data['waste_percentage'].mean()
        ai_waste_reduction = (current_week_data[current_week_data['ai_priority'] == 'high']['waste_percentage'].mean())
        waste_status = "🎯 AI Optimized" if st.session_state.ai_enabled else "Standard"
        st.metric(
            "♻️ Waste Rate", 
            f"{avg_waste:.1f}%",
            delta=waste_status
        )
    
    with col4:
        ai_decisions_count = len([d for d in st.session_state.ai_decisions if d.get('week') == selected_week])
        st.metric(
            "🤖 AI Decisions", 
            ai_decisions_count,
            delta="Real-time"
        )
    
    with col5:
        if st.session_state.optimization_history:
            recent_improvement = st.session_state.optimization_history[-1]['improvement']
            st.metric(
                "📈 AI Performance", 
                f"+{recent_improvement:.1%}",
                delta="Week-over-week"
            )
        else:
            st.metric("📈 AI Performance", "Learning...", delta="Initializing")

    # AI Decision Timeline
    st.subheader("🧠 AI Decision Intelligence Center")
    
    ai_col1, ai_col2 = st.columns([2, 1])
    
    with ai_col1:
        # Current week AI decisions
        current_ai_decisions = [d for d in st.session_state.ai_decisions if d.get('week') == selected_week]
        
        if current_ai_decisions:
            st.markdown("### 🎯 Active AI Optimizations")
            for i, decision in enumerate(current_ai_decisions):
                confidence_color = "🟢" if decision['confidence'] > 0.8 else "🟡" if decision['confidence'] > 0.6 else "🟠"
                
                with st.expander(f"{confidence_color} {decision['action']} - {decision['product']}", expanded=i==0):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write(f"**Type:** {decision['type'].replace('_', ' ').title()}")
                        st.write(f"**Confidence:** {decision['confidence']:.1%}")
                        st.write(f"**Expected Impact:** ${decision['expected_impact']:,.0f}")
                    with col_b:
                        st.write(f"**AI Reasoning:**")
                        st.write(decision['reasoning'])
                        
                    # Auto-execute button for high-confidence decisions
                    if decision['confidence'] > 0.85:
                        if st.button(f"⚡ Auto-Execute", key=f"exec_{i}"):
                            st.success(f"✅ AI Decision Executed: {decision['action']}")
        else:
            st.info("🤖 AI is analyzing current conditions. New optimizations will appear here.")
    
    with ai_col2:
        # AI Learning Progress
        st.markdown("### 📈 AI Learning Metrics")
        
        # Learning rate visualization
        fig_learning = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = st.session_state.learning_rate * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "AI Intelligence Level"},
            delta = {'reference': 80},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "lightgreen"},
                'steps': [
                    {'range': [0, 60], 'color': "lightgray"},
                    {'range': [60, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}))
        fig_learning.update_layout(height=300)
        st.plotly_chart(fig_learning, use_container_width=True)
        
        # AI Strategy Impact
        if st.session_state.optimization_history:
            st.metric("🎯 Total AI Impact", 
                     f"+{sum([h['improvement'] for h in st.session_state.optimization_history]):.1%}")
            st.metric("⚡ Decisions Made", 
                     sum([h['ai_decisions'] for h in st.session_state.optimization_history]))

    # AI-Enhanced Business Charts
    st.subheader("📊 Intelligent Business Analytics")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Revenue trend with AI impact highlighting
        weekly_data = df.groupby('week').agg({
            'revenue': 'sum',
            'ai_optimized': 'sum'
        }).reset_index()
        weekly_data['ai_revenue'] = df[df['ai_optimized'] == True].groupby('week')['revenue'].sum().reindex(weekly_data['week'], fill_value=0).values
        
        fig_revenue_ai = go.Figure()
        fig_revenue_ai.add_trace(go.Scatter(
            x=weekly_data['week'],
            y=weekly_data['revenue'],
            mode='lines+markers',
            name='Total Revenue',
            line=dict(color='blue', width=3)
        ))
        fig_revenue_ai.add_trace(go.Scatter(
            x=weekly_data['week'],
            y=weekly_data['ai_revenue'],
            mode='lines+markers',
            name='AI-Optimized Revenue',
            line=dict(color='green', width=2),
            fill='tonexty'
        ))
        fig_revenue_ai.update_layout(
            title='📈 Revenue Trend: AI vs Traditional',
            xaxis_title="Week",
            yaxis_title="Revenue ($)"
        )
        st.plotly_chart(fig_revenue_ai, use_container_width=True)
    
    with chart_col2:
        # AI Decision Impact Matrix
        ai_impact_data = current_week_data[['product', 'revenue', 'ai_optimized', 'ai_priority']].copy()
        ai_impact_data['ai_status'] = ai_impact_data['ai_optimized'].map({True: 'AI-Optimized', False: 'Standard'})
        
        fig_ai_impact = px.scatter(
            ai_impact_data,
            x='product',
            y='revenue',
            color='ai_status',
            size='revenue',
            symbol='ai_priority',
            title='🤖 AI Optimization Impact Map',
            color_discrete_map={'AI-Optimized': 'green', 'Standard': 'gray'}
        )
        fig_ai_impact.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_ai_impact, use_container_width=True)

    # AI Performance Dashboard
    if st.session_state.optimization_history:
        st.subheader("🚀 AI Performance Timeline")
        
        perf_df = pd.DataFrame(st.session_state.optimization_history)
        
        perf_col1, perf_col2 = st.columns(2)
        
        with perf_col1:
            fig_ai_perf = px.line(
                perf_df,
                x='week',
                y='improvement',
                title='📈 AI Performance Improvement',
                markers=True
            )
            fig_ai_perf.update_layout(yaxis_tickformat='.1%')
            st.plotly_chart(fig_ai_perf, use_container_width=True)
        
        with perf_col2:
            fig_decisions = px.bar(
                perf_df,
                x='week',
                y='ai_decisions',
                title='⚡ AI Decisions per Week',
                color='ai_decisions',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_decisions, use_container_width=True)

    # Enhanced Business Data Table
    st.subheader("📋 AI-Enhanced Business Intelligence Data")
    
    display_df = current_week_data[[
        'product', 'category', 'current_stock', 'forecasted_demand', 'actual_sales',
        'current_price', 'revenue', 'profit_margin', 'waste_percentage',
        'ai_optimized', 'ai_actions', 'ai_priority'
    ]].copy()
    
    # Color-code AI optimized rows
    def highlight_ai_rows(row):
        if row['ai_optimized']:
            return ['background-color: #e8f5e8'] * len(row)
        return [''] * len(row)
    
    st.dataframe(
        display_df.style
        .format({
            'current_price': '${:.2f}',
            'revenue': '${:,.0f}',
            'waste_percentage': '{:.1f}%',
            'profit_margin': '{:.1f}%'
        })
        .apply(highlight_ai_rows, axis=1)
        .background_gradient(subset=['revenue', 'profit_margin']),
        use_container_width=True
    )

    # AI Insights & Recommendations
    st.subheader("💡 AI Strategic Insights")
    
    insight_col1, insight_col2, insight_col3 = st.columns(3)
    
    with insight_col1:
        st.success("🎯 **AI Recommendations**")
        st.write("• Increase Premium Coffee inventory by 25%")
        st.write("• Apply dynamic pricing to Fresh Milk")
        st.write("• Scale Flash Friday campaigns")
        st.write(f"• Confidence Level: {st.session_state.learning_rate:.1%}")
    
    with insight_col2:
        st.info("🔮 **Predictive Insights**")
        next_week_revenue_pred = total_revenue * (1 + st.session_state.learning_rate * 0.1)
        st.write(f"• Predicted Next Week Revenue: ${next_week_revenue_pred:,.0f}")
        st.write("• High-priority items need attention: 2")
        st.write("• Optimal reorder window: 3-5 days")
    
    with insight_col3:
        st.warning("⚠️ **Risk Mitigation**")
        st.write("• Monitor Seasonal Fruits expiry closely")
        st.write("• Dairy category waste trending up")
        st.write("• Consider promotional campaigns for slow movers")

# Real-time AI Status Updates
if st.session_state.auto_optimize and st.session_state.ai_enabled:
    current_time = datetime.now()
    if (current_time - st.session_state.last_optimization).seconds > 30:  # Every 30 seconds
        st.session_state.last_optimization = current_time
        with st.empty():
            st.info("🤖 AI Agent is analyzing real-time data...")
            time.sleep(2)
            st.success("✅ AI optimization cycle completed!")

# Footer with AI impact metrics
st.markdown("---")
if st.session_state.optimization_history:
    total_ai_improvement = sum([h['improvement'] for h in st.session_state.optimization_history])
    total_decisions = sum([h['ai_decisions'] for h in st.session_state.optimization_history])
    
    st.markdown(f"""
    <div style='text-align: center; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px;'>
    <h3>🤖 AI Performance Summary</h3>
    <p><strong>Total Performance Improvement:</strong> +{total_ai_improvement:.1%} | <strong>AI Decisions Made:</strong> {total_decisions} | <strong>Learning Rate:</strong> {st.session_state.learning_rate:.1%}</p>
    <p><em>Self-optimizing AI delivered measurable business impact through automated decision-making</em></p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style='text-align: center; background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 20px; border-radius: 10px;'>
    <h3>🚀 AI Agent Initialization Complete</h3>
    <p><strong>Ready for Autonomous Optimization</strong> • Real-time Learning • Predictive Intelligence</p>
    <p><em>Run AI simulation to see self-optimizing workflows in action</em></p>
    </div>
    """, unsafe_allow_html=True)
