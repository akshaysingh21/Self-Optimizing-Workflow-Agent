import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import random
import time

# Set page config
st.set_page_config(
    page_title="Self-Optimizing AI Platform",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'business_data' not in st.session_state:
    st.session_state.business_data = []
if 'ai_decisions' not in st.session_state:
    st.session_state.ai_decisions = []
if 'optimization_history' not in st.session_state:
    st.session_state.optimization_history = []
if 'strategy_performance' not in st.session_state:
    st.session_state.strategy_performance = {}
if 'function_level_metrics' not in st.session_state:
    st.session_state.function_level_metrics = {}
if 'intervention_alerts' not in st.session_state:
    st.session_state.intervention_alerts = []
if 'current_week' not in st.session_state:
    st.session_state.current_week = 1
if 'ai_enabled' not in st.session_state:
    st.session_state.ai_enabled = True
if 'learning_rate' not in st.session_state:
    st.session_state.learning_rate = 0.85
if 'auto_optimize' not in st.session_state:
    st.session_state.auto_optimize = False

# AI Strategies and Configuration
AI_STRATEGIES = {
    'conservative': {
        'risk_tolerance': 0.3, 'learning_speed': 0.1, 'description': 'Low risk, steady improvements',
        'discount_cap': 25, 'inventory_buffer': 1.8, 'campaign_scaling': 1.2
    },
    'balanced': {
        'risk_tolerance': 0.6, 'learning_speed': 0.15, 'description': 'Moderate risk, balanced growth',
        'discount_cap': 40, 'inventory_buffer': 1.5, 'campaign_scaling': 1.4
    },
    'aggressive': {
        'risk_tolerance': 0.9, 'learning_speed': 0.25, 'description': 'High risk, rapid optimization',
        'discount_cap': 60, 'inventory_buffer': 1.2, 'campaign_scaling': 1.8
    }
}

OPTIMIZATION_FUNCTIONS = {
    'pricing_optimization': {'weight': 0.3, 'description': 'Dynamic pricing based on demand and shelf life'},
    'inventory_management': {'weight': 0.25, 'description': 'Predictive inventory rebalancing'},
    'campaign_optimization': {'weight': 0.2, 'description': 'Marketing campaign effectiveness'},
    'waste_reduction': {'weight': 0.15, 'description': 'Proactive waste minimization'},
    'demand_forecasting': {'weight': 0.1, 'description': 'Advanced demand prediction'}
}

PRODUCTS = {
    'Fresh Milk': {'category': 'Dairy', 'shelf_life': 7, 'base_price': 4.99, 'seasonality': 1.0, 'ai_priority': 'high'},
    'Organic Bread': {'category': 'Bakery', 'shelf_life': 5, 'base_price': 6.49, 'seasonality': 1.1, 'ai_priority': 'medium'},
    'Premium Coffee': {'category': 'Beverages', 'shelf_life': 365, 'base_price': 12.99, 'seasonality': 1.2, 'ai_priority': 'low'},
    'Seasonal Fruits': {'category': 'Produce', 'shelf_life': 4, 'base_price': 8.99, 'seasonality': 1.5, 'ai_priority': 'high'},
    'Winter Jackets': {'category': 'Apparel', 'shelf_life': 180, 'base_price': 89.99, 'seasonality': 2.0, 'ai_priority': 'medium'}
}

class AIOptimizationAgent:
    def __init__(self, strategy='balanced'):
        self.strategy_name = strategy
        self.strategy = AI_STRATEGIES[strategy]
        self.learning_memory = {}
        self.performance_history = []
    
    def generate_optimizations(self, current_data, week):
        """Generate AI optimization recommendations"""
        optimizations = []
        
        for _, item in current_data.iterrows():
            product = item['product']
            
            # Dynamic Pricing
            if item['days_to_expiry'] <= 3:
                discount = self._calculate_optimal_discount(item, week)
                optimizations.append({
                    'type': 'dynamic_pricing',
                    'function_type': 'pricing_optimization',
                    'product': product,
                    'action': f'Apply {discount:.0f}% discount',
                    'expected_impact': item['revenue'] * (1 + discount/100 * 0.5),
                    'confidence': min(0.95, st.session_state.learning_rate + 0.1),
                    'reasoning': f'AI predicts {discount:.0f}% discount will maximize revenue',
                    'week': week
                })
            
            # Inventory Management
            if item['current_stock'] < item['forecasted_demand'] * 0.7:
                reorder_qty = self._calculate_optimal_reorder(item, week)
                optimizations.append({
                    'type': 'inventory_reorder',
                    'function_type': 'inventory_management',
                    'product': product,
                    'action': f'Reorder {reorder_qty:.0f} units',
                    'expected_impact': reorder_qty * item['current_price'] * 0.4,
                    'confidence': st.session_state.learning_rate,
                    'reasoning': f'Prevent stockout for forecasted demand',
                    'week': week
                })
            
            # Campaign Optimization
            if item['actual_sales'] > item['forecasted_demand'] * 1.2:
                optimizations.append({
                    'type': 'campaign_scaling',
                    'function_type': 'campaign_optimization',
                    'product': product,
                    'action': 'Scale up marketing campaign',
                    'expected_impact': item['revenue'] * 0.3,
                    'confidence': st.session_state.learning_rate - 0.1,
                    'reasoning': f'High demand detected - scale campaign',
                    'week': week
                })
        
        return optimizations
    
    def assess_function_performance(self, week_data, decisions):
        """Assess performance at each optimization function level"""
        function_results = {}
        
        for func_name in OPTIMIZATION_FUNCTIONS.keys():
            func_decisions = [d for d in decisions if d.get('function_type') == func_name]
            
            if func_decisions:
                success_count = sum(1 for d in func_decisions if random.random() > 0.3)  # Simulated success
                total_impact = sum(d.get('expected_impact', 0) for d in func_decisions)
                
                function_results[func_name] = {
                    'success_rate': success_count / len(func_decisions),
                    'total_impact': total_impact,
                    'decisions_made': len(func_decisions),
                    'avg_impact': total_impact / len(func_decisions) if func_decisions else 0
                }
            else:
                function_results[func_name] = {
                    'success_rate': 0, 'total_impact': 0, 'decisions_made': 0, 'avg_impact': 0
                }
        
        week_num = week_data['week'].iloc[0] if not week_data.empty else 0
        st.session_state.function_level_metrics[week_num] = function_results
        return function_results
    
    def check_intervention_needed(self, week_data, function_results):
        """Check if manual intervention is needed"""
        interventions = []
        
        # Check for underperforming functions
        for func_name, results in function_results.items():
            if results['success_rate'] < 0.5 and results['decisions_made'] > 2:
                interventions.append({
                    'type': 'function_underperforming',
                    'severity': 'medium',
                    'function': func_name,
                    'message': f'{func_name.replace("_", " ").title()} success rate below 50%',
                    'recommended_action': 'Review and adjust parameters'
                })
        
        # Check for overall performance decline
        if len(st.session_state.optimization_history) >= 3:
            recent_performance = [h.get('improvement', 0) for h in st.session_state.optimization_history[-3:]]
            if np.mean(recent_performance) < -0.05:
                interventions.append({
                    'type': 'performance_decline',
                    'severity': 'high',
                    'message': 'Overall performance declining for 3+ periods',
                    'recommended_action': 'Strategy review required'
                })
        
        if interventions:
            st.session_state.intervention_alerts.extend(interventions)
        
        return interventions
    
    def _calculate_optimal_discount(self, item, week):
        base_discount = min((4 - item['days_to_expiry']) * 15, self.strategy['discount_cap'])
        return max(5, base_discount)
    
    def _calculate_optimal_reorder(self, item, week):
        base_reorder = item['forecasted_demand'] * self.strategy['inventory_buffer'] - item['current_stock']
        return max(0, base_reorder)

def generate_business_data(week, ai_optimizations=None):
    """Generate realistic business data"""
    data = []
    
    for product, details in PRODUCTS.items():
        days_to_expiry = random.randint(1, details['shelf_life'])
        current_stock = random.randint(50, 500)
        base_demand = random.randint(80, 200)
        seasonal_demand = int(base_demand * details['seasonality'])
        
        # AI optimization effects
        ai_impact_multiplier = 1.0
        applied_optimizations = []
        
        if ai_optimizations and st.session_state.ai_enabled:
            for opt in ai_optimizations:
                if opt['product'] == product:
                    impact = opt['confidence'] * 0.3
                    if opt['type'] == 'dynamic_pricing':
                        ai_impact_multiplier *= (1.0 + impact)
                    elif opt['type'] == 'campaign_scaling':
                        ai_impact_multiplier *= (1.0 + impact * 1.2)
                    applied_optimizations.append(opt['action'])
        
        # Pricing logic
        urgency_multiplier = 1.0
        if days_to_expiry <= 2:
            discount_rate = 30 + st.session_state.learning_rate * 10
            urgency_multiplier = 1 - (min(discount_rate, 50) / 100)
        elif days_to_expiry <= 5:
            discount_rate = 15 + st.session_state.learning_rate * 5
            urgency_multiplier = 1 - (min(discount_rate, 25) / 100)
        
        current_price = details['base_price'] * urgency_multiplier
        
        # Campaign effects
        campaign = 'AI-Optimized' if applied_optimizations else random.choice(['Holiday Sale', 'Flash Friday', 'Loyalty Rewards', 'Clearance'])
        campaign_lift = 1.5 + st.session_state.learning_rate * 0.4 if campaign == 'AI-Optimized' else random.uniform(1.1, 1.6)
        
        # Calculate metrics
        sales_volume = int(seasonal_demand * campaign_lift * ai_impact_multiplier * random.uniform(0.8, 1.2))
        revenue = sales_volume * current_price
        waste_percentage = max(0, (current_stock - sales_volume) / current_stock * 100) if current_stock > 0 else 0
        
        if st.session_state.ai_enabled and details['ai_priority'] == 'high':
            waste_percentage *= (1 - st.session_state.learning_rate * 0.3)
        
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
            'ai_priority': details['ai_priority'],
            'forecast_accuracy': 1 - abs(random.uniform(-0.2, 0.2)) * (1 - st.session_state.learning_rate * 0.5),
            'strategy_applied': st.session_state.ai_agent.strategy_name if 'ai_agent' in st.session_state else 'balanced'
        })
    
    return data

def generate_brd_document():
    """Generate comprehensive Business Requirements Document"""
    current_time = datetime.now()
    
    brd_content = f"""
# BUSINESS REQUIREMENTS DOCUMENT
## Self-Optimizing AI Business Intelligence Platform

### Document Information
- **Version**: 2.1
- **Date**: {current_time.strftime('%Y-%m-%d %H:%M:%S')}
- **Author**: AI Optimization Team
- **Status**: Active Implementation

---

## EXECUTIVE SUMMARY

### Business Objectives
- Increase revenue by 15-25% through AI-driven optimization
- Reduce operational waste by 40%
- Improve inventory turnover by 30%
- Achieve 90%+ forecast accuracy
- Automate 70% of routine business decisions

### Current Performance Summary
"""
    
    if st.session_state.optimization_history:
        total_improvement = sum([h['improvement'] for h in st.session_state.optimization_history])
        total_decisions = sum([h['ai_decisions'] for h in st.session_state.optimization_history])
        brd_content += f"""
- **Total Performance Improvement**: +{total_improvement:.1%}
- **AI Decisions Made**: {total_decisions}
- **Learning Rate**: {st.session_state.learning_rate:.1%}
- **Manual Interventions**: {len(st.session_state.intervention_alerts)} alerts
"""
    
    brd_content += f"""

## TECHNICAL ARCHITECTURE

### AI Optimization Functions
"""
    
    if st.session_state.function_level_metrics:
        latest_week = max(st.session_state.function_level_metrics.keys())
        for func, metrics in st.session_state.function_level_metrics[latest_week].items():
            success_rate = metrics['success_rate'] * 100
            brd_content += f"""
- **{func.replace('_', ' ').title()}**: {success_rate:.1f}% success rate, {metrics['decisions_made']} decisions
"""
    
    brd_content += f"""

## IMPLEMENTATION ROADMAP

### Phase 1: Foundation (Weeks 1-4)
- Data integration and warehouse setup
- Core AI algorithms implementation
- Basic dashboard deployment

### Phase 2: Optimization (Weeks 5-8)  
- Advanced AI decision engine
- Real-time performance monitoring
- Automated intervention detection

### Phase 3: Scale (Weeks 9-12)
- Full automation deployment
- Advanced analytics and reporting
- Performance optimization

## SUCCESS METRICS

### Quantitative Targets
- Revenue increase: 15-25%
- Waste reduction: 40%
- Forecast accuracy: 90%+
- Decision automation: 70%

### ROI Analysis
- Implementation cost: $425,000
- Expected first-year savings: $1,275,000
- Net ROI: 200%
- Break-even: 4 months

---
*Document generated on {current_time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    return brd_content

# Initialize AI Agent
if 'ai_agent' not in st.session_state:
    st.session_state.ai_agent = AIOptimizationAgent('balanced')

# Initialize sample data
if not st.session_state.business_data:
    for week in range(1, 8):
        week_data = generate_business_data(week)
        st.session_state.business_data.extend(week_data)
        
        if week > 1 and st.session_state.ai_enabled:
            current_data = pd.DataFrame([d for d in st.session_state.business_data if d['week'] == week])
            optimizations = st.session_state.ai_agent.generate_optimizations(current_data, week)
            st.session_state.ai_decisions.extend(optimizations)
            
            function_results = st.session_state.ai_agent.assess_function_performance(current_data, optimizations)
            st.session_state.ai_agent.check_intervention_needed(current_data, function_results)

# Main App Title
st.title("🤖 Self-Optimizing AI Business Platform")
st.markdown("### Comprehensive Business Intelligence with Organized Analytics")

# Sidebar Controls
st.sidebar.header("🎛️ AI Control Center")

# AI Strategy Selection
current_strategy = st.sidebar.selectbox(
    "AI Optimization Strategy",
    list(AI_STRATEGIES.keys()),
    format_func=lambda x: f"{x.title()} - {AI_STRATEGIES[x]['description']}"
)

if current_strategy != st.session_state.ai_agent.strategy_name:
    st.session_state.ai_agent = AIOptimizationAgent(current_strategy)

# AI Controls
st.session_state.ai_enabled = st.sidebar.toggle("🔄 Enable AI Optimization", value=st.session_state.ai_enabled)
st.session_state.auto_optimize = st.sidebar.toggle("⚡ Auto-Execute Decisions", value=st.session_state.auto_optimize)

# Week Selection
max_week = max([d['week'] for d in st.session_state.business_data]) if st.session_state.business_data else 1
selected_week = st.sidebar.slider("Business Week", min_value=1, max_value=max_week, value=max_week)

# Simulate Next Week
if st.sidebar.button("▶️ Run AI Simulation"):
    new_week = max_week + 1
    
    current_data = pd.DataFrame([d for d in st.session_state.business_data if d['week'] == max_week])
    ai_optimizations = st.session_state.ai_agent.generate_optimizations(current_data, new_week) if st.session_state.ai_enabled else []
    
    new_data = generate_business_data(new_week, ai_optimizations)
    st.session_state.business_data.extend(new_data)
    
    if ai_optimizations:
        st.session_state.ai_decisions.extend(ai_optimizations)
    
    # Performance tracking
    new_week_df = pd.DataFrame(new_data)
    function_results = st.session_state.ai_agent.assess_function_performance(new_week_df, ai_optimizations)
    st.session_state.ai_agent.check_intervention_needed(new_week_df, function_results)
    
    if max_week > 1:
        prev_week_data = pd.DataFrame([d for d in st.session_state.business_data if d['week'] == max_week])
        prev_revenue = prev_week_data['revenue'].sum()
        new_revenue = new_week_df['revenue'].sum()
        improvement = (new_revenue - prev_revenue) / prev_revenue if prev_revenue > 0 else 0
        
        st.session_state.optimization_history.append({
            'week': new_week,
            'improvement': improvement,
            'ai_decisions': len(ai_optimizations),
            'timestamp': datetime.now(),
            'strategy_used': st.session_state.ai_agent.strategy_name
        })
    
    st.session_state.learning_rate = min(0.95, st.session_state.learning_rate + random.uniform(0.01, 0.03))
    st.rerun()

# Process data for tabs
if st.session_state.business_data:
    df = pd.DataFrame(st.session_state.business_data)
    current_week_data = df[df['week'] == selected_week]
    
    # Create organized tabs
    tab1, tab2, tab3 = st.tabs(["📊 Business Level Metrics", "⚙️ Technical Level Metrics", "📈 Overall Performance & Optimization"])
    
    # ==========================================
    # TAB 1: BUSINESS LEVEL METRICS
    # ==========================================
    
    with tab1:
        st.header("📊 Business Level Metrics & KPIs")
        
        # Executive KPIs
        st.subheader("🎯 Executive Dashboard")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_revenue = current_week_data['revenue'].sum()
            prev_week_revenue = df[df['week'] == max(1, selected_week-1)]['revenue'].sum() if selected_week > 1 else total_revenue
            revenue_change = ((total_revenue - prev_week_revenue) / prev_week_revenue * 100) if prev_week_revenue > 0 else 0
            st.metric("💰 Weekly Revenue", f"${total_revenue:,.0f}", delta=f"{revenue_change:+.1f}%")
        
        with col2:
            avg_margin = current_week_data['profit_margin'].mean()
            st.metric("📈 Profit Margin", f"{avg_margin:.1f}%", delta="Healthy" if avg_margin > 25 else "Monitor")
        
        with col3:
            avg_waste = current_week_data['waste_percentage'].mean()
            st.metric("♻️ Waste Rate", f"{avg_waste:.1f}%", delta="Good" if avg_waste < 10 else "Action Needed")
        
        with col4:
            inventory_efficiency = current_week_data['inventory_turnover'].mean()
            st.metric("📦 Inventory Turnover", f"{inventory_efficiency:.2f}", delta="Optimal" if inventory_efficiency > 0.3 else "Slow")
        
        with col5:
            avg_forecast_accuracy = current_week_data['forecast_accuracy'].mean()
            st.metric("🎯 Forecast Accuracy", f"{avg_forecast_accuracy:.1%}", delta="Excellent" if avg_forecast_accuracy > 0.8 else "Improving")
        
        # Business Performance Charts
        st.subheader("📊 Business Performance Analysis")
        
        biz_col1, biz_col2 = st.columns(2)
        
        with biz_col1:
            # Revenue by Category
            category_perf = current_week_data.groupby('category').agg({
                'revenue': 'sum',
                'profit_margin': 'mean'
            }).reset_index()
            
            fig_category = px.bar(
                category_perf, 
                x='category', 
                y='revenue',
                title='💼 Revenue by Category',
                color='profit_margin',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_category, use_container_width=True)
        
        with biz_col2:
            # Campaign Effectiveness
            campaign_perf = current_week_data.groupby('campaign').agg({
                'revenue': 'sum',
                'actual_sales': 'sum'
            }).reset_index()
            
            fig_campaigns = px.pie(
                campaign_perf, 
                values='revenue', 
                names='campaign',
                title='📢 Campaign Revenue Distribution'
            )
            st.plotly_chart(fig_campaigns, use_container_width=True)
        
        # Business Insights
        st.subheader("💡 Business Intelligence Insights")
        
        insight_col1, insight_col2, insight_col3 = st.columns(3)
        
        with insight_col1:
            st.success("🚀 **Growth Opportunities**")
            top_performers = current_week_data.nlargest(3, 'revenue')
            for _, item in top_performers.iterrows():
                st.write(f"• {item['product']}: ${item['revenue']:,.0f}")
            
            high_margin_products = current_week_data[current_week_data['profit_margin'] > 30]
            st.write(f"• {len(high_margin_products)} products with >30% margin")
        
        with insight_col2:
            st.info("📊 **Market Analysis**")
            ai_optimized_count = len(current_week_data[current_week_data['ai_optimized'] == True])
            st.write(f"• {ai_optimized_count} products AI-optimized")
            
            avg_price_change = current_week_data['current_price'] / current_week_data['base_price']
            price_optimization_impact = (avg_price_change.mean() - 1) * 100
            st.write(f"• Avg price optimization: {price_optimization_impact:+.1f}%")
            
            seasonal_items = current_week_data[current_week_data['product'].str.contains('Seasonal|Winter')]
            if not seasonal_items.empty:
                st.write(f"• Seasonal revenue: ${seasonal_items['revenue'].sum():,.0f}")
        
        with insight_col3:
            st.warning("⚠️ **Risk Management**")
            expiring_soon = current_week_data[current_week_data['days_to_expiry'] <= 3]
            if not expiring_soon.empty:
                st.write(f"• {len(expiring_soon)} items expiring in 3 days")
                st.write(f"• Potential waste value: ${(expiring_soon['current_stock'] * expiring_soon['current_price']).sum():,.0f}")
            
            high_waste_items = current_week_data[current_week_data['waste_percentage'] > 15]
            if not high_waste_items.empty:
                st.write(f"• {len(high_waste_items)} high-waste products")
        
        # Product Performance Table
        st.subheader("📋 Detailed Product Performance")
        
        display_df = current_week_data[[
            'product', 'category', 'revenue', 'profit_margin', 'actual_sales',
            'current_price', 'waste_percentage', 'ai_optimized', 'campaign'
        ]].copy()
        
        st.dataframe(
            display_df.style.format({
                'revenue': '${:,.0f}',
                'profit_margin': '{:.1f}%',
                'current_price': '${:.2f}',
                'waste_percentage': '{:.1f}%',
                'actual_sales': '{:.0f}'
            }).background_gradient(subset=['revenue', 'profit_margin']),
            use_container_width=True
        )
    
    # ==========================================
    # TAB 2: TECHNICAL LEVEL METRICS
    # ==========================================
    
    with tab2:
        st.header("⚙️ Technical Level Metrics & AI Performance")
        
        # AI System Status
        st.subheader("🤖 AI System Status")
        tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
        
        with tech_col1:
            ai_status = "🟢 ACTIVE" if st.session_state.ai_enabled else "🔴 INACTIVE"
            st.metric("AI Agent Status", ai_status)
        
        with tech_col2:
            learning_progress = st.session_state.learning_rate
            st.metric("Learning Rate", f"{learning_progress:.1%}", delta=f"+{(learning_progress-0.8)*100:.1f}%")
        
        with tech_col3:
            total_decisions = len(st.session_state.ai_decisions)
            current_week_decisions = len([d for d in st.session_state.ai_decisions if d.get('week') == selected_week])
            st.metric("Total AI Decisions", total_decisions, delta=f"{current_week_decisions} this week")
        
        with tech_col4:
            intervention_count = len(st.session_state.intervention_alerts)
            intervention_status = "🟢 STABLE" if intervention_count == 0 else "🟡 MONITORING"
            st.metric("System Health", intervention_status, delta=f"{intervention_count} alerts")
        
        # Function-Level Performance
        st.subheader("⚙️ AI Function Performance Analysis")
        
        if st.session_state.function_level_metrics:
            latest_week = max(st.session_state.function_level_metrics.keys())
            function_data = st.session_state.function_level_metrics[latest_week]
            
            func_col1, func_col2 = st.columns(2)
            
            with func_col1:
                st.markdown("#### 🔧 Function Success Rates")
                for func_name, metrics in function_data.items():
                    func_display_name = func_name.replace('_', ' ').title()
                    success_rate = metrics['success_rate'] * 100
                    
                    # Color coding based on performance
                    if success_rate >= 80:
                        status_color = "🟢"
                        status_text = "Excellent"
                    elif success_rate >= 60:
                        status_color = "🟡"
                        status_text = "Good"
                    else:
                        status_color = "🔴"
                        status_text = "Needs Attention"
                    
                    st.write(f"{status_color} **{func_display_name}**: {success_rate:.1f}% ({status_text})")
                    st.write(f"   Decisions: {metrics['decisions_made']}, Impact: ${metrics['total_impact']:,.0f}")
            
            with func_col2:
                # Function performance chart
                func_names = [f.replace('_', ' ').title() for f in function_data.keys()]
                success_rates = [function_data[f]['success_rate'] * 100 for f in function_data.keys()]
                
                fig_func_perf = px.bar(
                    x=func_names,
                    y=success_rates,
                    title='📊 Function Success Rates',
                    color=success_rates,
                    color_continuous_scale='RdYlGn',
                    range_color=[0, 100]
                )
                fig_func_perf.update_layout(
                    yaxis_title="Success Rate (%)",
                    xaxis_title="AI Functions"
                )
                st.plotly_chart(fig_func_perf, use_container_width=True)
        
        # AI Decision Analysis
        st.subheader("🧠 AI Decision Intelligence")
        
        current_week_decisions = [d for d in st.session_state.ai_decisions if d.get('week') == selected_week]
        
        if current_week_decisions:
            decision_col1, decision_col2 = st.columns(2)
            
            with decision_col1:
                st.markdown("#### ⚡ Current Week Decisions")
                for i, decision in enumerate(current_week_decisions[:5]):  # Show top 5
                    confidence_color = "🟢" if decision['confidence'] > 0.8 else "🟡" if decision['confidence'] > 0.6 else "🟠"
                    
                    with st.expander(f"{confidence_color} {decision['action']}", expanded=i==0):
                        st.write(f"**Product:** {decision['product']}")
                        st.write(f"**Function:** {decision.get('function_type', 'general').replace('_', ' ').title()}")
                        st.write(f"**Confidence:** {decision['confidence']:.1%}")
                        st.write(f"**Expected Impact:** ${decision['expected_impact']:,.0f}")
                        st.write(f"**AI Reasoning:** {decision['reasoning']}")
            
            with decision_col2:
                # Decision type distribution
                decision_types = [d['type'] for d in current_week_decisions]
                decision_counts = pd.Series(decision_types).value_counts()
                
                fig_decision_types = px.pie(
                    values=decision_counts.values,
                    names=[name.replace('_', ' ').title() for name in decision_counts.index],
                    title='🎯 Decision Type Distribution'
                )
                st.plotly_chart(fig_decision_types, use_container_width=True)
        
        # Technical Performance Metrics
        st.subheader("📈 Technical Performance Indicators")
        
        tech_perf_col1, tech_perf_col2 = st.columns(2)
        
        with tech_perf_col1:
            # Algorithm performance over time
            if len(st.session_state.optimization_history) > 1:
                history_df = pd.DataFrame(st.session_state.optimization_history)
                
                fig_algo_perf = px.line(
                    history_df,
                    x='week',
                    y='improvement',
                    title='🤖 Algorithm Performance Over Time',
                    markers=True
                )
                fig_algo_perf.update_layout(
                    yaxis_title="Performance Improvement",
                    yaxis_tickformat='.1%'
                )
                st.plotly_chart(fig_algo_perf, use_container_width=True)
        
        with tech_perf_col2:
            # Learning rate progression
            weeks = list(range(1, selected_week + 1))
            simulated_learning = [0.85 + (w-1) * 0.015 + random.uniform(-0.01, 0.01) for w in weeks]
            simulated_learning = [min(0.95, max(0.8, rate)) for rate in simulated_learning]
            
            fig_learning = px.line(
                x=weeks,
                y=simulated_learning,
                title='📚 AI Learning Rate Progression',
                markers=True
            )
            fig_learning.update_layout(
                yaxis_title="Learning Rate",
                yaxis_tickformat='.1%',
                xaxis_title="Week"
            )
            st.plotly_chart(fig_learning, use_container_width=True)
        
        # System Alerts and Interventions
        st.subheader("🚨 System Alerts & Manual Intervention Needs")
        
        alert_col1, alert_col2 = st.columns(2)
        
        with alert_col1:
            if st.session_state.intervention_alerts:
                st.markdown("#### 🛠️ Active Alerts")
                
                high_priority = [a for a in st.session_state.intervention_alerts if a.get('severity') == 'high']
                medium_priority = [a for a in st.session_state.intervention_alerts if a.get('severity') == 'medium']
                
                if high_priority:
                    st.error("🔴 **HIGH PRIORITY**")
                    for alert in high_priority:
                        st.write(f"• {alert['message']}")
                        st.write(f"  *Action: {alert.get('recommended_action', 'Review required')}*")
                
                if medium_priority:
                    st.warning("🟡 **MEDIUM PRIORITY**")
                    for alert in medium_priority:
                        st.write(f"• {alert['message']}")
                        if alert.get('function'):
                            st.write(f"  *Function: {alert['function'].replace('_', ' ').title()}*")
                
                if not high_priority and not medium_priority:
                    st.success("✅ All systems operating normally")
            else:
                st.success("✅ **No Active Alerts**")
                st.write("All AI systems are functioning optimally.")
        
        with alert_col2:
            # System health indicators
            st.markdown("#### 📊 System Health Indicators")
            
            # Simulated system metrics
            system_metrics = {
                'CPU Usage': random.uniform(45, 75),
                'Memory Usage': random.uniform(60, 85),
                'AI Response Time': random.uniform(0.1, 0.3),
                'Data Processing Speed': random.uniform(85, 98)
            }
            
            for metric, value in system_metrics.items():
                if 'Usage' in metric:
                    color = 'normal' if value < 80 else 'inverse'
                    st.metric(metric, f"{value:.1f}%", delta="Normal" if value < 80 else "High")
                elif 'Time' in metric:
                    st.metric(metric, f"{value:.2f}s", delta="Fast")
                else:
                    st.metric(metric, f"{value:.1f}%", delta="Optimal")
        
        # Advanced Technical Diagnostics
        st.subheader("🔧 Advanced Diagnostics")
        
        diag_col1, diag_col2 = st.columns(2)
        
        with diag_col1:
            st.markdown("#### 🧪 Model Performance")
            
            # Model accuracy by function
            model_accuracy = {
                'Pricing Model': random.uniform(85, 95),
                'Inventory Model': random.uniform(80, 90),
                'Demand Forecast': random.uniform(75, 88),
                'Campaign Optimizer': random.uniform(82, 92)
            }
            
            for model, accuracy in model_accuracy.items():
                color = "🟢" if accuracy > 85 else "🟡" if accuracy > 75 else "🔴"
                st.write(f"{color} **{model}**: {accuracy:.1f}% accuracy")
        
        with diag_col2:
            st.markdown("#### 📡 Data Quality Metrics")
            
            data_quality = {
                'Data Completeness': random.uniform(95, 99),
                'Data Accuracy': random.uniform(92, 98),
                'Real-time Sync': random.uniform(98, 100),
                'Processing Speed': random.uniform(88, 96)
            }
            
            for metric, score in data_quality.items():
                color = "🟢" if score > 95 else "🟡" if score > 85 else "🔴"
                st.write(f"{color} **{metric}**: {score:.1f}%")
    
    # ==========================================
    # TAB 3: OVERALL PERFORMANCE & OPTIMIZATION
    # ==========================================
    
    with tab3:
        st.header("📈 Overall Performance & Optimization Journey")
        
        # Performance Overview
        st.subheader("🎯 Comprehensive Performance Overview")
        
        overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
        
        with overview_col1:
            if st.session_state.optimization_history:
                total_improvement = sum([h['improvement'] for h in st.session_state.optimization_history])
                st.metric("📊 Total Improvement", f"+{total_improvement:.1%}", delta="Cumulative")
            else:
                st.metric("📊 Total Improvement", "Initializing...", delta="Learning")
        
        with overview_col2:
            weeks_active = len(set(d['week'] for d in st.session_state.business_data))
            st.metric("📅 Weeks Optimized", weeks_active, delta="Active")
        
        with overview_col3:
            total_ai_decisions = len(st.session_state.ai_decisions)
            st.metric("🤖 Total AI Decisions", total_ai_decisions, delta="Automated")
        
        with overview_col4:
            current_learning_rate = st.session_state.learning_rate
            improvement_since_start = (current_learning_rate - 0.85) * 100
            st.metric("🧠 AI Intelligence", f"{current_learning_rate:.1%}", delta=f"+{improvement_since_start:.1f}%")
        
        # Gradual Improvement Analysis
        st.subheader("📈 Gradual Improvement Analysis")
        
        if len(st.session_state.optimization_history) > 1:
            improvement_col1, improvement_col2 = st.columns(2)
            
            with improvement_col1:
                # Weekly improvement trend
                history_df = pd.DataFrame(st.session_state.optimization_history)
                
                fig_improvement = go.Figure()
                
                # Add improvement line
                fig_improvement.add_trace(go.Scatter(
                    x=history_df['week'],
                    y=history_df['improvement'],
                    mode='lines+markers',
                    name='Weekly Improvement',
                    line=dict(color='blue', width=3),
                    marker=dict(size=8)
                ))
                
                # Add cumulative improvement
                cumulative_improvement = history_df['improvement'].cumsum()
                fig_improvement.add_trace(go.Scatter(
                    x=history_df['week'],
                    y=cumulative_improvement,
                    mode='lines+markers',
                    name='Cumulative Improvement',
                    line=dict(color='green', width=2, dash='dash'),
                    yaxis='y2'
                ))
                
                fig_improvement.update_layout(
                    title='📈 Performance Improvement Over Time',
                    xaxis_title='Week',
                    yaxis=dict(title='Weekly Improvement', tickformat='.1%'),
                    yaxis2=dict(title='Cumulative Improvement', overlaying='y', side='right', tickformat='.1%'),
                    legend=dict(x=0, y=1)
                )
                
                st.plotly_chart(fig_improvement, use_container_width=True)
            
            with improvement_col2:
                # AI decision effectiveness over time
                decision_effectiveness = []
                for _, row in history_df.iterrows():
                    effectiveness = min(1.0, 0.6 + row['improvement'] * 2)  # Simulated effectiveness
                    decision_effectiveness.append(effectiveness)
                
                fig_effectiveness = px.bar(
                    x=history_df['week'],
                    y=decision_effectiveness,
                    title='🎯 AI Decision Effectiveness',
                    color=decision_effectiveness,
                    color_continuous_scale='RdYlGn'
                )
                fig_effectiveness.update_layout(
                    yaxis_title="Effectiveness Score",
                    yaxis_tickformat='.1%'
                )
                st.plotly_chart(fig_effectiveness, use_container_width=True)
        
        # Optimization Strategy Analysis
        st.subheader("🎯 Optimization Strategy Deep Dive")
        
        strategy_col1, strategy_col2 = st.columns(2)
        
        with strategy_col1:
            st.markdown("#### 🧭 Current Strategy Performance")
            
            current_strategy = st.session_state.ai_agent.strategy_name
            strategy_config = st.session_state.ai_agent.strategy
            
            st.write(f"**Active Strategy:** {current_strategy.title()}")
            st.write(f"**Description:** {strategy_config['description']}")
            st.write(f"**Risk Tolerance:** {strategy_config['risk_tolerance']:.1%}")
            st.write(f"**Learning Speed:** {strategy_config['learning_speed']:.1%}")
            
            # Strategy effectiveness metrics
            if st.session_state.optimization_history:
                recent_improvements = [h['improvement'] for h in st.session_state.optimization_history[-3:]]
                avg_recent_improvement = np.mean(recent_improvements) if recent_improvements else 0
                
                if avg_recent_improvement > 0.05:
                    st.success(f"✅ **Highly Effective** - Avg improvement: +{avg_recent_improvement:.1%}")
                elif avg_recent_improvement > 0.02:
                    st.info(f"ℹ️ **Moderately Effective** - Avg improvement: +{avg_recent_improvement:.1%}")
                else:
                    st.warning(f"⚠️ **Needs Optimization** - Avg improvement: +{avg_recent_improvement:.1%}")
            
            # Strategy recommendations
            st.markdown("#### 💡 Strategy Optimization Recommendations")
            
            if current_strategy == 'conservative':
                st.info("💡 Consider upgrading to 'Balanced' for higher growth potential")
            elif current_strategy == 'aggressive':
                st.info("💡 Monitor risk levels - consider 'Balanced' if volatility is high")
            else:
                st.success("✅ Current strategy is well-balanced for most scenarios")
        
        with strategy_col2:
            # Key success factors
            st.markdown("#### 🏆 Key Success Factors")
            
            success_factors = {
                'Data Quality': random.uniform(85, 95),
                'Algorithm Accuracy': random.uniform(80, 92),
                'Decision Speed': random.uniform(88, 96),
                'Learning Rate': st.session_state.learning_rate * 100,
                'System Stability': random.uniform(92, 98)
            }
            
            fig_success = go.Figure(go.Bar(
                x=list(success_factors.values()),
                y=list(success_factors.keys()),
                orientation='h',
                marker_color=[
                    'green' if v > 90 else 'orange' if v > 75 else 'red' 
                    for v in success_factors.values()
                ]
            ))
            
            fig_success.update_layout(
                title='🏆 Success Factors Score',
                xaxis_title='Score (%)',
                xaxis_range=[0, 100]
            )
            
            st.plotly_chart(fig_success, use_container_width=True)
        
        # Achievement Timeline
        st.subheader("🏅 Optimization Achievement Timeline")
        
        # Create achievement milestones
        achievements = [
            {'week': 1, 'milestone': 'AI System Initialization', 'impact': 'Baseline established', 'status': 'completed'},
            {'week': 2, 'milestone': 'First Optimization Cycle', 'impact': '+3% improvement', 'status': 'completed'},
            {'week': 3, 'milestone': 'Dynamic Pricing Activated', 'impact': '+5% revenue boost', 'status': 'completed'},
            {'week': 4, 'milestone': 'Inventory Optimization', 'impact': '15% waste reduction', 'status': 'completed'},
            {'week': 5, 'milestone': 'Campaign Intelligence', 'impact': '+8% marketing ROI', 'status': 'completed'},
            {'week': 6, 'milestone': 'Advanced Learning', 'impact': '90%+ accuracy', 'status': 'in-progress' if selected_week >= 6 else 'pending'},
            {'week': 8, 'milestone': 'Full Automation', 'impact': '70% decision automation', 'status': 'pending'}
        ]
        
        timeline_col1, timeline_col2 = st.columns([2, 1])
        
        with timeline_col1:
            for achievement in achievements:
                if achievement['week'] <= selected_week:
                    if achievement['status'] == 'completed':
                        st.success(f"✅ **Week {achievement['week']}**: {achievement['milestone']}")
                        st.write(f"   Impact: {achievement['impact']}")
                    elif achievement['status'] == 'in-progress':
                        st.info(f"🔄 **Week {achievement['week']}**: {achievement['milestone']}")
                        st.write(f"   Expected: {achievement['impact']}")
                else:
                    st.write(f"⏳ **Week {achievement['week']}**: {achievement['milestone']}")
                    st.write(f"   Planned: {achievement['impact']}")
        
        with timeline_col2:
            # Progress indicator
            completed_milestones = len([a for a in achievements if a['week'] <= selected_week and a['status'] == 'completed'])
            total_milestones = len(achievements)
            progress_percentage = (completed_milestones / total_milestones) * 100
            
            fig_progress = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=progress_percentage,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Implementation Progress"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "lightgreen"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgray"},
                        {'range': [25, 50], 'color': "yellow"},
                        {'range': [50, 75], 'color': "orange"},
                        {'range': [75, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_progress.update_layout(height=300)
            st.plotly_chart(fig_progress, use_container_width=True)
        
        # ROI and Business Impact
        st.subheader("💰 Return on Investment & Business Impact")
        
        roi_col1, roi_col2, roi_col3 = st.columns(3)
        
        with roi_col1:
            st.markdown("#### 💵 Financial Impact")
            
            # Calculate cumulative financial impact
            if st.session_state.optimization_history:
                total_revenue_improvement = sum([h['improvement'] for h in st.session_state.optimization_history])
                baseline_revenue = current_week_data['revenue'].sum()
                estimated_annual_impact = baseline_revenue * total_revenue_improvement * 52
                
                st.metric("Annual Revenue Impact", f"${estimated_annual_impact:,.0f}")
                st.metric("Implementation Cost", "$425,000")
                
                if estimated_annual_impact > 425000:
                    roi = ((estimated_annual_impact - 425000) / 425000) * 100
                    st.metric("ROI", f"{roi:.0f}%", delta="Positive")
                else:
                    st.metric("ROI", "Calculating...", delta="In Progress")
        
        with roi_col2:
            st.markdown("#### ⏱️ Time to Value")
            
            months_to_breakeven = 425000 / (estimated_annual_impact / 12) if 'estimated_annual_impact' in locals() and estimated_annual_impact > 0 else 12
            
            st.metric("Break-even Period", f"{months_to_breakeven:.1f} months")
            st.metric("Time to 90% Accuracy", "6-8 weeks")
            st.metric("Full Automation", "12 weeks")
        
        with roi_col3:
            st.markdown("#### 🎯 Operational Benefits")
            
            st.write("**Quantified Benefits:**")
            st.write(f"• Waste reduction: {20-current_week_data['waste_percentage'].mean():.1f}%")
            st.write(f"• Forecast accuracy: {current_week_data['forecast_accuracy'].mean():.1%}")
            st.write(f"• Decision automation: 70%")
            st.write(f"• Processing speed: +150%")
        
        # Future Roadmap
        st.subheader("🚀 Optimization Roadmap & Future Enhancements")
        
        roadmap_phases = {
            'Phase 1 (Weeks 1-4)': {
                'status': 'Completed',
                'items': ['Core AI implementation', 'Basic optimization', 'Performance tracking']
            },
            'Phase 2 (Weeks 5-8)': {
                'status': 'In Progress',
                'items': ['Advanced algorithms', 'Multi-objective optimization', 'Real-time adaptation']
            },
            'Phase 3 (Weeks 9-12)': {
                'status': 'Planned',
                'items': ['Predictive analytics', 'Cross-channel optimization', 'Advanced reporting']
            },
            'Phase 4 (Weeks 13-16)': {
                'status': 'Future',
                'items': ['Machine learning enhancement', 'Autonomous scaling', 'Market adaptation']
            }
        }
        
        for phase, details in roadmap_phases.items():
            status_color = {
                'Completed': '🟢', 'In Progress': '🟡', 'Planned': '🔵', 'Future': '⚪'
            }.get(details['status'], '⚪')
            
            with st.expander(f"{status_color} {phase} - {details['status']}", expanded=details['status']=='In Progress'):
                for item in details['items']:
                    st.write(f"• {item}")
    
    # Download BRD Section (outside tabs, always visible)
    st.markdown("---")
    st.subheader("📋 Complete Business Requirements Document")
    
    brd_col1, brd_col2, brd_col3 = st.columns([2, 1, 1])
    
    with brd_col1:
        st.write("**Generate comprehensive documentation including:**")
        st.write("• Executive summary with current performance metrics")
        st.write("• Technical architecture and AI system specifications") 
        st.write("• Business impact analysis and ROI calculations")
        st.write("• Implementation roadmap and success criteria")
    
    with brd_col2:
        brd_content = generate_brd_document()
        st.download_button(
            label="📥 Download Complete BRD",
            data=brd_content,
            file_name=f"AI_Business_Requirements_Document_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            help="Complete business requirements document"
        )
    
    with brd_col3:
        # Performance data export
        performance_data = {
            'current_week_data': current_week_data.to_dict('records'),
            'optimization_history': st.session_state.optimization_history,
            'ai_decisions': st.session_state.ai_decisions,
            'function_metrics': st.session_state.function_level_metrics,
            'generated_at': datetime.now().isoformat()
        }
        
        st.download_button(
            label="📊 Export All Data",
            def normalize_dict_keys(d):
            if isinstance(d, dict):
                return {int(k) if isinstance(k, np.integer) else k: normalize_dict_keys(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [normalize_dict_keys(x) for x in d]
            else:
                return d
            
            data=json.dumps(performance_data, indent=2, default=str),
            file_name=f"ai_platform_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            help="Complete platform data export"
        )

# Enhanced Footer
st.markdown("---")
if st.session_state.optimization_history:
    total_improvement = sum([h['improvement'] for h in st.session_state.optimization_history])
    total_decisions = sum([h['ai_decisions'] for h in st.session_state.optimization_history])
    
    st.markdown(f"""
    <div style='text-align: center; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px;'>
    <h3>🎯 Platform Performance Summary</h3>
    <p><strong>Total Business Improvement:</strong> +{total_improvement:.1%} | <strong>AI Decisions:</strong> {total_decisions} | <strong>Intelligence Level:</strong> {st.session_state.learning_rate:.1%}</p>
    <p><strong>System Health:</strong> {len(st.session_state.intervention_alerts)} alerts | <strong>Weeks Optimized:</strong> {selected_week}</p>
    <p><em>Organized analytics delivering measurable business value through intelligent automation</em></p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style='text-align: center; background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 20px; border-radius: 10px;'>
    <h3>🚀 Self-Optimizing AI Platform Ready</h3>
    <p><strong>Organized Analytics Dashboard</strong> • Business Metrics • Technical Insights • Performance Optimization</p>
    <p><em>Run AI simulation to see comprehensive optimization in action</em></p>
    </div>
    """, unsafe_allow_html=True)
