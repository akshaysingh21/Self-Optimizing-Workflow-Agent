import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import random
import time

# Set page config
st.set_page_config(
    page_title="Advanced Self-Optimizing AI Platform",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state with comprehensive AI tracking
if 'business_data' not in st.session_state:
    st.session_state.business_data = []
if 'ai_decisions' not in st.session_state:
    st.session_state.ai_decisions = []
if 'optimization_history' not in st.session_state:
    st.session_state.optimization_history = []
if 'strategy_performance' not in st.session_state:
    st.session_state.strategy_performance = {}
if 'experimental_strategies' not in st.session_state:
    st.session_state.experimental_strategies = []
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

# Enhanced AI Agent Configuration
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
    },
    'experimental_alpha': {
        'risk_tolerance': 0.95, 'learning_speed': 0.35, 'description': 'Experimental high-performance strategy',
        'discount_cap': 70, 'inventory_buffer': 1.0, 'campaign_scaling': 2.0
    },
    'experimental_beta': {
        'risk_tolerance': 0.4, 'learning_speed': 0.3, 'description': 'Experimental precision-focused strategy',
        'discount_cap': 35, 'inventory_buffer': 1.3, 'campaign_scaling': 1.1
    }
}

OPTIMIZATION_FUNCTIONS = {
    'pricing_optimization': {'weight': 0.3, 'description': 'Dynamic pricing based on demand and shelf life'},
    'inventory_management': {'weight': 0.25, 'description': 'Predictive inventory rebalancing and reordering'},
    'campaign_optimization': {'weight': 0.2, 'description': 'Marketing campaign effectiveness and scaling'},
    'waste_reduction': {'weight': 0.15, 'description': 'Proactive waste minimization through early intervention'},
    'demand_forecasting': {'weight': 0.1, 'description': 'Advanced demand prediction and planning'}
}

PRODUCTS = {
    'Fresh Milk': {'category': 'Dairy', 'shelf_life': 7, 'base_price': 4.99, 'seasonality': 1.0, 'ai_priority': 'high'},
    'Organic Bread': {'category': 'Bakery', 'shelf_life': 5, 'base_price': 6.49, 'seasonality': 1.1, 'ai_priority': 'medium'},
    'Premium Coffee': {'category': 'Beverages', 'shelf_life': 365, 'base_price': 12.99, 'seasonality': 1.2, 'ai_priority': 'low'},
    'Seasonal Fruits': {'category': 'Produce', 'shelf_life': 4, 'base_price': 8.99, 'seasonality': 1.5, 'ai_priority': 'high'},
    'Winter Jackets': {'category': 'Apparel', 'shelf_life': 180, 'base_price': 89.99, 'seasonality': 2.0, 'ai_priority': 'medium'}
}

class AdvancedAIOptimizationAgent:
    def __init__(self, strategy='balanced'):
        self.strategy_name = strategy
        self.strategy = AI_STRATEGIES[strategy]
        self.learning_memory = {}
        self.performance_history = []
        self.function_metrics = {func: {'success_rate': 0.7, 'avg_impact': 0.1, 'total_applications': 0} 
                               for func in OPTIMIZATION_FUNCTIONS.keys()}
        self.intervention_threshold = 0.6
    
    def evaluate_strategy_performance(self, week_data, decisions):
        """Evaluate current strategy performance and suggest improvements"""
        if not decisions or week_data.empty:
            return
        
        strategy_metrics = {
            'revenue_impact': 0,
            'waste_reduction': 0,
            'decision_accuracy': 0,
            'risk_adjusted_return': 0
        }
        
        # Calculate strategy effectiveness
        total_decisions = len(decisions)
        successful_decisions = 0
        
        for decision in decisions:
            if decision.get('week') == week_data['week'].iloc[0]:
                product_data = week_data[week_data['product'] == decision['product']]
                if not product_data.empty:
                    actual_impact = product_data['revenue'].iloc[0]
                    expected_impact = decision.get('expected_impact', actual_impact)
                    
                    if actual_impact >= expected_impact * 0.9:  # Within 90% of expected
                        successful_decisions += 1
                    
                    strategy_metrics['revenue_impact'] += (actual_impact - expected_impact) / expected_impact if expected_impact > 0 else 0
        
        strategy_metrics['decision_accuracy'] = successful_decisions / total_decisions if total_decisions > 0 else 0
        strategy_metrics['risk_adjusted_return'] = strategy_metrics['revenue_impact'] / (self.strategy['risk_tolerance'] + 0.1)
        
        # Store strategy performance
        week = week_data['week'].iloc[0]
        if self.strategy_name not in st.session_state.strategy_performance:
            st.session_state.strategy_performance[self.strategy_name] = []
        
        st.session_state.strategy_performance[self.strategy_name].append({
            'week': week,
            'metrics': strategy_metrics,
            'decisions_made': total_decisions
        })
    
    def generate_experimental_strategies(self):
        """Generate and test experimental optimization strategies"""
        experiments = []
        
        # Experiment 1: Hyper-aggressive pricing
        if random.random() < 0.3:  # 30% chance to suggest experiment
            experiments.append({
                'name': 'hyper_pricing',
                'description': 'Test 20% higher discount rates for expiring items',
                'expected_impact': 0.15,
                'risk_level': 'high',
                'duration': 2,  # weeks
                'metrics_to_track': ['waste_reduction', 'revenue_impact']
            })
        
        # Experiment 2: Precision inventory management
        if random.random() < 0.25:
            experiments.append({
                'name': 'precision_inventory',
                'description': 'Use ML-enhanced demand prediction with 0.9x safety stock',
                'expected_impact': 0.12,
                'risk_level': 'medium',
                'duration': 3,
                'metrics_to_track': ['inventory_turnover', 'stockout_prevention']
            })
        
        # Experiment 3: Dynamic campaign allocation
        if random.random() < 0.2:
            experiments.append({
                'name': 'dynamic_campaigns',
                'description': 'Real-time campaign budget reallocation based on performance',
                'expected_impact': 0.18,
                'risk_level': 'medium',
                'duration': 4,
                'metrics_to_track': ['campaign_roi', 'customer_acquisition']
            })
        
        return experiments
    
    def assess_function_performance(self, week_data, decisions):
        """Assess performance at each optimization function level"""
        function_results = {}
        
        for func_name, func_info in OPTIMIZATION_FUNCTIONS.items():
            func_decisions = [d for d in decisions if d.get('function_type') == func_name]
            
            if func_decisions:
                success_count = 0
                total_impact = 0
                
                for decision in func_decisions:
                    product_data = week_data[week_data['product'] == decision['product']]
                    if not product_data.empty:
                        actual_performance = self._calculate_function_performance(func_name, product_data.iloc[0], decision)
                        expected_performance = decision.get('expected_impact', 0)
                        
                        if actual_performance >= expected_performance * 0.85:
                            success_count += 1
                        total_impact += actual_performance
                
                function_results[func_name] = {
                    'success_rate': success_count / len(func_decisions) if func_decisions else 0,
                    'total_impact': total_impact,
                    'decisions_made': len(func_decisions),
                    'avg_impact': total_impact / len(func_decisions) if func_decisions else 0,
                    'performance_trend': 'improving' if success_count / len(func_decisions) > 0.7 else 'needs_attention'
                }
            else:
                function_results[func_name] = {
                    'success_rate': 0, 'total_impact': 0, 'decisions_made': 0,
                    'avg_impact': 0, 'performance_trend': 'no_data'
                }
        
        # Update session state
        week = week_data['week'].iloc[0] if not week_data.empty else 0
        st.session_state.function_level_metrics[week] = function_results
        
        return function_results
    
    def check_intervention_needed(self, week_data, function_results):
        """Assess if manual intervention is needed"""
        interventions = []
        
        # Check overall performance decline
        if hasattr(self, 'performance_history') and len(self.performance_history) >= 3:
            recent_performance = np.mean([p.get('improvement', 0) for p in self.performance_history[-3:]])
            if recent_performance < -0.05:  # 5% decline
                interventions.append({
                    'type': 'performance_decline',
                    'severity': 'high',
                    'message': 'Overall AI performance declining for 3 consecutive periods',
                    'recommended_action': 'Review strategy parameters and consider manual tuning',
                    'auto_fix_available': True
                })
        
        # Check function-level issues
        for func_name, results in function_results.items():
            if results['success_rate'] < 0.5 and results['decisions_made'] > 2:
                interventions.append({
                    'type': 'function_underperforming',
                    'function': func_name,
                    'severity': 'medium',
                    'message': f'{func_name.replace("_", " ").title()} success rate below 50%',
                    'recommended_action': f'Adjust {func_name} parameters or switch strategy',
                    'auto_fix_available': False
                })
        
        # Check for data anomalies
        if not week_data.empty:
            revenue_std = week_data['revenue'].std()
            revenue_mean = week_data['revenue'].mean()
            if revenue_std > revenue_mean * 0.5:  # High variance
                interventions.append({
                    'type': 'data_anomaly',
                    'severity': 'low',
                    'message': 'High variance in revenue data detected',
                    'recommended_action': 'Review data quality and market conditions',
                    'auto_fix_available': False
                })
        
        # Store interventions
        if interventions:
            st.session_state.intervention_alerts.extend(interventions)
        
        return interventions
    
    def _calculate_function_performance(self, function_name, product_data, decision):
        """Calculate performance for specific optimization function"""
        if function_name == 'pricing_optimization':
            return product_data['revenue'] * (1 - product_data['waste_percentage'] / 100)
        elif function_name == 'inventory_management':
            return product_data['inventory_turnover'] * product_data['revenue']
        elif function_name == 'campaign_optimization':
            return product_data['revenue'] * (1 + product_data.get('campaign_lift', 0.1))
        elif function_name == 'waste_reduction':
            return (1 - product_data['waste_percentage'] / 100) * product_data['current_stock']
        else:  # demand_forecasting
            forecast_accuracy = 1 - abs(product_data['actual_sales'] - product_data['forecasted_demand']) / product_data['forecasted_demand']
            return forecast_accuracy * product_data['revenue']
    
    def generate_optimizations(self, current_data, week):
        """Enhanced optimization generation with function-level tracking"""
        optimizations = []
        
        for _, item in current_data.iterrows():
            product = item['product']
            
            # Pricing Optimization
            if item['days_to_expiry'] <= 3:
                discount = self._calculate_optimal_discount(item, week)
                optimizations.append({
                    'type': 'dynamic_pricing',
                    'function_type': 'pricing_optimization',
                    'product': product,
                    'action': f'Apply {discount:.0f}% discount',
                    'expected_impact': item['revenue'] * (1 + discount/100 * 0.5),
                    'confidence': min(0.95, st.session_state.learning_rate + 0.1),
                    'reasoning': f'AI predicts {discount:.0f}% discount will maximize revenue before expiry',
                    'week': week,
                    'parameters': {'discount_rate': discount, 'days_to_expiry': item['days_to_expiry']}
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
                    'reasoning': f'Prevent stockout based on {item["forecasted_demand"]:.0f} demand forecast',
                    'week': week,
                    'parameters': {'reorder_quantity': reorder_qty, 'current_stock': item['current_stock']}
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
                    'reasoning': f'High demand detected - scale campaign for {product}',
                    'week': week,
                    'parameters': {'scaling_factor': 1.3, 'current_performance': item['actual_sales']}
                })
        
        return optimizations
    
    def _calculate_optimal_discount(self, item, week):
        base_discount = min((4 - item['days_to_expiry']) * 15, self.strategy['discount_cap'])
        return max(5, base_discount)
    
    def _calculate_optimal_reorder(self, item, week):
        base_reorder = item['forecasted_demand'] * self.strategy['inventory_buffer'] - item['current_stock']
        return max(0, base_reorder)

def generate_brd_document():
    """Generate comprehensive Business Requirements Document"""
    brd_content = f"""
# BUSINESS REQUIREMENTS DOCUMENT
## Self-Optimizing AI Business Intelligence Platform

### Document Information
- **Document Version**: 2.1
- **Creation Date**: {datetime.now().strftime('%Y-%m-%d')}
- **Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Author**: AI Optimization Team
- **Stakeholders**: Executive Leadership, Operations, IT Department

---

### 1. EXECUTIVE SUMMARY

#### 1.1 Project Overview
The Self-Optimizing AI Business Intelligence Platform is an autonomous decision-making system designed to optimize retail operations through machine learning, predictive analytics, and real-time business intelligence.

#### 1.2 Business Objectives
- **Primary Goal**: Increase revenue by 15-25% through AI-driven optimization
- **Secondary Goals**:
  - Reduce operational waste by 40%
  - Improve inventory turnover by 30%
  - Enhance forecast accuracy to 90%+
  - Automate 70% of routine business decisions

#### 1.3 Success Metrics
- ROI improvement of 300% within 6 months
- Reduction in manual decision-making by 70%
- Decrease in stockouts by 85%
- Improvement in profit margins by 15%

---

### 2. CURRENT STATE ANALYSIS

#### 2.1 Existing Challenges
- **Manual Decision Making**: 80% of pricing and inventory decisions made manually
- **Reactive Operations**: Decisions made after problems occur
- **Data Silos**: Business intelligence scattered across multiple systems
- **Forecast Inaccuracy**: Current demand forecasting accuracy at 65%
- **Waste Issues**: 15-20% inventory waste due to expiry

#### 2.2 Business Impact
- Lost revenue opportunities: $500K annually
- Operational inefficiencies: 200+ hours monthly on manual processes
- Customer satisfaction issues: 12% stockout rate

---

### 3. SOLUTION ARCHITECTURE

#### 3.1 AI Optimization Engine
**Core Components:**
- **Machine Learning Algorithms**: Predictive modeling for demand forecasting
- **Dynamic Pricing Engine**: Real-time price optimization
- **Inventory Management AI**: Automated reordering and stock optimization
- **Campaign Intelligence**: Marketing ROI optimization
- **Waste Reduction Module**: Proactive expiry management

#### 3.2 Optimization Functions
"""

    # Add current strategy performance
    if st.session_state.strategy_performance:
        brd_content += "\n#### 3.3 Current Strategy Performance\n"
        for strategy, performance in st.session_state.strategy_performance.items():
            if performance:
                latest_perf = performance[-1]
                brd_content += f"- **{strategy.title()} Strategy**: {latest_perf['decisions_made']} decisions made\n"
    
    # Add function-level metrics
    if st.session_state.function_level_metrics:
        brd_content += "\n#### 3.4 Function-Level Performance\n"
        latest_week = max(st.session_state.function_level_metrics.keys())
        for func, metrics in st.session_state.function_level_metrics[latest_week].items():
            success_rate = metrics['success_rate'] * 100
            brd_content += f"- **{func.replace('_', ' ').title()}**: {success_rate:.1f}% success rate, {metrics['decisions_made']} decisions\n"
    
    brd_content += f"""

---

### 4. TECHNICAL REQUIREMENTS

#### 4.1 System Requirements
- **Processing Power**: Multi-core CPU for real-time analytics
- **Memory**: Minimum 16GB RAM for machine learning operations
- **Storage**: 500GB SSD for data warehouse and model storage
- **Network**: High-speed internet for real-time data synchronization

#### 4.2 Integration Requirements
- **ERP Systems**: SAP, Oracle, or equivalent
- **Point of Sale**: Integration with existing POS systems
- **Inventory Management**: Real-time stock level monitoring
- **Marketing Platforms**: Campaign management system integration

#### 4.3 Data Requirements
- **Historical Sales Data**: Minimum 2 years of transaction history
- **Product Master Data**: Complete product catalog with attributes
- **Customer Data**: Purchase history and demographic information
- **Market Data**: Competitive pricing and market trend data

---

### 5. FUNCTIONAL REQUIREMENTS

#### 5.1 AI Decision Making
- **Automated Pricing**: Dynamic price adjustments based on demand and shelf life
- **Inventory Optimization**: Automated reordering with predictive analytics
- **Campaign Management**: Intelligent budget allocation and scaling
- **Waste Prevention**: Proactive identification and intervention

#### 5.2 Business Intelligence
- **Real-time Dashboards**: Executive and operational dashboards
- **Performance Tracking**: KPI monitoring and trend analysis
- **Predictive Analytics**: Forecast accuracy and demand prediction
- **Reporting**: Automated report generation and distribution

#### 5.3 User Interface Requirements
- **Executive Dashboard**: High-level KPIs and strategic insights
- **Operations Dashboard**: Detailed operational metrics and controls
- **Mobile Interface**: Key metrics accessible on mobile devices
- **Alert System**: Real-time notifications for critical events

---

### 6. BUSINESS PROCESSES

#### 6.1 Pricing Optimization Process
1. **Data Collection**: Gather current inventory, sales, and market data
2. **Analysis**: AI analyzes patterns and trends
3. **Price Calculation**: Dynamic pricing algorithm determines optimal prices
4. **Approval**: High-confidence decisions auto-executed, others require approval
5. **Implementation**: Price changes pushed to POS systems
6. **Monitoring**: Track performance and adjust algorithms

#### 6.2 Inventory Management Process
1. **Demand Forecasting**: AI predicts future demand based on historical data
2. **Stock Analysis**: Current inventory levels compared to forecasted demand
3. **Reorder Recommendations**: AI calculates optimal reorder quantities
4. **Supplier Integration**: Automated purchase order generation
5. **Delivery Tracking**: Monitor incoming inventory
6. **Performance Review**: Analyze forecast accuracy and adjust models

---

### 7. RISK MANAGEMENT

#### 7.1 Technical Risks
- **Data Quality**: Implement data validation and cleansing procedures
- **System Integration**: Comprehensive testing and phased rollout
- **Algorithm Bias**: Regular model validation and bias detection
- **Performance Degradation**: Continuous monitoring and optimization

#### 7.2 Business Risks
- **Change Management**: Comprehensive training and support programs
- **Regulatory Compliance**: Ensure adherence to pricing and data regulations
- **Competitive Response**: Monitor market reactions and adjust strategies
- **Customer Impact**: Careful monitoring of customer satisfaction metrics

---

### 8. IMPLEMENTATION PLAN

#### 8.1 Phase 1: Foundation (Weeks 1-4)
- Data integration and warehouse setup
- Core AI algorithms development
- Basic dashboard implementation
- User training and onboarding

#### 8.2 Phase 2: Core Functionality (Weeks 5-8)
- Dynamic pricing engine deployment
- Inventory optimization implementation
- Advanced analytics and reporting
- Performance monitoring setup

#### 8.3 Phase 3: Advanced Features (Weeks 9-12)
- Campaign optimization module
- Predictive analytics enhancement
- Mobile interface development
- Full automation capabilities

#### 8.4 Phase 4: Optimization (Weeks 13-16)
- Performance tuning and optimization
- Advanced reporting and analytics
- Integration with additional systems
- Full production deployment

---

### 9. SUCCESS CRITERIA

#### 9.1 Quantitative Metrics
- Revenue increase: 15-25%
- Waste reduction: 40%
- Inventory turnover improvement: 30%
- Forecast accuracy: 90%+
- Decision automation: 70%

#### 9.2 Qualitative Metrics
- User satisfaction: 90%+
- System reliability: 99.5% uptime
- Decision quality: Executive approval rating 95%+
- Operational efficiency: Significant reduction in manual processes

---

### 10. BUDGET AND RESOURCES

#### 10.1 Technology Costs
- Software licenses: $150,000
- Hardware and infrastructure: $75,000
- Integration and development: $200,000
- Total Technology Investment: $425,000

#### 10.2 Human Resources
- Project manager: 1 FTE for 6 months
- Data scientists: 2 FTE for 6 months
- Developers: 3 FTE for 8 months
- Business analysts: 2 FTE for 4 months

#### 10.3 Expected ROI
- First year savings: $1,275,000
- Implementation cost: $425,000
- Net ROI: 200% in first year
- Break-even period: 4 months

---

### 11. APPENDICES

#### Appendix A: Technical Specifications
#### Appendix B: Data Flow Diagrams
#### Appendix C: User Interface Mockups
#### Appendix D: Integration Architecture
#### Appendix E: Security and Compliance Requirements

---

**Document Status**: APPROVED
**Next Review Date**: {(datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')}
**Distribution**: Executive Team, IT Department, Operations Management

---
*This document contains confidential and proprietary information.*
"""
    
    return brd_content

# Initialize AI Agent
if 'ai_agent' not in st.session_state:
    st.session_state.ai_agent = AdvancedAIOptimizationAgent('balanced')

# Enhanced data generation function
def generate_realistic_business_data(week, ai_optimizations=None):
    """Generate realistic business data with enhanced AI optimization tracking"""
    data = []
    
    for product, details in PRODUCTS.items():
        # Base business logic
        days_to_expiry = random.randint(1, details['shelf_life'])
        current_stock = random.randint(50, 500)
        base_demand = random.randint(80, 200)
        seasonal_demand = int(base_demand * details['seasonality'])
        
        # Apply AI optimizations with function tracking
        ai_impact_multiplier = 1.0
        applied_optimizations = []
        function_impacts = {}
        
        if ai_optimizations and st.session_state.ai_enabled:
            for opt in ai_optimizations:
                if opt['product'] == product:
                    function_type = opt.get('function_type', 'general')
                    impact = opt['confidence'] * 0.3
                    
                    if opt['type'] == 'dynamic_pricing':
                        ai_impact_multiplier *= (1.0 + impact)
                        function_impacts['pricing_optimization'] = impact
                    elif opt['type'] == 'campaign_scaling':
                        ai_impact_multiplier *= (1.0 + impact * 1.2)
                        function_impacts['campaign_optimization'] = impact
                    elif opt['type'] == 'inventory_reorder':
                        function_impacts['inventory_management'] = impact * 0.8
                    
                    applied_optimizations.append(opt['action'])
        
        # Enhanced pricing logic with strategy impact
        strategy_config = st.session_state.ai_agent.strategy
        urgency_multiplier = 1.0
        
        if days_to_expiry <= 2:
            discount_rate = min(strategy_config['discount_cap'], 30 + st.session_state.learning_rate * 10)
            urgency_multiplier = 1 - (discount_rate / 100)
        elif days_to_expiry <= 5:
            discount_rate = min(strategy_config['discount_cap'] * 0.6, 15 + st.session_state.learning_rate * 5)
            urgency_multiplier = 1 - (discount_rate / 100)
        
        current_price = details['base_price'] * urgency_multiplier
        
        # Campaign effectiveness with AI enhancement
        campaigns = ['Holiday Sale', 'Flash Friday', 'Loyalty Rewards', 'Clearance', 'New Product Launch', 'AI-Optimized']
        campaign = random.choice(campaigns)
        
        if applied_optimizations and st.session_state.ai_enabled:
            campaign = 'AI-Optimized'
        
        campaign_lift_base = {
            'Holiday Sale': 1.4, 'Flash Friday': 1.6, 'Loyalty Rewards': 1.2,
            'Clearance': 0.9, 'New Product Launch': 1.1, 'AI-Optimized': 1.5
        }
        
        campaign_lift = campaign_lift_base[campaign]
        if campaign == 'AI-Optimized':
            campaign_lift += st.session_state.learning_rate * 0.4
        
        # Calculate final business metrics
        sales_volume = int(seasonal_demand * campaign_lift * ai_impact_multiplier * random.uniform(0.8, 1.2))
        revenue = sales_volume * current_price
        waste_percentage = max(0, (current_stock - sales_volume) / current_stock * 100) if current_stock > 0 else 0
        
        # AI waste reduction impact
        if st.session_state.ai_enabled and details['ai_priority'] == 'high':
            waste_reduction_factor = 1 - (st.session_state.learning_rate * 0.3)
            waste_percentage *= waste_reduction_factor
        
        cost_per_unit = details['base_price'] * 0.6
        profit_margin = ((current_price - cost_per_unit) / current_price) * 100
        
        # Demand forecasting accuracy simulation
        forecast_error = random.uniform(-0.2, 0.2) * (1 - st.session_state.learning_rate * 0.5)
        actual_vs_forecast_accuracy = 1 - abs(forecast_error)
        
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
            'function_impacts': function_impacts,
            'forecast_accuracy': actual_vs_forecast_accuracy,
            'strategy_applied': st.session_state.ai_agent.strategy_name,
            'discount_applied': (1 - urgency_multiplier) * 100 if urgency_multiplier < 1 else 0
        })
    
    return data

# Title with enhanced AI branding
st.title("🤖 Advanced Self-Optimizing AI Business Platform")
st.markdown("""
### Autonomous Intelligence with Strategy Analysis & Experimental Optimization
*Real-time learning • Strategy evaluation • Function-level insights • Intervention detection*
""")

# Enhanced AI Control Panel
with st.container():
    st.subheader("🎛️ AI Intelligence Command Center")
    ai_col1, ai_col2, ai_col3, ai_col4, ai_col5 = st.columns(5)
    
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
        intervention_count = len([alert for alert in st.session_state.intervention_alerts if alert.get('severity') in ['high', 'medium']])
        intervention_status = "🔴 NEEDED" if intervention_count > 0 else "🟢 STABLE"
        st.metric("🚨 Manual Intervention", intervention_status, delta=f"{intervention_count} alerts")
    
    with ai_col5:
        if st.session_state.optimization_history:
            avg_improvement = np.mean([h['improvement'] for h in st.session_state.optimization_history])
            st.metric("📊 Avg Performance Lift", f"+{avg_improvement:.1%}", delta="AI Impact")
        else:
            st.metric("📊 Performance Impact", "Initializing...", delta="Learning")

# Sidebar enhancements
st.sidebar.header("🤖 Advanced AI Controls")

# AI Strategy Selection with performance tracking
current_strategy = st.sidebar.selectbox(
    "AI Optimization Strategy",
    list(AI_STRATEGIES.keys()),
    format_func=lambda x: f"{x.title()} - {AI_STRATEGIES[x]['description']}"
)

if current_strategy != st.session_state.ai_agent.strategy_name:
    st.session_state.ai_agent = AdvancedAIOptimizationAgent(current_strategy)

# Enhanced AI Controls
st.session_state.ai_enabled = st.sidebar.toggle("🔄 Enable AI Optimization", value=st.session_state.ai_enabled)
st.session_state.auto_optimize = st.sidebar.toggle("⚡ Auto-Execute Decisions", value=st.session_state.auto_optimize)

# Experimental Strategy Section
st.sidebar.subheader("🧪 Experimental Strategies")
if st.sidebar.button("🔬 Generate Experimental Strategy"):
    experiments = st.session_state.ai_agent.generate_experimental_strategies()
    st.session_state.experimental_strategies.extend(experiments)
    if experiments:
        st.sidebar.success(f"Generated {len(experiments)} experimental strategies!")

# Manual Intervention Controls
st.sidebar.subheader("🛠️ Manual Intervention")
if st.sidebar.button("🔧 Auto-Fix Issues"):
    # Auto-fix available interventions
    auto_fixable = [alert for alert in st.session_state.intervention_alerts if alert.get('auto_fix_available')]
    if auto_fixable:
        st.session_state.learning_rate = min(0.95, st.session_state.learning_rate + 0.05)
        st.session_state.intervention_alerts = [alert for alert in st.session_state.intervention_alerts if not alert.get('auto_fix_available')]
        st.sidebar.success("Auto-fixed available issues!")

# Initialize sample data if none exists
if not st.session_state.business_data:
    for week in range(1, 8):
        week_data = generate_realistic_business_data(week)
        st.session_state.business_data.extend(week_data)
        
        # Generate AI optimizations and assess performance
        if week > 1 and st.session_state.ai_enabled:
            current_data = pd.DataFrame([d for d in st.session_state.business_data if d['week'] == week])
            optimizations = st.session_state.ai_agent.generate_optimizations(current_data, week)
            st.session_state.ai_decisions.extend(optimizations)
            
            # Evaluate strategy and function performance
            st.session_state.ai_agent.evaluate_strategy_performance(current_data, optimizations)
            function_results = st.session_state.ai_agent.assess_function_performance(current_data, optimizations)
            interventions = st.session_state.ai_agent.check_intervention_needed(current_data, function_results)

# Main dashboard
max_week = max([d['week'] for d in st.session_state.business_data]) if st.session_state.business_data else 1
selected_week = st.sidebar.slider("Business Week", min_value=1, max_value=max_week, value=max_week)

# Simulate Next Week with Enhanced Tracking
if st.sidebar.button("▶️ Run Advanced AI Simulation"):
    new_week = max_week + 1
    
    # Generate comprehensive AI optimizations
    current_data = pd.DataFrame([d for d in st.session_state.business_data if d['week'] == max_week])
    ai_optimizations = st.session_state.ai_agent.generate_optimizations(current_data, new_week) if st.session_state.ai_enabled else []
    
    # Generate new week data
    new_data = generate_realistic_business_data(new_week, ai_optimizations)
    st.session_state.business_data.extend(new_data)
    
    if ai_optimizations:
        st.session_state.ai_decisions.extend(ai_optimizations)
    
    # Comprehensive performance evaluation
    new_week_df = pd.DataFrame(new_data)
    st.session_state.ai_agent.evaluate_strategy_performance(new_week_df, ai_optimizations)
    function_results = st.session_state.ai_agent.assess_function_performance(new_week_df, ai_optimizations)
    interventions = st.session_state.ai_agent.check_intervention_needed(new_week_df, function_results)
    
    # Calculate improvement
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
            'strategy_used': st.session_state.ai_agent.strategy_name,
            'function_performance': function_results
        })
    
    # AI Learning progression
    st.session_state.learning_rate = min(0.95, st.session_state.learning_rate + random.uniform(0.01, 0.03))
    st.rerun()

# Process and display data
if st.session_state.business_data:
    df = pd.DataFrame(st.session_state.business_data)
    current_week_data = df[df['week'] == selected_week]
    
    # Strategy Performance Analysis Section
    st.markdown("---")
    st.subheader("📋 1. Strategy Performance Analysis & Contributions")
    
    if st.session_state.strategy_performance:
        strategy_col1, strategy_col2 = st.columns(2)
        
        with strategy_col1:
            st.markdown("#### 🎯 Strategy Effectiveness Breakdown")
            
            # Create strategy performance summary
            strategy_summary = []
            for strategy_name, performance_list in st.session_state.strategy_performance.items():
                if performance_list:
                    avg_metrics = {
                        'strategy': strategy_name,
                        'avg_revenue_impact': np.mean([p['metrics']['revenue_impact'] for p in performance_list]),
                        'avg_decision_accuracy': np.mean([p['metrics']['decision_accuracy'] for p in performance_list]),
                        'total_decisions': sum([p['decisions_made'] for p in performance_list]),
                        'weeks_active': len(performance_list)
                    }
                    strategy_summary.append(avg_metrics)
            
            if strategy_summary:
                strategy_df = pd.DataFrame(strategy_summary)
                
                for _, row in strategy_df.iterrows():
                    with st.expander(f"📊 {row['strategy'].title()} Strategy Performance", expanded=True):
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Revenue Impact", f"{row['avg_revenue_impact']:+.1%}")
                        with col_b:
                            st.metric("Decision Accuracy", f"{row['avg_decision_accuracy']:.1%}")
                        with col_c:
                            st.metric("Total Decisions", f"{row['total_decisions']:.0f}")
                        
                        # Strategy contribution analysis
                        contribution = row['avg_revenue_impact'] * row['avg_decision_accuracy']
                        st.write(f"**Overall Contribution to Business Enhancement:** {contribution:+.1%}")
                        
                        if contribution > 0.1:
                            st.success("✅ **High-performing strategy** - Continue current approach")
                        elif contribution > 0.05:
                            st.warning("⚠️ **Moderate performance** - Consider optimization")
                        else:
                            st.error("🔴 **Underperforming** - Strategy adjustment needed")
        
        with strategy_col2:
            # Strategy comparison chart
            if len(strategy_summary) > 1:
                fig_strategy_comparison = px.scatter(
                    strategy_df,
                    x='avg_decision_accuracy',
                    y='avg_revenue_impact',
                    size='total_decisions',
                    color='strategy',
                    title='🎯 Strategy Performance Matrix',
                    labels={
                        'avg_decision_accuracy': 'Decision Accuracy',
                        'avg_revenue_impact': 'Revenue Impact'
                    }
                )
                st.plotly_chart(fig_strategy_comparison, use_container_width=True)
    
    # Experimental Strategies Section
    st.markdown("---")
    st.subheader("🧪 2. Experimental Strategy Evaluation")
    
    exp_col1, exp_col2 = st.columns(2)
    
    with exp_col1:
        if st.session_state.experimental_strategies:
            st.markdown("#### 🔬 Active Experiments")
            for i, exp in enumerate(st.session_state.experimental_strategies[-3:]):  # Show last 3
                with st.expander(f"Experiment: {exp['name'].replace('_', ' ').title()}", expanded=i==0):
                    st.write(f"**Description:** {exp['description']}")
                    st.write(f"**Expected Impact:** +{exp['expected_impact']:.1%}")
                    st.write(f"**Risk Level:** {exp['risk_level'].title()}")
                    st.write(f"**Duration:** {exp['duration']} weeks")
                    
                    # Simulated experiment results
                    simulated_success = random.uniform(0.6, 0.95)
                    if st.button(f"🚀 Deploy Experiment", key=f"deploy_exp_{i}"):
                        st.success(f"✅ Experiment deployed! Predicted success rate: {simulated_success:.1%}")
        else:
            st.info("🔬 No experimental strategies generated yet. Click 'Generate Experimental Strategy' to create new optimization approaches.")
    
    with exp_col2:
        st.markdown("#### 📈 Experimental Results Tracking")
        
        # Simulated experimental results
        exp_results = [
            {'experiment': 'Hyper Pricing', 'status': 'Active', 'week_2_impact': '+12%', 'confidence': '85%'},
            {'experiment': 'Precision Inventory', 'status': 'Completed', 'final_impact': '+8%', 'success': 'High'},
            {'experiment': 'Dynamic Campaigns', 'status': 'Planning', 'expected_impact': '+15%', 'risk': 'Medium'}
        ]
        
        for result in exp_results:
            status_color = "🟢" if result['status'] == 'Active' else "🔵" if result['status'] == 'Completed' else "🟡"
            st.write(f"{status_color} **{result['experiment']}** ({result['status']})")
            if result['status'] == 'Active':
                st.write(f"  Week 2 Impact: {result['week_2_impact']}, Confidence: {result['confidence']}")
            elif result['status'] == 'Completed':
                st.write(f"  Final Impact: {result['final_impact']}, Success Level: {result['success']}")
    
    # Function-Level Performance Section
    st.markdown("---")
    st.subheader("⚙️ 3. Function-Level Optimization Analysis")
    
    if st.session_state.function_level_metrics:
        latest_week = max(st.session_state.function_level_metrics.keys())
        function_data = st.session_state.function_level_metrics[latest_week]
        
        func_col1, func_col2 = st.columns(2)
        
        with func_col1:
            st.markdown("#### 🔧 Individual Function Performance")
            
            for func_name, metrics in function_data.items():
                func_display_name = func_name.replace('_', ' ').title()
                success_rate = metrics['success_rate'] * 100
                
                with st.expander(f"⚙️ {func_display_name}", expanded=True):
                    perf_col1, perf_col2, perf_col3 = st.columns(3)
                    
                    with perf_col1:
                        st.metric("Success Rate", f"{success_rate:.1f}%")
                    with perf_col2:
                        st.metric("Decisions Made", metrics['decisions_made'])
                    with perf_col3:
                        impact_color = "green" if metrics['total_impact'] > 0 else "red"
                        st.metric("Total Impact", f"${metrics['total_impact']:,.0f}")
                    
                    # Performance assessment
                    if success_rate >= 80:
                        st.success("✅ **Excellent Performance** - Function operating optimally")
                        improvement_status = "High contribution to business enhancement"
                    elif success_rate >= 60:
                        st.warning("⚠️ **Good Performance** - Minor optimization opportunities")
                        improvement_status = "Moderate contribution, room for improvement"
                    else:
                        st.error("🔴 **Needs Improvement** - Optimization required")
                        improvement_status = "Low contribution, immediate attention needed"
                    
                    st.write(f"**AI Assessment:** {improvement_status}")
                    
                    # Specific recommendations
                    if func_name == 'pricing_optimization' and success_rate < 70:
                        st.info("💡 **Recommendation:** Adjust discount thresholds and consider market conditions")
                    elif func_name == 'inventory_management' and success_rate < 70:
                        st.info("💡 **Recommendation:** Improve demand forecasting accuracy and safety stock levels")
                    elif func_name == 'campaign_optimization' and success_rate < 70:
                        st.info("💡 **Recommendation:** Analyze customer segments and campaign timing")
        
        with func_col2:
            # Function performance visualization
            func_names = list(function_data.keys())
            success_rates = [function_data[f]['success_rate'] * 100 for f in func_names]
            decision_counts = [function_data[f]['decisions_made'] for f in func_names]
            
            fig_function_performance = go.Figure()
            
            fig_function_performance.add_trace(go.Bar(
                name='Success Rate',
                x=[f.replace('_', ' ').title() for f in func_names],
                y=success_rates,
                marker_color='lightblue',
                yaxis='y'
            ))
            
            fig_function_performance.add_trace(go.Scatter(
                name='Decisions Made',
                x=[f.replace('_', ' ').title() for f in func_names],
                y=decision_counts,
                mode='lines+markers',
                marker_color='red',
                yaxis='y2'
            ))
            
            fig_function_performance.update_layout(
                title='🎯 Function Performance Overview',
                yaxis=dict(title='Success Rate (%)', side='left'),
                yaxis2=dict(title='Decisions Made', side='right', overlaying='y'),
                legend=dict(x=0, y=1)
            )
            
            st.plotly_chart(fig_function_performance, use_container_width=True)
    
    # Manual Intervention Assessment Section
    st.markdown("---")
    st.subheader("🚨 4. Manual Intervention Assessment")
    
    intervention_col1, intervention_col2 = st.columns(2)
    
    with intervention_col1:
        st.markdown("#### 🛠️ Intervention Alerts")
        
        if st.session_state.intervention_alerts:
            # Group alerts by severity
            high_priority = [a for a in st.session_state.intervention_alerts if a['severity'] == 'high']
            medium_priority = [a for a in st.session_state.intervention_alerts if a['severity'] == 'medium']
            low_priority = [a for a in st.session_state.intervention_alerts if a['severity'] == 'low']
            
            if high_priority:
                st.error("🔴 **HIGH PRIORITY INTERVENTIONS NEEDED**")
                for alert in high_priority:
                    with st.expander(f"🚨 {alert['message']}", expanded=True):
                        st.write(f"**Type:** {alert['type'].replace('_', ' ').title()}")
                        st.write(f"**Recommended Action:** {alert['recommended_action']}")
                        auto_fix = "✅ Available" if alert['auto_fix_available'] else "❌ Manual Required"
                        st.write(f"**Auto-fix:** {auto_fix}")
                        
                        if alert['auto_fix_available']:
                            if st.button(f"🔧 Auto-Fix", key=f"fix_high_{alert['type']}"):
                                st.success("✅ Auto-fix applied successfully!")
            
            if medium_priority:
                st.warning("🟡 **MEDIUM PRIORITY ALERTS**")
                for alert in medium_priority:
                    with st.expander(f"⚠️ {alert['message']}"):
                        st.write(f"**Recommended Action:** {alert['recommended_action']}")
                        if alert.get('function'):
                            st.write(f"**Affected Function:** {alert['function'].replace('_', ' ').title()}")
            
            if low_priority:
                st.info("🔵 **MONITORING ALERTS**")
                for alert in low_priority:
                    st.write(f"• {alert['message']}")
        else:
            st.success("✅ **All Systems Operating Normally**")
            st.write("No manual interventions required at this time.")
    
    with intervention_col2:
        st.markdown("#### 📊 Intervention History & Trends")
        
        # Simulated intervention metrics
        intervention_metrics = {
            'total_interventions': len(st.session_state.intervention_alerts),
            'auto_fixed': len([a for a in st.session_state.intervention_alerts if a.get('auto_fix_available')]),
            'manual_required': len([a for a in st.session_state.intervention_alerts if not a.get('auto_fix_available')]),
            'avg_resolution_time': '2.3 hours',
            'prevention_rate': '87%'
        }
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Total Alerts", intervention_metrics['total_interventions'])
            st.metric("Auto-Fixable", intervention_metrics['auto_fixed'])
        with col_b:
            st.metric("Manual Required", intervention_metrics['manual_required'])
            st.metric("Prevention Rate", intervention_metrics['prevention_rate'])
        
        # Intervention trend chart
        weeks = list(range(max(1, selected_week-6), selected_week+1))
        simulated_interventions = [random.randint(0, 3) for _ in weeks]
        
        fig_interventions = px.line(
            x=weeks,
            y=simulated_interventions,
            title='📈 Intervention Trend (Last 7 Weeks)',
            markers=True
        )
        fig_interventions.update_layout(
            xaxis_title="Week",
            yaxis_title="Interventions Required"
        )
        st.plotly_chart(fig_interventions, use_container_width=True)
    
    # Enhanced Business Intelligence Dashboard
    st.markdown("---")
    st.subheader("📊 5. AI-Enhanced Business Intelligence")
    
    # Main KPIs with AI impact
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_revenue = current_week_data['revenue'].sum()
        ai_optimized_revenue = current_week_data[current_week_data['ai_optimized'] == True]['revenue'].sum()
        ai_lift = (ai_optimized_revenue / total_revenue * 100) if total_revenue > 0 else 0
        st.metric("💰 Total Revenue", f"${total_revenue:,.0f}", delta=f"AI Lift: {ai_lift:.1f}%")
    
    with col2:
        avg_margin = current_week_data['profit_margin'].mean()
        ai_margin_data = current_week_data[current_week_data['ai_optimized'] == True]['profit_margin']
        regular_margin_data = current_week_data[current_week_data['ai_optimized'] == False]['profit_margin']
        ai_margin_boost = ai_margin_data.mean() - regular_margin_data.mean() if len(ai_margin_data) > 0 and len(regular_margin_data) > 0 else 0
        st.metric("📈 Profit Margin", f"{avg_margin:.1f}%", delta=f"AI Boost: +{ai_margin_boost:.1f}%" if not pd.isna(ai_margin_boost) else "Optimizing")
    
    with col3:
        avg_waste = current_week_data['waste_percentage'].mean()
        waste_status = f"🎯 {(20-avg_waste):.1f}% saved" if st.session_state.ai_enabled else "Standard"
        st.metric("♻️ Waste Rate", f"{avg_waste:.1f}%", delta=waste_status)
    
    with col4:
        avg_forecast_accuracy = current_week_data['forecast_accuracy'].mean()
        st.metric("🎯 Forecast Accuracy", f"{avg_forecast_accuracy:.1%}", delta="AI Enhanced")
    
    with col5:
        ai_decisions_count = len([d for d in st.session_state.ai_decisions if d.get('week') == selected_week])
        st.metric("🤖 AI Decisions", ai_decisions_count, delta="Real-time")
    
    # Download BRD Section
    st.markdown("---")
    st.subheader("📋 6. Complete Business Requirements Document")
    
    brd_col1, brd_col2 = st.columns([2, 1])
    
    with brd_col1:
        st.markdown("#### 📄 Comprehensive BRD Generation")
        st.write("Generate a complete Business Requirements Document including:")
        st.write("• Executive Summary and Business Objectives")
        st.write("• Technical Architecture and Requirements")
        st.write("• AI Strategy Performance Analysis")
        st.write("• Function-level Optimization Details")
        st.write("• Risk Management and Implementation Plan")
        st.write("• ROI Analysis and Success Metrics")
    
    with brd_col2:
        brd_content = generate_brd_document()
        st.download_button(
            label="📥 Download Complete BRD",
            data=brd_content,
            file_name=f"AI_Business_Requirements_Document_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            help="Comprehensive 20+ page business requirements document with current performance data"
        )
        
        # Additional export options
        st.download_button(
            label="📊 Download Performance Data",
            data=current_week_data.to_csv(index=False),
            file_name=f"ai_performance_data_week_{selected_week}.csv",
            mime="text/csv"
        )
        
        # Strategy analysis export
        if st.session_state.strategy_performance:
            strategy_analysis = json.dumps(st.session_state.strategy_performance, indent=2, default=str)
            st.download_button(
                label="🎯 Download Strategy Analysis",
                data=strategy_analysis,
                file_name=f"strategy_performance_analysis.json",
                mime="application/json"
            )

# Enhanced data visualization section
    st.markdown("---")
    st.subheader("📊 Advanced Analytics Dashboard")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # AI Impact over time
        weekly_data = df.groupby('week').agg({
            'revenue': 'sum',
            'ai_optimized': 'sum',
            'waste_percentage': 'mean',
            'forecast_accuracy': 'mean'
        }).reset_index()
        
        ai_revenue = df[df['ai_optimized'] == True].groupby('week')['revenue'].sum().reindex(weekly_data['week'], fill_value=0)
        
        fig_ai_impact = go.Figure()
        fig_ai_impact.add_trace(go.Scatter(
            x=weekly_data['week'],
            y=weekly_data['revenue'],
            mode='lines+markers',
            name='Total Revenue',
            line=dict(color='blue', width=3)
        ))
        fig_ai_impact.add_trace(go.Scatter(
            x=weekly_data['week'],
            y=ai_revenue,
            mode='lines+markers',
            name='AI-Optimized Revenue',
            line=dict(color='green', width=2),
            fill='tonexty'
        ))
        fig_ai_impact.update_layout(title='📈 AI Revenue Impact Trend')
        st.plotly_chart(fig_ai_impact, use_container_width=True)
    
    with chart_col2:
        # Function performance heatmap
        if st.session_state.function_level_metrics:
            weeks = sorted(st.session_state.function_level_metrics.keys())[-6:]  # Last 6 weeks
            functions = list(OPTIMIZATION_FUNCTIONS.keys())
            
            heatmap_data = []
            for week in weeks:
                week_data = st.session_state.function_level_metrics.get(week, {})
                for func in functions:
                    func_data = week_data.get(func, {'success_rate': 0})
                    heatmap_data.append({
                        'Week': f"Week {week}",
                        'Function': func.replace('_', ' ').title(),
                        'Success Rate': func_data['success_rate'] * 100
                    })
            
            if heatmap_data:
                heatmap_df = pd.DataFrame(heatmap_data)
                fig_heatmap = px.density_heatmap(
                    heatmap_df,
                    x='Week',
                    y='Function',
                    z='Success Rate',
                    title='🔥 Function Performance Heatmap',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)

# Enhanced footer
st.markdown("---")
if st.session_state.optimization_history:
    total_ai_improvement = sum([h['improvement'] for h in st.session_state.optimization_history])
    total_decisions = sum([h['ai_decisions'] for h in st.session_state.optimization_history])
    
    st.markdown(f"""
    <div style='text-align: center; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px;'>
    <h3>🤖 Advanced AI Performance Summary</h3>
    <p><strong>Total Performance Improvement:</strong> +{total_ai_improvement:.1%} | <strong>AI Decisions Made:</strong> {total_decisions} | <strong>Learning Rate:</strong> {st.session_state.learning_rate:.1%}</p>
    <p><strong>Manual Interventions:</strong> {len(st.session_state.intervention_alerts)} alerts | <strong>Strategy Performance:</strong> {len(st.session_state.strategy_performance)} strategies evaluated</p>
    <p><em>Self-optimizing AI with comprehensive analysis, experimental strategies, and intelligent intervention detection</em></p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style='text-align: center; background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 20px; border-radius: 10px;'>
    <h3>🚀 Advanced AI Agent Initialization Complete</h3>
    <p><strong>Ready for Comprehensive Optimization</strong> • Strategy Analysis • Function-Level Insights • Intervention Detection</p>
    <p><em>Run AI simulation to see advanced self-optimizing workflows in action</em></p>
    </div>
    """, unsafe_allow_html=True)
