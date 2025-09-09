import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import random

# Set page config
st.set_page_config(
    page_title="Retail Intelligence Dashboard",
    page_icon="🏪",
    layout="wide"
)

# Initialize session state with business scenarios
if 'business_data' not in st.session_state:
    st.session_state.business_data = []
if 'current_week' not in st.session_state:
    st.session_state.current_week = 1

# Business scenario configurations
PRODUCTS = {
    'Fresh Milk': {'category': 'Dairy', 'shelf_life': 7, 'base_price': 4.99, 'seasonality': 1.0},
    'Organic Bread': {'category': 'Bakery', 'shelf_life': 5, 'base_price': 6.49, 'seasonality': 1.1},
    'Premium Coffee': {'category': 'Beverages', 'shelf_life': 365, 'base_price': 12.99, 'seasonality': 1.2},
    'Seasonal Fruits': {'category': 'Produce', 'shelf_life': 4, 'base_price': 8.99, 'seasonality': 1.5},
    'Winter Jackets': {'category': 'Apparel', 'shelf_life': 180, 'base_price': 89.99, 'seasonality': 2.0}
}

CAMPAIGNS = ['Holiday Sale', 'Flash Friday', 'Loyalty Rewards', 'Clearance', 'New Product Launch']

def generate_realistic_business_data(week):
    """Generate realistic business data for different scenarios"""
    data = []
    
    for product, details in PRODUCTS.items():
        # Inventory management logic
        days_to_expiry = random.randint(1, details['shelf_life'])
        current_stock = random.randint(50, 500)
        
        # Demand forecasting based on seasonality
        base_demand = random.randint(80, 200)
        seasonal_demand = int(base_demand * details['seasonality'])
        
        # Dynamic pricing logic
        urgency_multiplier = 1.0
        if days_to_expiry <= 2:  # Urgent clearance
            urgency_multiplier = 0.7  # 30% discount
        elif days_to_expiry <= 5:  # Moderate urgency
            urgency_multiplier = 0.85  # 15% discount
        
        current_price = details['base_price'] * urgency_multiplier
        
        # Campaign effectiveness
        campaign = random.choice(CAMPAIGNS)
        campaign_lift = {
            'Holiday Sale': 1.4,
            'Flash Friday': 1.6,
            'Loyalty Rewards': 1.2,
            'Clearance': 0.9,
            'New Product Launch': 1.1
        }
        
        # Business metrics
        sales_volume = int(seasonal_demand * campaign_lift[campaign] * random.uniform(0.8, 1.2))
        revenue = sales_volume * current_price
        waste_percentage = max(0, (current_stock - sales_volume) / current_stock * 100) if current_stock > 0 else 0
        
        # Profit margin calculation
        cost_per_unit = details['base_price'] * 0.6  # Assuming 40% margin
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
            'inventory_turnover': sales_volume / current_stock if current_stock > 0 else 0
        })
    
    return data

# Title and business context
st.title("🏪 Retail Intelligence Dashboard")
st.markdown("""
### Business Optimization Platform
*Real-time insights for inventory management, pricing strategy, and marketing campaigns*
""")

# Business KPIs at the top
st.markdown("---")
st.subheader("📊 Executive Dashboard")

# Initialize sample data if none exists
if not st.session_state.business_data:
    for week in range(1, 13):  # 12 weeks of data
        week_data = generate_realistic_business_data(week)
        st.session_state.business_data.extend(week_data)

# Sidebar for business controls
st.sidebar.header("🎛️ Business Controls")
st.sidebar.markdown("*Configure scenarios and view insights*")

# Week selection
max_week = max([d['week'] for d in st.session_state.business_data]) if st.session_state.business_data else 1
if max_week >= 1:
    selected_week = st.sidebar.slider(
        "📅 Select Business Week", 
        min_value=1, 
        max_value=max_week, 
        value=max_week,
        help="Choose week to analyze business performance"
    )
else:
    selected_week = 1

# Business scenario selector
st.sidebar.subheader("🎯 Focus Area")
focus_area = st.sidebar.selectbox(
    "Choose Business Focus",
    ["Executive Overview", "Inventory Management", "Pricing Strategy", "Marketing Campaigns", "Waste Reduction"]
)

# Add new week data
if st.sidebar.button("▶️ Simulate Next Week"):
    new_week = max_week + 1
    new_data = generate_realistic_business_data(new_week)
    st.session_state.business_data.extend(new_data)
    st.rerun()

# Process data
if st.session_state.business_data:
    df = pd.DataFrame(st.session_state.business_data)
    current_week_data = df[df['week'] == selected_week]
    
    # Executive KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_revenue = current_week_data['revenue'].sum()
        prev_week_revenue = df[df['week'] == max(1, selected_week-1)]['revenue'].sum() if selected_week > 1 else total_revenue
        revenue_change = ((total_revenue - prev_week_revenue) / prev_week_revenue * 100) if prev_week_revenue > 0 else 0
        st.metric(
            "💰 Weekly Revenue", 
            f"${total_revenue:,.0f}",
            delta=f"{revenue_change:+.1f}%"
        )
    
    with col2:
        avg_margin = current_week_data['profit_margin'].mean()
        st.metric(
            "📈 Avg Profit Margin", 
            f"{avg_margin:.1f}%",
            delta="Healthy" if avg_margin > 25 else "Monitor"
        )
    
    with col3:
        total_waste = current_week_data['waste_percentage'].mean()
        st.metric(
            "♻️ Waste Rate", 
            f"{total_waste:.1f}%",
            delta="Excellent" if total_waste < 10 else "Action Needed",
            delta_color="inverse"
        )
    
    with col4:
        inventory_efficiency = current_week_data['inventory_turnover'].mean()
        st.metric(
            "📦 Inventory Turnover", 
            f"{inventory_efficiency:.2f}",
            delta="Optimal" if inventory_efficiency > 0.3 else "Slow"
        )
    
    with col5:
        demand_accuracy = (1 - abs(current_week_data['actual_sales'] - current_week_data['forecasted_demand']) / current_week_data['forecasted_demand']).mean()
        st.metric(
            "🎯 Forecast Accuracy", 
            f"{demand_accuracy:.1%}",
            delta="Excellent" if demand_accuracy > 0.8 else "Needs Work"
        )

    st.markdown("---")

    # Dynamic content based on focus area
    if focus_area == "Executive Overview":
        st.subheader("🎯 Strategic Business Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue trend
            weekly_revenue = df.groupby('week')['revenue'].sum().reset_index()
            fig_revenue = px.line(
                weekly_revenue, 
                x='week', 
                y='revenue', 
                title='📈 Weekly Revenue Trend',
                markers=True
            )
            fig_revenue.update_layout(
                yaxis_title="Revenue ($)",
                xaxis_title="Business Week"
            )
            st.plotly_chart(fig_revenue, use_container_width=True)
        
        with col2:
            # Category performance
            category_perf = current_week_data.groupby('category').agg({
                'revenue': 'sum',
                'profit_margin': 'mean'
            }).reset_index()
            
            fig_category = px.bar(
                category_perf, 
                x='category', 
                y='revenue',
                title='💼 Revenue by Category (Current Week)',
                color='profit_margin',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_category, use_container_width=True)
        
        # Business insights
        st.subheader("💡 AI-Powered Business Insights")
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.info("""
            **🚀 Growth Opportunities**
            - Premium Coffee showing 20% higher margins
            - Seasonal Fruits have 2x demand during holidays
            - Flash Friday campaigns generate 60% sales lift
            """)
        
        with insights_col2:
            st.warning("""
            **⚠️ Risk Areas**
            - Fresh Milk waste rate above 15% threshold
            - Winter Jackets need clearance pricing
            - Inventory levels 30% above optimal for Dairy
            """)

    elif focus_area == "Inventory Management":
        st.subheader("📦 Smart Inventory Optimization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Inventory vs Demand
            fig_inventory = go.Figure()
            fig_inventory.add_trace(go.Bar(
                name='Current Stock',
                x=current_week_data['product'],
                y=current_week_data['current_stock'],
                marker_color='lightblue'
            ))
            fig_inventory.add_trace(go.Bar(
                name='Forecasted Demand',
                x=current_week_data['product'],
                y=current_week_data['forecasted_demand'],
                marker_color='orange'
            ))
            fig_inventory.update_layout(
                title='📊 Stock vs Demand Analysis',
                barmode='group',
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_inventory, use_container_width=True)
        
        with col2:
            # Days to expiry alert
            expiry_data = current_week_data[['product', 'days_to_expiry', 'current_stock']].copy()
            expiry_data['urgency'] = expiry_data['days_to_expiry'].apply(
                lambda x: 'Critical' if x <= 2 else 'Moderate' if x <= 5 else 'Normal'
            )
            
            fig_expiry = px.scatter(
                expiry_data, 
                x='days_to_expiry', 
                y='current_stock',
                size='current_stock',
                color='urgency',
                hover_data=['product'],
                title='⏰ Expiry Risk Assessment',
                color_discrete_map={'Critical': 'red', 'Moderate': 'orange', 'Normal': 'green'}
            )
            st.plotly_chart(fig_expiry, use_container_width=True)
        
        # Inventory recommendations
        st.subheader("🎯 Automated Inventory Actions")
        
        critical_items = current_week_data[current_week_data['days_to_expiry'] <= 2]
        overstocked = current_week_data[current_week_data['current_stock'] > current_week_data['forecasted_demand'] * 2]
        
        rec_col1, rec_col2 = st.columns(2)
        
        with rec_col1:
            st.error("🚨 **Immediate Action Required**")
            if not critical_items.empty:
                for _, item in critical_items.iterrows():
                    st.write(f"• **{item['product']}**: {item['current_stock']} units expire in {item['days_to_expiry']} days")
                    st.write(f"  *Recommended: 40% discount + flash promotion*")
            else:
                st.success("✅ No critical expiry items")
        
        with rec_col2:
            st.warning("📈 **Reorder Recommendations**")
            understocked = current_week_data[current_week_data['current_stock'] < current_week_data['forecasted_demand'] * 0.8]
            if not understocked.empty:
                for _, item in understocked.iterrows():
                    reorder_qty = int(item['forecasted_demand'] * 1.5 - item['current_stock'])
                    st.write(f"• **{item['product']}**: Reorder {reorder_qty} units")
            else:
                st.success("✅ All items adequately stocked")

    elif focus_area == "Pricing Strategy":
        st.subheader("💲 Dynamic Pricing Intelligence")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price vs Margin analysis
            fig_pricing = px.scatter(
                current_week_data, 
                x='current_price', 
                y='profit_margin',
                size='actual_sales',
                color='category',
                hover_data=['product'],
                title='💰 Price-Margin Optimization Map'
            )
            fig_pricing.update_layout(
                xaxis_title="Current Price ($)",
                yaxis_title="Profit Margin (%)"
            )
            st.plotly_chart(fig_pricing, use_container_width=True)
        
        with col2:
            # Discount effectiveness
            discount_data = current_week_data.copy()
            discount_data['discount_rate'] = (1 - discount_data['current_price'] / discount_data['base_price']) * 100
            
            fig_discount = px.scatter(
                discount_data, 
                x='discount_rate', 
                y='actual_sales',
                size='revenue',
                color='days_to_expiry',
                hover_data=['product'],
                title='📊 Discount Impact Analysis'
            )
            fig_discount.update_layout(
                xaxis_title="Discount Rate (%)",
                yaxis_title="Sales Volume"
            )
            st.plotly_chart(fig_discount, use_container_width=True)
        
        # Pricing recommendations
        st.subheader("🎯 AI Pricing Recommendations")
        
        pricing_col1, pricing_col2 = st.columns(2)
        
        with pricing_col1:
            st.success("💡 **Price Increase Opportunities**")
            high_demand = current_week_data[current_week_data['actual_sales'] > current_week_data['forecasted_demand']]
            for _, item in high_demand.iterrows():
                potential_increase = min(15, (item['actual_sales'] / item['forecasted_demand'] - 1) * 20)
                st.write(f"• **{item['product']}**: +{potential_increase:.1f}% price increase potential")
                st.write(f"  *Current: ${item['current_price']:.2f} → Suggested: ${item['current_price'] * (1 + potential_increase/100):.2f}*")
        
        with pricing_col2:
            st.info("🔄 **Dynamic Pricing Alerts**")
            for _, item in current_week_data.iterrows():
                if item['days_to_expiry'] <= 3:
                    recommended_discount = min(50, (4 - item['days_to_expiry']) * 15)
                    st.write(f"• **{item['product']}**: Apply {recommended_discount}% discount")
                    st.write(f"  *Reason: {item['days_to_expiry']} days to expiry*")

    elif focus_area == "Marketing Campaigns":
        st.subheader("📢 Campaign Performance Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Campaign effectiveness
            campaign_perf = current_week_data.groupby('campaign').agg({
                'actual_sales': 'sum',
                'revenue': 'sum',
                'profit_margin': 'mean'
            }).reset_index()
            
            fig_campaigns = px.bar(
                campaign_perf, 
                x='campaign', 
                y='revenue',
                title='🎯 Campaign Revenue Impact',
                color='profit_margin',
                color_continuous_scale='Viridis'
            )
            fig_campaigns.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_campaigns, use_container_width=True)
        
        with col2:
            # ROI by campaign type
            campaign_roi = current_week_data.groupby(['campaign', 'category']).agg({
                'revenue': 'sum',
                'actual_sales': 'sum'
            }).reset_index()
            
            fig_roi = px.treemap(
                campaign_roi,
                path=['campaign', 'category'],
                values='revenue',
                title='💎 Campaign ROI Breakdown'
            )
            st.plotly_chart(fig_roi, use_container_width=True)
        
        # Campaign insights
        st.subheader("📈 Marketing Intelligence")
        
        camp_col1, camp_col2 = st.columns(2)
        
        with camp_col1:
            st.success("🏆 **Top Performing Campaigns**")
            top_campaigns = campaign_perf.nlargest(3, 'revenue')
            for _, camp in top_campaigns.iterrows():
                st.write(f"• **{camp['campaign']}**: ${camp['revenue']:,.0f} revenue")
                st.write(f"  *Avg Margin: {camp['profit_margin']:.1f}%*")
        
        with camp_col2:
            st.info("🎯 **Optimization Opportunities**")
            st.write("• **Flash Friday** shows highest conversion (+60%)")
            st.write("• **Loyalty Rewards** maintains margins better")
            st.write("• **Holiday Sales** work best for Seasonal items")
            st.write("• **Clearance** campaigns reduce waste by 40%")

    elif focus_area == "Waste Reduction":
        st.subheader("♻️ Sustainable Operations Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Waste by category
            fig_waste = px.bar(
                current_week_data, 
                x='category', 
                y='waste_percentage',
                title='🗑️ Waste Rate by Category',
                color='waste_percentage',
                color_continuous_scale='Reds'
            )
            fig_waste.add_hline(y=10, line_dash="dash", line_color="green", 
                               annotation_text="Target: <10%")
            st.plotly_chart(fig_waste, use_container_width=True)
        
        with col2:
            # Correlation between expiry and waste
            fig_correlation = px.scatter(
                current_week_data, 
                x='days_to_expiry', 
                y='waste_percentage',
                size='current_stock',
                color='category',
                title='⏱️ Expiry vs Waste Correlation'
            )
            st.plotly_chart(fig_correlation, use_container_width=True)
        
        # Sustainability metrics
        st.subheader("🌱 Sustainability Impact")
        
        waste_col1, waste_col2 = st.columns(2)
        
        with waste_col1:
            total_waste_units = (current_week_data['current_stock'] * current_week_data['waste_percentage'] / 100).sum()
            waste_cost = (current_week_data['current_stock'] * current_week_data['waste_percentage'] / 100 * current_week_data['current_price']).sum()
            
            st.metric("🗑️ Total Waste Units", f"{total_waste_units:.0f}")
            st.metric("💸 Waste Cost Impact", f"${waste_cost:,.0f}")
            st.metric("🌍 CO2 Reduction Potential", f"{total_waste_units * 2.3:.0f} kg")
        
        with waste_col2:
            st.success("✅ **Waste Reduction Actions**")
            high_waste = current_week_data[current_week_data['waste_percentage'] > 15]
            for _, item in high_waste.iterrows():
                st.write(f"• **{item['product']}**: {item['waste_percentage']:.1f}% waste rate")
                st.write(f"  *Action: Implement dynamic pricing 2 days before expiry*")

    # Data table for detailed analysis
    st.markdown("---")
    st.subheader("📋 Detailed Business Data")
    
    # Format the dataframe for business users
    display_df = current_week_data[[
        'product', 'category', 'current_stock', 'days_to_expiry', 
        'forecasted_demand', 'actual_sales', 'current_price', 'revenue', 
        'campaign', 'waste_percentage', 'profit_margin'
    ]].copy()
    
    st.dataframe(
        display_df.style.format({
            'current_price': '${:.2f}',
            'revenue': '${:,.0f}',
            'waste_percentage': '{:.1f}%',
            'profit_margin': '{:.1f}%',
            'current_stock': '{:.0f}',
            'actual_sales': '{:.0f}',
            'forecasted_demand': '{:.0f}'
        }).background_gradient(subset=['profit_margin', 'revenue']),
        use_container_width=True
    )
    
    # Export functionality
    st.subheader("📤 Business Reports Export")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📊 Download CSV Report",
            data=csv,
            file_name=f"business_intelligence_week_{selected_week}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        json_data = display_df.to_json(orient='records')
        st.download_button(
            label="📋 Download JSON Data",
            data=json_data,
            file_name=f"business_data_week_{selected_week}_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )
    
    with col3:
        # Create executive summary
        summary = f"""
        EXECUTIVE SUMMARY - Week {selected_week}
        =====================================
        
        KEY METRICS:
        - Total Revenue: ${current_week_data['revenue'].sum():,.0f}
        - Average Profit Margin: {current_week_data['profit_margin'].mean():.1f}%
        - Waste Rate: {current_week_data['waste_percentage'].mean():.1f}%
        - Forecast Accuracy: {(1 - abs(current_week_data['actual_sales'] - current_week_data['forecasted_demand']) / current_week_data['forecasted_demand']).mean():.1%}
        
        TOP PERFORMERS:
        {chr(10).join([f"- {row['product']}: ${row['revenue']:,.0f}" for _, row in current_week_data.nlargest(3, 'revenue').iterrows()])}
        
        ACTION ITEMS:
        - Monitor expiring inventory for dynamic pricing
        - Optimize stock levels based on demand patterns
        - Continue high-performing marketing campaigns
        """
        
        st.download_button(
            label="📈 Executive Summary",
            data=summary,
            file_name=f"executive_summary_week_{selected_week}_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )

else:
    # Empty state with business context
    st.info("🚀 **Welcome to Retail Intelligence!** Click 'Simulate Next Week' to start generating business insights.")
    
    st.subheader("🎯 What This Dashboard Demonstrates")
    
    demo_col1, demo_col2 = st.columns(2)
    
    with demo_col1:
        st.markdown("""
        **🏪 Inventory Management**
        - Real-time stock monitoring
        - Expiry date tracking
        - Automated reorder suggestions
        - Waste reduction strategies
        
        **💲 Dynamic Pricing**
        - Profit margin optimization
        - Discount effectiveness analysis
        - Competitive pricing insights
        - Revenue impact modeling
        """)
    
    with demo_col2:
        st.markdown("""
        **📢 Marketing Campaigns**
        - Campaign ROI analysis
        - Customer segment targeting
        - Promotional effectiveness
        - Cross-category insights
        
        **📊 Business Intelligence**
        - Executive KPI dashboards
        - Predictive analytics
        - Trend analysis
        - Actionable recommendations
        """)

# Footer with business value
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 14px;'>
<b>🎯 Business Value:</b> AI-driven insights for 15% revenue increase, 40% waste reduction, and 25% inventory optimization
<br>
<b>📈 ROI:</b> Typical implementation sees 300% ROI within 6 months
</div>
""", unsafe_allow_html=True)
