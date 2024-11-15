import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Set the default template for Plotly
pio.templates.default = "plotly_white"

# Load the dataset
data = pd.read_csv("D11_last.csv")
print(data.head())

# Convert 'PurchaseDate' to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Calculate Recency
data['Recency'] = (datetime.now() - data['Date']).dt.days

# Calculate Frequency
frequency_data = data.groupby('household')['basket'].count().reset_index()
frequency_data.rename(columns={'basket': 'Frequency'}, inplace=True)
data = data.merge(frequency_data, on='household', how='left')

# Calculate Monetary Value
monetary_data = data.groupby('household')['dollar_sales'].sum().reset_index()
monetary_data.rename(columns={'dollar_sales': 'MonetaryValue'}, inplace=True)
data = data.merge(monetary_data, on='household', how='left')

# Define scoring criteria for each RFM value
recency_scores = [5, 4, 3, 2, 1]  # Higher score for lower recency (more recent)
frequency_scores = [1, 2, 3, 4, 5]  # Higher score for higher frequency
monetary_scores = [1, 2, 3, 4, 5]  # Higher score for higher monetary value

# Calculate RFM scores
data['RecencyScore'] = pd.cut(data['Recency'], bins=5, labels=recency_scores)
data['FrequencyScore'] = pd.cut(data['Frequency'], bins=5, labels=frequency_scores)
data['MonetaryScore'] = pd.cut(data['MonetaryValue'], bins=5, labels=monetary_scores)

# Convert RFM scores to numeric type
data['RecencyScore'] = data['RecencyScore'].astype(int)
data['FrequencyScore'] = data['FrequencyScore'].astype(int)
data['MonetaryScore'] = data['MonetaryScore'].astype(int)

# Calculate RFM score by combining the individual scores
data['RFM_Score'] = data['RecencyScore'] + data['FrequencyScore'] + data['MonetaryScore']

# Create RFM segments based on the RFM score
segment_labels = ['Low-Value', 'Mid-Value', 'High-Value']
data['Value Segment'] = pd.qcut(data['RFM_Score'], q=3, labels=segment_labels)

# RFM Segment Distribution
segment_counts = data['Value Segment'].value_counts().reset_index()
segment_counts.columns = ['Value Segment', 'Count']

# Define the pastel color palette
pastel_colors = px.colors.qualitative.Pastel

# Create the bar chart with pastel colors
fig_segment_dist = px.bar(segment_counts, x='Value Segment', y='Count',
                          color='Value Segment', color_discrete_sequence=pastel_colors,
                          title='RFM Value Segment Distribution')

# Update the layout
fig_segment_dist.update_layout(xaxis_title='RFM Value Segment',
                               yaxis_title='Count',
                               showlegend=False)

# Show the figure
# fig_segment_dist.show()

# Create a new column for RFM Customer Segments
data['RFM Customer Segments'] = ''

# Assign RFM segments based on the RFM score
data.loc[data['RFM_Score'] >= 9, 'RFM Customer Segments'] = 'Champions'
data.loc[(data['RFM_Score'] >= 6) & (data['RFM_Score'] < 9), 'RFM Customer Segments'] = 'Potential Loyalists'
data.loc[(data['RFM_Score'] >= 5) & (data['RFM_Score'] < 6), 'RFM Customer Segments'] = 'At Risk Customers'
data.loc[(data['RFM_Score'] >= 4) & (data['RFM_Score'] < 5), 'RFM Customer Segments'] = "Can't Lose"
data.loc[(data['RFM_Score'] >= 3) & (data['RFM_Score'] < 4), 'RFM Customer Segments'] = "Lost"

# Count of customers in each segment and value segment
segment_product_counts = data.groupby(['Value Segment', 'RFM Customer Segments']).size().reset_index(name='Count')
segment_product_counts = segment_product_counts.sort_values('Count', ascending=False)

# Create a treemap for RFM Customer Segments by Value
fig_treemap_segment_product = px.treemap(segment_product_counts,
                                         path=['Value Segment', 'RFM Customer Segments'],
                                         values='Count',
                                         color='Value Segment', color_discrete_sequence=pastel_colors,
                                         title='RFM Customer Segments by Value')
# fig_treemap_segment_product.show()

# Filter the data to include only the customers in the Champions segment
champions_segment = data[data['RFM Customer Segments'] == 'Champions']

# Create a box plot for RFM values within the Champions segment
champions_segment_fig = go.Figure()
champions_segment_fig.add_trace(go.Box(y=champions_segment['RecencyScore'], name='Recency'))
champions_segment_fig.add_trace(go.Box(y=champions_segment['FrequencyScore'], name='Frequency'))
champions_segment_fig.add_trace(go.Box(y=champions_segment['MonetaryScore'], name='Monetary'))

champions_segment_fig.update_layout(title='Distribution of RFM Values within Champions Segment',
                                    yaxis_title='RFM Value',
                                    showlegend=True)

# champions_segment_fig.show()

# Calculate the correlation matrix for RFM values within the Champions segment
correlation_matrix = champions_segment[['RecencyScore', 'FrequencyScore', 'MonetaryScore']].corr()

# Visualize the correlation matrix using a heatmap
fig_corr_heatmap = go.Figure(data=go.Heatmap(
                   z=correlation_matrix.values,
                   x=correlation_matrix.columns,
                   y=correlation_matrix.columns,
                   colorscale='RdBu',
                   colorbar=dict(title='Correlation')))

fig_corr_heatmap.update_layout(title='Correlation Matrix of RFM Values within Champions Segment')

# fig_corr_heatmap.show()

# Define the pastel color palette
pastel_colors = px.colors.qualitative.Pastel

# Segment counts for RFM Customer Segments
segment_counts = data['RFM Customer Segments'].value_counts()

# Create a bar chart to compare segment counts
comparison_fig = go.Figure(data=[go.Bar(x=segment_counts.index, y=segment_counts.values,
                                        marker=dict(color=pastel_colors))])

# Set the color of the Champions segment as a different color
champions_color = 'rgb(158, 202, 225)'
comparison_fig.update_traces(marker_color=[champions_color if segment == 'Champions' else pastel_colors[i]
                                           for i, segment in enumerate(segment_counts.index)],
                             marker_line_color='rgb(8, 48, 107)',
                             marker_line_width=1.5, opacity=0.6)

# Update the layout
comparison_fig.update_layout(title='Comparison of RFM Segments',
                             xaxis_title='RFM Segments',
                             yaxis_title='Number of Customers',
                             showlegend=False)

# Show the figure
# comparison_fig.show()

# Calculate the average Recency, Frequency, and Monetary scores for each segment
segment_scores = data.groupby('RFM Customer Segments')[['RecencyScore', 'FrequencyScore', 'MonetaryScore']].mean().reset_index()

# Create a grouped bar chart to compare segment scores
fig = go.Figure()

# Add bars for Recency score
fig.add_trace(go.Bar(
    x=segment_scores['RFM Customer Segments'],
    y=segment_scores['RecencyScore'],
    name='Recency Score',
    marker_color='rgb(158,202,225)'
))

# Add bars for Frequency score
fig.add_trace(go.Bar(
    x=segment_scores['RFM Customer Segments'],
    y=segment_scores['FrequencyScore'],
    name='Frequency Score',
    marker_color='rgb(94,158,217)'
))

# Add bars for Monetary score
fig.add_trace(go.Bar(
    x=segment_scores['RFM Customer Segments'],
    y=segment_scores['MonetaryScore'],
    name='Monetary Score',
    marker_color='rgb(32,102,148)'
))

# Update the layout
fig.update_layout(
    title='Comparison of RFM Segments based on Recency, Frequency, and Monetary Scores',
    xaxis_title='RFM Segments',
    yaxis_title='Score',
    barmode='group',
    showlegend=True
)

#fig.show()

def determine_promotional_discount(data):
    """
    This function performs clustering on RFM data to segment customers based on risk and assigns
    a promotion level to each segment.
    
    Parameters:
    - data (pd.DataFrame): The input RFM dataset
    
    Returns:
    - data (pd.DataFrame): The dataset with an additional 'Promotion Level' column
    """
    # Select RFM scores for clustering
    rfm_scores = data[['RecencyScore', 'FrequencyScore', 'MonetaryScore']]
    
    # Standardize the RFM scores
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_scores)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=0)  # Adjust number of clusters as needed
    data['RiskCluster'] = kmeans.fit_predict(rfm_scaled)
    # Map clusters to promotion levels based on potential churn risk
    # Assigning promotion levels to clusters (Adjust based on your specific clustering results)
    risk_to_promotion = {
        0: 'lv0 - No Promotion',  # Lowest churn risk
        1: 'lv1 - Soft Promotion (1%, 3%, 5%)',
        2: 'lv2 - Hard Promotion (10%, 15%, 20%)',
        3: 'lv3 - Extreme Promotion (Buy 1 Free 1)'  # Highest churn risk
    }
    data['Promotion Level'] = data['RiskCluster'].map(risk_to_promotion)
    
    fig = px.histogram(
        data, 
        x='Promotion Level',
        title='Promotion Level Distribution Among Customers',
        color='Promotion Level',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    fig.update_layout(xaxis_title="Promotion Level", yaxis_title="Number of Customers", showlegend=False)

    return fig

def top_10_products_histogram(data, segment):
    """
    This function takes the data and a specific customer segment as input,
    filters the top 10 purchased products in that segment, and returns a Plotly histogram chart.
    
    Parameters:
    - data (pd.DataFrame): The input dataset
    - segment (str): The customer segment to analyze
    
    Returns:
    - fig (plotly.graph_objects.Figure): A histogram chart showing the top 10 products for the given segment
    """
    # Filter data for the specified segment
    segment_data = data[data['RFM Customer Segments'] == segment]
    
    # Group by product and count occurrences (or use sum of sales for top revenue)
    top_products = (
        segment_data.groupby('product_description')['basket'].count()  # or 'dollar_sales' for top revenue
        .reset_index()
        .rename(columns={'basket': 'PurchaseCount'})
        .sort_values(by='PurchaseCount', ascending=False)
        .head(10)
    )
    
    # Create a histogram chart
    fig = px.bar(
        top_products,
        x='product_description',
        y='PurchaseCount',
        title=f"Top 10 Products for {segment}",
        labels={'product_description': 'Product', 'PurchaseCount': 'Number of Purchases'},
        color_discrete_sequence=['#789DBC']
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Product",
        yaxis_title="Number of Purchases",
        showlegend=False
    )
    
    return fig


def cal_churn_rate():

    data = pd.read_csv("D11_last.csv")
    # First, convert the 'Date' column to datetime to filter by months
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

    # Filter data for September and October
    september_data = data[data['Date'].dt.month == 9]
    october_data = data[data['Date'].dt.month == 10]

    # Get unique households for September and October
    september_households = set(september_data['household'].unique())
    october_households = set(october_data['household'].unique())

    # Calculate the number of churned households (active in September but not in October)
    churned_households = september_households - october_households

    # Calculate churn rates
    september_churn_rate = (len(churned_households) / len(september_households)) * 100 if len(september_households) > 0 else 0
    october_churn_rate = ((len(september_households - october_households)) / len(october_households)) * 100 if len(october_households) > 0 else 0

    # Calculate the difference in churn rate
    churn_rate_difference = october_churn_rate - september_churn_rate
    if churn_rate_difference > 0:
        churn_rate_difference = "+" + str(round(churn_rate_difference, 2))
    return round(september_churn_rate, 2), round(october_churn_rate, 2), churn_rate_difference

def cal_avg_basket():
    data = pd.read_csv("D11_last.csv")
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    october_data = data[data['Date'].dt.month == 10]
    
    basket_count = len(october_data['basket'].unique())
    dollar_sales_sum = october_data['dollar_sales'].sum()
    return round(dollar_sales_sum / basket_count, 2)

def count_household():
    data = pd.read_csv("D11_last.csv")
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    october_data = data[data['Date'].dt.month == 10]
    october_households = len(october_data['household'].unique())
    return october_households

def cal_coupon():
    data = pd.read_csv("D11_last.csv")
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    october_data = data[data['Date'].dt.month == 10]
    used_coupon = october_data['coupon'].sum()
    return used_coupon

def cal_diff_month():
    data = pd.read_csv("D11_last.csv")
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    september_data = data[data['Date'].dt.month == 9]['dollar_sales'].sum()
    october_data = data[data['Date'].dt.month == 10]['dollar_sales'].sum()
    data_difference = september_data - october_data
    if data_difference > 0:
        data_difference = "+" + str(round(data_difference, 2))
    return round(september_data, 2), round(october_data, 2), data_difference

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.io as pio
import plotly.colors as pc

box = {'padding': '10px', 'font-size' : '26px', 'margin-left' : '10px', 'margin-right' : '10px'}
smallbox = {'padding': '10px'}


# Initialize the Dash app
app = dash.Dash(__name__)
churn_rate = cal_churn_rate()
Revenue_month = cal_diff_month()

# Define the app layout using Bootstrap components
app.layout = html.Div(style={'text-align' : 'center','align-items': 'center'},children=[
    html.H1("Custormer Dashboard Summary October 2024"),
    html.Div(style={'display': 'flex', 'flex-wrap' : 'wrap' ,'padding': '10px', 'text-align' : 'center', 'align-items': 'center', 'background' : '#789DBC', 'justify-content': 'center'},
            children=[html.Div(style=smallbox,
                    children=[html.H2("Previous Month"),html.H2(str(churn_rate[0]) + "%")]),
                    html.Div(style=smallbox,
                    children=[html.H2("Change"),html.H2(str(churn_rate[2]) + "%")]),
                    html.Div(style=box,
                    children=[html.H2("Churn rate"),html.H2(str(churn_rate[1]) + "%")]),
                    html.Div(style=box,
                    children=[html.H2("Household"),html.H2(str(count_household()))]),
                    html.Div(style=box,
                    children=[html.H2("Used Coupon"),html.H2(str(cal_coupon()))]),
                    html.Div(style=box,
                    children=[html.H2("AVG. Basket"),html.H2("$" + str(cal_avg_basket()))]),
                    html.Div(style=box,
                    children=[html.H2("Revenue Month"),html.H2(str(Revenue_month[0]))]),
                    html.Div(style=smallbox,
                    children=[html.H2("Change"),html.H2(str(Revenue_month[2]))]),
                    html.Div(style=smallbox,
                    children=[html.H2("Previous Month"),html.H2(str(Revenue_month[1]))]),
                       ]),


    # Graph container
    html.Div(style={'display' : 'flex','justify-content': 'center', 'text-align' : 'center', 'align-items': 'center'},
    children=[html.Div(style={'width': '80%'}, children=[
        html.H1("RFM Analysis", className="text-center mb-4"),
        html.Div("Analyze customer segments based on RFM scores.", className="text-center mb-4"),
    # Dropdown for selecting the chart
    dcc.Dropdown(
        id='chart-type-dropdown',
        options=[
            {'label': 'RFM Value Segment Distribution', 'value': 'segment_distribution'},
            {'label': 'Distribution of RFM Values within Customer Segment', 'value': 'RFM_distribution'},
            {'label': 'Correlation Matrix of RFM Values within Champions Segment', 'value': 'correlation_matrix'},
            {'label': 'Comparison of RFM Segments', 'value': 'segment_comparison'},
            {'label': 'Comparison of RFM Segments based on Scores', 'value': 'segment_scores'},
        ],
        value='segment_distribution',  # Default selection
        className="mb-4",
    ),dcc.Graph(id='rfm-chart', className="mb-4")])]),

    html.Div(style={'display': 'flex', 'flex-wrap' : 'wrap', 'justify-content': 'center'}, children=[

        html.Div([
            dcc.Graph(figure=top_10_products_histogram(data,"Champions"))
        ], style={'width': '30%', 'margin-top': '20px', 'display': 'inline-block'}),
            html.Div([
            dcc.Graph(figure=top_10_products_histogram(data,"Potential Loyalists"))
        ], style={'width': '30%', 'margin-top': '20px', 'display': 'inline-block'}),
            html.Div([
            dcc.Graph(figure=top_10_products_histogram(data,"At Risk Customers"))
        ], style={'width': '30%', 'margin-top': '20px', 'display': 'inline-block'}),
            html.Div([
            dcc.Graph(figure=top_10_products_histogram(data,"Can't Lose"))
        ], style={'width': '30%', 'margin-top': '20px', 'display': 'inline-block'}),
            html.Div([
            dcc.Graph(figure=top_10_products_histogram(data,"Lost"))
        ], style={'width': '30%', 'margin-top': '20px', 'display': 'inline-block'}),
    ]),
])

# Define callback to update the selected chart
@app.callback(
    Output('rfm-chart', 'figure'),
    [Input('chart-type-dropdown', 'value')]
)
def update_chart(selected_chart_type):
    if selected_chart_type == 'segment_distribution':
        return fig_segment_dist
    elif selected_chart_type == 'RFM_distribution':
        return fig_treemap_segment_product
    elif selected_chart_type == 'correlation_matrix':
        return fig_corr_heatmap
    elif selected_chart_type == 'segment_comparison':
        return comparison_fig
    elif selected_chart_type == 'segment_scores':
        return fig

    # Return a default chart if no valid selection
    return fig_segment_dist

if __name__ == '__main__':
    app.run_server(host="0.0.0.0", port=8052, debug=True)