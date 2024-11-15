import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dash import dash_table
from datetime import datetime
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
    # Group by 'household' and calculate the desired metrics
    aggregated_data = data.groupby('household').agg(
        Recency_min=('Recency', 'min'),
        Frequency_avg=('Frequency', 'mean'),
        MonetaryValue_avg=('MonetaryValue', 'mean'),
        RFM_Customer_Segments=('RFM Customer Segments', lambda x: x.mode()[0] if not x.mode().empty else 'N/A')
    ).reset_index()

    # Rename columns if needed to match your desired output format
    aggregated_data.rename(columns={'household': 'household.unique'}, inplace=True)

    # Standardize the data
    rfm_scaled = aggregated_data[['Recency_min', 'Frequency_avg', 'MonetaryValue_avg']]

    # Use the elbow method to find the optimal number of clusters
    inertias = []
    k_values = range(1, 11)  # Test k from 1 to 10

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(rfm_scaled)
        inertias.append(kmeans.inertia_)

    # Create an elbow plot using Plotly
    kmeanfig = go.Figure()
    kmeanfig.add_trace(go.Scatter(
        x=list(k_values), 
        y=inertias, 
        mode='lines+markers',
        marker=dict(color='blue'),
        line=dict(dash='solid')
    ))
    
    kmeanfig.update_layout(
        title='Elbow Method for Optimal K',
        xaxis_title='Number of clusters (k)',
        yaxis_title='Inertia',
        xaxis=dict(tickmode='linear'),
        template='plotly_white'
    )
    #fig.show()

    # Choose the optimal k (manually based on the elbow plot)
    optimal_k = 4  # Replace with your chosen k after viewing the plot

    # Apply KMeans clustering with the chosen k
    kmeans = KMeans(n_clusters=optimal_k, random_state=0)
    kmeans.fit(rfm_scaled)
    aggregated_data['RiskCluster'] = kmeans.labels_

    # Map clusters to promotion levels based on potential churn risk
    risk_to_promotion = {
        3: 'lv0 - No Promotion',
        2: 'lv1 - Soft Promotion (1%, 3%, 5%)',
        0: 'lv2 - Hard Promotion (10%, 15%, 20%)',
        1: 'lv3 - Extreme Promotion (Buy 1 Free 1)'
    }
    aggregated_data['Promotion Level'] = aggregated_data['RiskCluster'].map(risk_to_promotion)

    aggregated_data.to_csv("predict_promotion.csv", index=False)

    # Plot the promotion level distribution
    fig = px.histogram(
        aggregated_data,
        x='Promotion Level',
        title='Promotion Level Distribution Among Customers',
        color='Promotion Level',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    fig.update_layout(xaxis_title="Promotion Level", yaxis_title="Number of Customers", showlegend=False)

    ccfig = px.scatter_3d(aggregated_data, x='Recency_min', y='Frequency_avg', z='MonetaryValue_avg', color='RiskCluster',
                     title='Customer Segmentation using K-means Clustering')
    
    return aggregated_data, kmeanfig, ccfig, fig

def create_promotion_table(data):
    """
    This function generates a table showing the count of customers in each segment
    that falls under each promotion level.
    
    Parameters:
    - data (pd.DataFrame): The DataFrame containing customer segments and promotion levels.
    
    Returns:
    - promotion_table (pd.DataFrame): A pivot table with customer counts per segment and promotion level.
    """
    # Create a pivot table to count customers in each segment with each promotion level
    promotion_table = data.pivot_table(
        index='RFM_Customer_Segments',
        columns='Promotion Level',
        aggfunc='size',
        fill_value=0
    ).reset_index()
    
    return promotion_table

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.io as pio
import plotly.colors as pc

box = {'padding': '10px', 'font-size' : '26px', 'margin-left' : '10px', 'margin-right' : '10px'}
smallbox = {'padding': '10px'}


# Initialize the Dash app
app = dash.Dash(__name__)
loadfig = determine_promotional_discount(data)
promotion_table = create_promotion_table(loadfig[0])

# Define the app layout using Bootstrap components
app.layout = html.Div(style={'text-align' : 'center','align-items': 'center'},children=[
    html.H1("Machine learning", className="text-center mb-4"),
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

    html.Div([
        dcc.Graph(figure=loadfig[1])
    ], style={'width': '40%','display': 'inline-block'}),
    html.Div([
        dcc.Graph(figure=loadfig[2])
    ], style={'width': '40%','display': 'inline-block'}),
    html.Div([
        dcc.Graph(figure=loadfig[3])
    ], style={'width': '40%','display': 'inline-block'}),
    html.Div([
        html.H1("Customer Segment Promotion Table"),
            dash_table.DataTable(
                data=promotion_table.to_dict('records'),
                columns=[{"name": i, "id": i} for i in promotion_table.columns],
                style_table={'width': '100%'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                style_cell={'textAlign': 'center', 'font-size' : '20px'}
    )
])
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