"""
dashboard.py

This script generate an interpretability dashboard for explaining why a customer
was granted the loan he/she applied for.

params:
    thres:
        Threshold risk value above which a customer's loan is denied.
    n_sample : 
        Number of customers to include in the dashboard.
        If None, all customers are included.

returns:
    a web application displaying the interpretability dashboard
"""

import pandas as pd
import numpy as np
import tracemalloc
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_functions

from dash.dependencies import Input, Output, State


# Dashboard parameters
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(external_stylesheets=external_stylesheets)
server = app.server
thres = 0.3
n_sample=500


# Load data
df_crit=dash_functions.load_criteria_descriptions()
df_cust=dash_functions.load_customer_data(n_sample=n_sample)


customer_list = df_cust.index.map(
    lambda x : {'label': str(x), 'value':x}).tolist()

logo = 'https://user.oc-static.com/upload/2019/02/25/15510866018677_'+\
    'logo%20projet%20fintech.png'  

# Define some styles
title_style={'font-weight':'bold', 'text-align':'center', 
             'background-color':'darkblue', 'color':'white'}
H2_style = {'background-color':'lightblue', 'font-weight':'bold'}

# Dashboard layout
app.layout = html.Div(children=[

    # Dash header
    html.Div(
        className='row',
        children=[

        # "Pret à depenser" logo
        html.Img(
            src=logo, width=218, height=200,
            className="two columns"
        ),

        html.Div(
            className='ten columns',
            children=[
                # Dash title
                html.H1(
                    className='row',
                    style=title_style,
                    children='Decision-making dashboard'),

                html.H4(
                    className='row',
                    children='Select a customer ID to retrieve decision' +\
                            'and to explain why the loan is granted or denied'),

                # Customer selection and loan decision
                html.Div(
                    className='row',
                    children=[
                        html.Div(
                            className='three columns',
                            style={'fontSize':20},
                            children=[
                                html.Div(children='Customer ID :'),
                                dcc.Dropdown(
                                    id='customer_selection',
                                    options=customer_list
                                )]),

                        html.Div(
                            className='three columns',
                            style={'fontSize':20},
                            children=[
                                html.Div('Estimated risk', 
                                         style={'text-align': 'center'}),
                                html.Div(id='customer_risk', 
                                         style={'text-align': 'center'})]),

                        html.Div(
                            className='three columns',
                            style={'fontSize':20},
                            children=[
                                html.Div('Loan is', 
                                         style={'text-align': 'center'}),
                                html.Div(id='customer_decision', 
                                         style={'text-align': 'center'})]),

                        html.Div(
                            className='three columns',
                            style={'fontSize':20},
                            children=[
                                html.Div('Maximum allowed risk is', 
                                         style={'text-align': 'center'}),
                                html.Div(children='{:.0%}'.format(thres),
                                        style={'text-align': 'center'})]) 
                    ])])]
    ),

    # Storing shap values for selected customer
    dcc.Store(id='shapleys'),
    
    # Customer position vs customers panel
    html.H2(children='Customer position in customer panel',style=H2_style),
    html.Button('Update panel', id='maj_panel', n_clicks=0),
    dcc.Graph(id='panel'),

    # Top criteria section
    html.H2(children='Most important criteria', style =H2_style),
    html.Div(
        className='row',
        children = [
            html.Button('Load explanations', id='maj_explain', 
                        n_clicks=0, className='two columns'),
            html.Button('Update waterfall', id='maj_water', 
                        n_clicks=0, className='two columns'),
            html.Button('Update top tables', id='maj_tables', 
                        n_clicks=0, className='two columns') 
        ]
    ),
    
    # Top criteria KPI
    html.Div(children=[
        html.Div(children=[
            html.H3(children='Waterfall for selected customer'),

            # Slider for number of criteria to display
            html.Label('Number of criteria to display'),
            dcc.Slider(id='top_slider', 
                   min=5, max=50, value=15, step=5,
                   marks={
                       x: 'Top {}'.format(x) if x==5 else str(x) for x in range(5,55,5)
                   }),

            # Waterfall plot
            dcc.Graph(id='waterfall')
            ], className='five columns'),

        # Top criteria for selected customer
        html.Div(children=[
            html.Div(id='top_tables')],
                 className='seven columns')
     ], className='row'),
    

    # Criteria section
    html.Div(
        className='row',
        children=[
         # Criteria selection and description
        html.H2(children='Criteria description', style=H2_style),
        html.Button('Update criteria description', id='maj_crit', n_clicks=0),
        html.Div(
            children=[
                html.Div(
                    className='four columns',
                    children=[
                        html.H3(children='Select a criteria'),
                        dcc.Dropdown(
                            id='crit_selection',
                            options=df_crit['options'].tolist()),

                        html.H3(children='Description :'), 
                        html.Div(id='crit_descr'),

                        html.H3(children='Customer value :'), 
                        html.Div(id='cust_crit_value'),

                        html.H3(children='Impact on customer :'), 
                        html.Div(id='cust_crit_impact'),
            ]),

        # Shap vs value scatter plot
        html.Div(
            className='eight columns',
            children=[
                html.H3(id='scatter_title'),
                dcc.Graph(id='scatter_plot')
            ])
    ])])
])

# Callbacks and component updates

# Callback when new customer is selected
@app.callback(
    [Output('customer_risk', 'children'),
     Output('customer_decision', 'children')],
    Input('customer_selection', 'value'),
    prevent_initial_call=True
)

def update_customer(customer_id):
    """
    Update decision, position in panel, waterfall and top 15 criteria
    when a customer is selected in dropdown.
    """
    
    tracemalloc.start()
    models = dash_functions.load_models()
    
    # Update customer estimated risk and decision
    risk, decision = dash_functions.predict_decision(
        models, df_cust, customer_id, thres)
    del models

    risk_output='{:.1%}'.format(risk)
    del risk
    
    decision_output = 'granted' if decision else 'denied'
    del decision
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Decision update - Peak memory usage was {peak / 10**6}MB")

    return risk_output, decision_output
    del risk_output, decision_output, current, peak

    
# Callback for loading explainations
@app.callback(
    Output('shapleys', 'data'),
    Input('maj_explain', 'n_clicks'),
    State('customer_selection', 'value'),
    prevent_initial_call=True
)

def update_explaination(n_clicks, customer_id):
    """
    """
    shaps, base_value =  dash_functions.shap_explain(customer_id, df_cust)
    dic={'shaps':shaps, 'base':base_value}
    
    return dic
    del shaps, base_value, dic
    
    
    
# Callback for updating panel
@app.callback(
    Output('panel', 'figure'),
    Input('maj_panel', 'n_clicks'),
    State('customer_selection', 'value'),
    State('customer_risk', 'children'),
    prevent_initial_call=True
)   
    
def update_panel(n_clicks, customer_id, customer_risk):
    """
    Display customer panel as an histogram and show position of selected customer.
    """
    # Show customer position on customer panel
    tracemalloc.start()
    
    panel_hist = dash_functions.load_panel()
    fig_panel = dash_functions.plot_panel(panel_hist, thres)
    
    cust_bin = np.floor(float(customer_risk[:-1]))/100
    i_bin = panel_hist[1].tolist().index(cust_bin)
    cust_height = panel_hist[0][i_bin]

    fig_panel.add_shape(
        type='rect',
        x0=cust_bin-0.005, 
        x1=cust_bin+0.005, 
        y0=0, y1=cust_height, 
        fillcolor='yellow')
    del cust_bin, cust_height, i_bin
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Panel update - Peak memory usage was {peak / 10**6}MB")
    
    return fig_panel
    del panel_hist, fig_panel, current, peak
    
    
# Callback for updating waterfall
@app.callback(
    Output('waterfall', 'figure'),
    Input('maj_water', 'n_clicks'),
    [State('shapleys', 'data'),
     State('top_slider', 'value'),
     State('customer_selection', 'value')],
    prevent_initial_call=True
)

def update_water(n_clicks, shapleys, n_top, customer_id):
    """
    Generate waterfall and top tables of major criteria for a give customer.
    """
    # Update waterfall
    tracemalloc.start()
    
    shaps = np.array(shapleys['shaps'])
    base_value = shapleys['base']
    
    fig_waterfall = dash_functions.plot_waterfall(
        df_cust, customer_id, n_top, thres, base_value, shaps)
    del shaps, base_value
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Waterfall update - Peak memory usage was {peak / 10**6}MB")

    return fig_waterfall
    del fig_waterfall, current, peak, shaps, base_value
    
    
# Callback for updating top tables
@app.callback(
    Output('top_tables', 'children'),
    Input('maj_tables', 'n_clicks'),
    [State('shapleys', 'data'),
     State('top_slider', 'value'),
     State('customer_selection', 'value')],
    prevent_initial_call=True
)    

def update_tables(n_click, shapleys, n_top, customer_id):
    """
    """
    # Update top n_top tables
    tracemalloc.start()
    shaps = np.array(shapleys['shaps'])
    
    children_top = dash_functions.generate_top_tables(
        n_top, df_cust, customer_id, shaps)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Top tables update - Peak memory usage was {peak / 10**6}MB")
    
    return children_top
    del children_top, shaps, base_value, current, peak

    
# Callbacks with a new criteria selected
@app.callback(
    [Output('crit_descr', 'children'),
     Output('scatter_title', 'children'),
     Output('scatter_plot', 'figure'),
     Output('cust_crit_value', 'children'),
     Output('cust_crit_impact', 'children')],
    Input('maj_crit', 'n_clicks'),
    [State('shapleys', 'data'),
     State('crit_selection', 'value'),
     State('customer_selection', 'value')],
    prevent_initial_call=True
)

def update_description(n_click, shapleys, crit, cust):
    """
    Plot scatter plot for evolution of impact with change in criteria value.
    """
    tracemalloc.start()
    
    if crit is not None:
        output=df_crit[df_crit['Row']==crit]['Description'].values[0]
        title=f'Evolution of impact with {crit} value :'
                
        shaps=np.array(shapleys['shaps'])
        df_shap=dash_functions.load_shap_values()
        
        fig=dash_functions.plot_shap_scatter(
            df_cust, df_shap, crit, cust, shaps, thres)

        if cust is not None:
            cust_crit_val=df_cust.loc[cust, crit]

            df_shaps=pd.DataFrame(shaps[0].T, index = df_cust.columns)

            cust_crit_imp=df_shaps.loc[crit, 0]
            cust_crit_imp='{:.4f}'.format(cust_crit_imp)
            
        else :
            cust_crit_val='NA'
            cust_crit_imp='NA'

        del df_shap
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"Criteria update - Peak memory usage was {peak / 10**6}MB")
        
        return output, title, fig, cust_crit_val, cust_crit_imp


# Run the dashboard   
if __name__=="__main__":
    app.run_server(debug=False)
    #app.enable_dev_tools(dev_tools_ui=True)
    #app.app.enable_dev_tools(dev_tools_ui=True, use_reloader=False)
    
