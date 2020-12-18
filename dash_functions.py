"""
dash_functions.py

This script contains functions called by dashboard.py to update the dashboard.
"""

import pandas as pd
import numpy as np
import joblib
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objs as go


source_path = 'data/'

def load_models():
    """
    Loads and returns the list of LGBM classifiers.
    """
    # Load the file containing models
    file = 'LGBM_classifiers.joblib'
    models = joblib.load(source_path+file)
    
    return models


def predict_decision(models, df_cust, customer_id, thres):
    """
    Use models to predict whether loan shall be granted for a given customer.
    
    params:
        models:
            A list of models to use for making prediction.
        df_cust : 
            A customer description DataFrame.
        customer_id :
            The ID of the customer for whom prediction will be made.
        thres :
            Threshold risk value above which a customer's loan is denied.
            
    returns :
        A tuple where the first value is the estimated risk and the second
        value is a boolean stating is loan is granted.
    """
    # Prepare data for prediction
    n_models = len(models)
    cust_array = df_cust.loc[customer_id].values.reshape(1,-1)
    decision_vector = np.array([[0,0]], dtype='float')
    
    # Make prediction with each individual model
    for clf in range(n_models):
        decision_vector += models[clf].predict_proba(cust_array)
        
    # Average probability and make decision
    decision_vector/=n_models
    
    prob_deny = decision_vector[0][1]
    decision = prob_deny < thres
    
    return prob_deny, decision


def load_customer_data(n_sample=None):
    """
    Load dataset containing customer information (after feature engineering).
    Dataset is restricted to the n_sample-th first customers.
    
    params:
        n_sample : 
            Number of customers to include in the dataset.
            If None, all customers are included.
    
    returns:
        A DataFrame containing preprocessed data describing all customers.
    """
    # Load data
    file = 'customers_values.csv.gzip'
    df_cust=pd.read_csv(source_path+file, compression='gzip')
    df_cust.index = df_cust['SK_ID_CURR']
    
    # Select features and sample dataset
    l_drop = ['Unnamed: 0', 'TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index', 'source']
    feats = [f for f in df_cust.columns if f not in l_drop]
    df_cust=df_cust[feats]
    
    if n_sample != None :
        df_cust = df_cust.iloc[:n_sample]
    
    return df_cust


def load_explainer(n_expl):
    """
    Returns the list of Shapley explainers (one per LightGBM classifier)
    
    params:
        n_expl :
            number of the shap explainer to load
        
    returns:
        the shap explainer number n_expl
    """
    file = f'shapExplainers/ShapExplainer_{n_expl}.joblib'
    explainer = joblib.load(source_path+file)
    
    return explainer


def shap_explain(customer_id, df_cust, n_expl=5):
    """
    Gives shapley values for each feature for a given customer.
    
    params:
        customer_id :
            The customer for whom to explain loan decision ;
        df_cust :
            A customer description DataFrame ;
        n_expl :
            numbers of explainers to load (from 0 to 4).
            
    returns :
        shap_vals : 
            A numpy array containing aggregated shapley values ;
        base_value :
            the aggregated base value for all selected explainers.
    """
    customer_values = df_cust.loc[customer_id].values.reshape(1,-1)
    shap_vals = np.zeros(customer_values.shape)
    base_value=0
    
    # Aggregate explainations from every SHAP explainer
    for i in range(n_expl):
        explainer = load_explainer(i)
        shapley = explainer.shap_values(customer_values)[0]
        shap_vals += (shapley/n_expl)
        base_value += (explainer.expected_value[0]/n_expl)
        
    return shap_vals, base_value


def load_shap_values():
    """
    Load dataset containing shap values for a sample of 1000 customers.
    
    returns:
        A DataFrame containing shapley values for all criteria per customer.
    """
    # Load data
    file='shap_values.csv.gzip'
    df_shap = pd.read_csv(source_path+file, compression='gzip', index_col=0)
    
    return df_shap


def load_criteria_descriptions():
    """
    Load a table containing description of each criteria.
    
    returns:
        Filtered HomeCredit columns description provided for the project.
    """
    # Load data
    file='HomeCredit_columns_description.csv'
    df_crit = pd.read_csv(source_path+file, encoding ='cp1252', usecols=[2,3])

    # Drop customer id and target
    l_drop = ['SK_ID_CURR', 'TARGET']
    df_crit=df_crit[~df_crit['Row'].isin(l_drop)]
    
    # Define options for dropdown
    df_crit['options']=df_crit['Row'].apply(lambda x: {'label':x, 'value':x})
    
    return df_crit

    
def load_panel():
    """
    Loads data to construct the histogram of representative customer panel.
    """
    file = 'panel_hist.joblib'
    heights, bins = joblib.load(source_path+file)
    heights=100*heights/heights.sum()
    
    return (heights, bins)
    
    
def plot_panel(panel_hist, thres):
    """
    Display estimated risk on a representative panel of customers.
    
    params:
        panel_hist:
            A numpy histogram representing the distribution of estimated risk
            on a representative customer panel.
        thres:
            Threshold risk value above which a customer's loan is denied.
            
    returns:
        A figure displaying distribution of estimated risk on the panel.
    """
    # Plot customer risk distribution
    fig = go.Figure([go.Bar(x=panel_hist[1], y=panel_hist[0], width=0.01)])
    
    fig.update_layout(
        yaxis_title='% of customers', xaxis_tickformat = ',.0%',
        title='Distribution of estimated risk on a representative panel',
        #labels={'TARGET':'Estimated risk'},
        margin_t=30
    )
    
    # Display threshold
    fig.add_shape(type='line', x0=thres, x1=thres, y0=0, y1=15, 
                  line_color='red', line_dash='dot')
    fig.add_annotation(text='Maximum allowed risk ({:.0%})'.format(thres),
                       x=thres, y=15)
    
    return fig


def plot_waterfall(df_cust, customer_id, n_top, thres, base_value, shaps):
    """
    Calculate waterfall based on shapley values for a given customer.
    
    params:
        df_cust :
            A customer description DataFrame.
        customer_id :
            The SK_ID_CURR value of the customer for whom application decision
            will be explained.
        n_top:
            Number of top criteria to display.
        thres:
            Threshold risk value above which a customer's loan is denied.
        base_value : 
            the aggregated base value for selected shap explainers.
        shaps :
            A numpy array containing aggregated shapley values.
            
    returns:
        The waterfall figure for selected customer.
        Loan applications with a final score below 0 are denied.
    """ 
    # Set data for waterfall   
    df_waterfall=pd.DataFrame(shaps.T, index = df_cust.columns)
    df_waterfall.columns = ['values']
    df_waterfall['abs']=df_waterfall['values'].apply('abs')
    df_waterfall.sort_values(by='abs', inplace=True)
    
    # Aggregate shap values not in top n
    df_top=df_waterfall.tail(n_top)
    df_others = pd.DataFrame(df_waterfall.iloc[:-n_top].sum(axis=0)).T
    df_others.index = [f'others (n={len(df_waterfall.iloc[:-n_top])})']
    df_waterfall = df_others.append(df_top)
    
    # Plot waterfall
    fig = go.Figure(
        go.Waterfall(
            base = base_value,
            orientation = 'h',
            y=df_waterfall.index,
            x=df_waterfall['values']),
        layout = go.Layout(
            height=200+(25*n_top), 
            #width=600,
            xaxis_title='Confidence score', 
            yaxis_title = 'Criteria',
            yaxis_side='right',
            yaxis_tickfont=dict(size=10),
            margin_l=10, margin_r=10, 
            margin_t=30, margin_b=10
        )
    )
    
    # Add base value and final result on the plot
    fig.add_shape(type='line', x0=base_value, x1=base_value, y0=-1, y1=1)
    fig.add_annotation(text='Base value', x=base_value, y=0)
    
    final_value = df_waterfall['values'].sum() + base_value
    fig.add_shape(type='line', x0=final_value, x1=final_value, y0=n_top, y1=n_top+1)
    fig.add_annotation(text='score = {:.3}'.format(final_value), 
                       x=final_value, y=n_top+1)
    
    # Threshold line
    fig.add_shape(type='line', x0=0, x1=0, y0=-1, y1=n_top+1, 
                  line_color='red', line_dash='dot')
    
    return fig


def generate_top_tables(n_top, df_cust, customer_id, shaps):
    """
    For a given customer id, retrieves the n_top criteria having most impact on loan decision.
    
    params:
        n_top:
            Number of top criteria to display.
        df_cust:
            A customer description DataFrame.
        customer_id :
            The SK_ID_CURR value of the customer for whom main criteria are searched.
        shaps :
            A numpy array containing aggregated shapley values.
    
    returns:
        Two tables of main criteria :
        - Top decisive criteria for selected customer;
        - Top decisive overall criteria compared to the customer values.
    """
    # Retrieve shap values for selected customer 
    df_1 =pd.DataFrame(shaps[0].T, index = df_cust.columns)
    df_1.columns=['Impact']

    # Retrieve criteria values for selected customer
    df_2 = df_cust.loc[[customer_id]].T
    df_2.columns=['Values']

    # Merge
    df_table = df_1.merge(df_2, left_index=True, right_index=True)
    df_table['abs']=df_1['Impact'].apply('abs')
    df_table['Criteria'] = df_table.index
    df_table['Criteria']=df_table['Criteria'].apply(lambda x:str(x)[:30])
    
    # Top n table sorted by impact for selected customer
    df_table_c = df_table.sort_values(by='abs', ascending=False)
    df_table_c = df_table_c[['Criteria', 'Values', 'Impact']].head(n_top)
    df_table_c = df_table_c.applymap(lambda x: round(x,3) if pd.api.types.is_number(x) else x)
    
    child_c = [
        html.Div(children=[
            html.H3(children=f'Top {n_top} criteria - Customer'),
            html.Table([
                html.Thead(
                    html.Tr([html.Th(col) for col in df_table_c.columns])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(
                            children=df_table_c.iloc[i][col], 
                            style={'fontSize':10}
                        ) for col in df_table_c.columns
                    ]) for i in range(len(df_table_c))])
                ])
        ], className='five columns'
    )]
    
    # Top table sorted by mean absolute impact for all customers
    file = 'mean_abs_shaps.joblib'
    mean_abs_shaps = joblib.load(source_path+file)
    
    overall_top = mean_abs_shaps.sort_values(ascending=False).head(n_top)
    df_overall = df_table.loc[overall_top.index]
    df_overall['Mean impact'] = overall_top
    df_overall['Criteria'] = df_overall.index
    df_overall['Criteria']=df_overall['Criteria'].apply(lambda x:str(x)[:30])
    df_overall = df_overall.applymap(lambda x: round(x,3) if pd.api.types.is_number(x) else x)
    df_overall = df_overall[['Criteria', 'Mean impact', 'Values', 'Impact']]

    child_o = [
        html.Div(children=[
            html.H3(children=f'Top {n_top} criteria - Overall'),
            html.Table([
                html.Thead(html.Tr([html.Th(col) for col in df_overall.columns])),
                html.Tbody([
                    html.Tr([
                        html.Td(
                            children=df_overall.iloc[i][col],
                            style={'fontSize':10}
                        ) for col in df_overall.columns
                    ]) for i in range(len(df_overall))])
            ], style={"textOverflow":'ellipsis', 'overflow':'hidden', 'maxWidth':0})
        ], className='seven columns'
    )]
    
    # Append and return children
    children=child_c+child_o
    return children


def plot_shap_scatter(df_cust, df_shap, crit, cust, shaps, thres):
    """
    Shows evolution of SHAP value depending on selected criteria's value.
    
    params:
        df_cust:
            A customer description DataFrame.
        crit:
            The criteria to we want to plot.
        cust:
            A customer's ID.
            If not None, shows where the selected customer stands on the plot.
        shaps :
            A numpy array containing aggregated shapley values.
        thres:
            Threshold risk value above which a customer's loan is denied.
            
    returns:
        A partial dependence plot, where x is the criteria value and y the SHAP value.
    """
    # Shap values
    s_shap=df_shap[[crit, 'est_risk']].copy()
    s_shap['est_risk'] = s_shap['est_risk'].clip(0,thres)
    s_shap.columns=['shap_value', 'estimated_risk']
    
    # Criteria values
    s_vals = df_cust[[crit]].copy()
    s_vals.columns=['crit_value']
    
    df_summary = s_shap.join(s_vals)
    
    # Correspondence table
    corr_tab = {
        'NAME_CONTRACT_TYPE':{0:'Cash loans', 1:'Revolving loans'},
        'CODE_GENDER':{0:'F', 1:'M'}
    }
    
    if crit in corr_tab.keys():
        df_summary['new_crit_value'] = df_summary['crit_value'].replace(
            corr_tab[crit])
    else :
        df_summary['new_crit_value'] = df_summary['crit_value']
    
    # Visualization
    fig=px.scatter(df_summary, 
                   x='new_crit_value', y='shap_value', 
                   color='estimated_risk',
                   color_continuous_scale=['green', 'yellow', 'red']
                  )
    
    fig.update_layout(
        xaxis_title='Criteria value',  yaxis_title = 'Impact', margin_t=30
    )
    
    # Selected customer data
    if cust is not None:
        # Customer Shap and value for selected criteria
        cust_value=df_cust.loc[cust, crit]
        
        df_shaps=pd.DataFrame(shaps[0], index = df_cust.columns)
        cust_shap=df_shaps.loc[crit, 0]
        
        # Show selected customer on the plot
        fig.add_shape(type='line', 
              x0=cust_value, x1=cust_value, 
              y0=df_summary['shap_value'].min(), y1=cust_shap,
              line_dash='dot', line_color='red')

        if crit in corr_tab.keys():
            xmin=df_summary['new_crit_value'].iloc[0]
        else:
            xmin=df_summary['crit_value'].min()
        
        fig.add_shape(type='line', 
                      x0=xmin, x1=cust_value, 
                      y0=cust_shap, y1=cust_shap,
                      line_dash='dot', line_color='red')

    return fig

