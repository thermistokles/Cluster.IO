
#Dash related imports
import os
import base64
from dash import Dash, Output, Input, State, MATCH
from dash import html
from dash import dcc
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import dash_daq as daq
import asyncio
import random
import string
import zipfile
import pandas as pd

from os import listdir
from os.path import isfile, join

from server import resolve_kMeans_cluster, resolve_birch_cluster, resolve_agglomerative_cluster, resolve_dbscan_cluster, resolve_Datasets, resolve_uploadDataset, get_column_names

#Dash application
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, dbc.icons.BOOTSTRAP, "dash_app.css", dbc.icons.BOOTSTRAP], suppress_callback_exceptions=True)

dataset_features = []

#Initialize datasets
datasets = list(resolve_Datasets(None, None, "Project_3"))

#Datasets radio buttons
def create_radio_button(value):
    radioitem = html.Div(
        [
            dbc.RadioItems(
                options=[
                    {"value": value},
                ],
                id="radioitems-" + value,
            ),
        ]
    )

    return radioitem

#Datasets delete button
def create_delete_button(index):
    delete_icon = html.I(className="bi bi-trash")

    return dbc.Button(
        delete_icon, color="danger",
        className="me-1",
        # id={
        #     'type' : 'delete_button',
        #     'index' : index
        # }
    )

#Delete dataset functionality based on pattern matching
# @app.callback(
#     Output({'type': 'my-delete-button-output', 'index': MATCH}, 'children'),
#     Input({'type': 'delete_button', 'index': MATCH}, 'index'),
# )
# def delete_dataset(index):
#     if index:
#         print("dataset_name: ", index)
#         return ''
#     raise PreventUpdate

df = pd.DataFrame(
    {'select' : [create_radio_button(dataset['name']) for dataset in datasets],
    'Dataset' : [dataset['name'] for dataset in datasets],
    'Delete' : [create_delete_button(index) for index in range(len(datasets))]}
)

previous_results = []
previous_images = {}
cluster_details = {}


#This is the initial layout of the application
app.layout = dbc.Container(
    [
        dbc.NavbarSimple(
            children=[
            ],
            brand="CLUSTER.IO",
            brand_href="/",
            color="primary",
            dark=True,
        ),
        html.Br(),
        
        dbc.Row(
            [
                dbc.Col(
                    [
                        #Datasets card
                        dbc.Card(
                        [
                            html.H3("Datasets"),
                            dcc.Upload(
                            id='upload-data',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select 1 or more Datasets')
                            ]),
                            style={
                                'width': '80%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px',
                                'align': 'center'
                            },
                            # Allow multiple files to be uploaded
                            multiple=True
                        ),
                        html.Div(id='output-file-info'),
                            html.Label("Select a dataset"),
                            dcc.Dropdown(
                                id='Dataset',
                                options=[{'label': dataset['name'], 'value': dataset['name']} for dataset in datasets],
                                style={'width': '100%'}
                            ),
                            dbc.Table.from_dataframe(df, bordered=True, striped=True),
                            html.Div(id='my-delete-button-output'),
                            html.Br(),
                            daq.BooleanSwitch(
                                id='my-toggle-switch',
                                label='Compare results',
                                labelPosition='bottom',
                                on=False,
                                color="green",
                            ),
                            html.Div(id='my-toggle-switch-output')
                        ],
                        style={'padding' : '2%'}
                    ),
                ],
                width=3
            ),
                #Output container
                dbc.Col(
                    [
                        html.Div(id="output-container", style={'width': '100%', 'height': '100px'})
                    ],
                ),

                #Algorithms and parameters card
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                html.H3("Algorithms and Paramters"),
                                html.Label("Select a Clustering Algorithm"),
                                dcc.Dropdown(
                                    id='Algorithm',
                                    options=[
                                        {'label': 'K Means', 'value': 'K Means'},
                                        {'label': 'Birch', 'value': 'Birch'},
                                        {'label': 'Agglomerative', 'value': 'Agglomerative'},
                                        {'label': 'DBSCAN', 'value': 'DBSCAN'},
                                    ],
                                    className="dash-bootstrap"
                                ),
                                html.Br(),
                                html.Label("Enter number of clusters", id="clusters_label", style={'display': 'none'}),
                                dbc.Input(id="num_clusters", placeholder="num_clusters", type="number"),
                                html.Label("Enter number of random state", id="random_state_label", style={'display': 'none'}),
                                dbc.Input(id="random_state", placeholder="random_state", type="number"),
                                html.Label("Enter number of threshold", id="threshold_label", style={'display': 'none'}),
                                dbc.Input(id="threshold", placeholder="threshold", type="number", style={'display': 'none'}),
                                html.Label("Enter branching factor", id="branching_factor_label", style={'display': 'none'}),
                                dbc.Input(id="branching_factor", placeholder="branching_factor", type="number", style={'display': 'none'}),
                                html.Label("Enter Linkage criterion", id="linkage_label", style={'display': 'none'}),
                                dbc.Input(id="linkage", placeholder="linkage", type="text", style={'display': 'none'}),
                                html.Label("Enter Epsilon", id="eps_label", style={'display': 'none'}),
                                dbc.Input(id="eps", placeholder="eps", type="number", style={'display': 'none'}),
                                html.Label("Enter number of min_samples", id="min_samples_label", style={'display': 'none'}),
                                dbc.Input(id="min_samples", placeholder="min_samples", type="number", style={'display': 'none'}),
                                html.Label("Enter algorithm", id="dbs_algorithm_label", style={'display': 'none'}),
                                dcc.Dropdown(
                                    id='dbs_algorithm',
                                    options=[
                                        {'label': 'auto', 'value': 'auto'},
                                        {'label': 'ball_tree', 'value': 'ball_tree'},
                                        {'label': 'kd_tree', 'value': 'kd_tree'},
                                        {'label': 'brute', 'value': 'brute'},
                                    ],
                                    style={'width': '100%'}
                                ),
                                html.Br(),

                                #Check the datasets output
                                dbc.Label("Select Cluster Data On:"),
                                dcc.Checklist(id="radioitems-input"),

                                dbc.Button("Perform Clustering", color="success", className="me-1", id="perform-button"),
                            ],
                            style={'padding' : '2%'}
                        ),
                    ],
                    width=3
                ),
            ],
            style={'padding' : '2%'}
        ),
    ],
    fluid=True
)

#Behaviour of persist results toggle switch
@app.callback(
    Output('my-toggle-switch-output', 'children'),
    Input('my-toggle-switch', 'on')
)
def persist_results(value):
    global previous_results
    if value:
        return 'Clustering results will persist now'
    else:
        previous_results = []

#Upload file functionality
@app.callback(
    Output('output-file-info', 'children'),
    Output('Dataset', 'options'),
    Input('upload-data', 'filename'),
    Input('upload-data', 'contents'),
    prevent_initial_call=True
)
def uploadFile(filename, contents):
    global datasets

    if contents is None:
        return "Invalid file"

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    #Get list of files
    for name, data in zip(filename, contents):
        result = loop.run_until_complete(resolve_uploadDataset(None, None, name, data, "", "Project_3"))
    loop.close

    #Update the datasets list after the upload
    datasets = list(resolve_Datasets(None, None, "Project_3"))
    options = [{'label': dataset['name'], 'value': dataset['name']} for dataset in datasets]
    return f"File {filename[0]} uploaded successfully.", options

#Display dataset columns
@app.callback(
    Output('radioitems-input', 'options'),
    Input('Dataset', 'value'),
    prevent_initial_call=True
)
def updateClusteredDataOn(dataset):
    #global dataset_features

    features = []

    if dataset:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        #Get the column names when the user selects a dataset
        results = loop.run_until_complete(get_column_names(dataset))

        for result in results:
            features.append({'label' : result, 'value' : result})

    return features

#Updating the parameters according to the algorithm
@app.callback(
    #Parameters
    [Output("num_clusters", "style"),
     Output("random_state", "style"),
     Output("threshold", "style"),
     Output("branching_factor", "style"),
     Output("linkage", "style"),
     Output("eps", "style"),
     Output("min_samples", "style"),
     Output("dbs_algorithm", "style"),
     #Labels
     Output("clusters_label", "style"),
     Output("random_state_label", "style"),
     Output("threshold_label", "style"),
     Output("branching_factor_label", "style"),
     Output("linkage_label", "style"),
     Output("eps_label", "style"),
     Output("min_samples_label", "style"),
     Output("dbs_algorithm_label", "style"),],
    [Input("Algorithm", "value")]
)

def update_algorithm_inputs(selected_algorithm):
    #The parameters are initially hidden. They will be displayed when the user inputs an algorithm
    #Would be more efficient if we change this to pattern matching callbacks
    num_clusters_style = {'display': 'none'}
    random_state_style = {'display': 'none'}
    threshold_style = {'display': 'none'}
    branching_factor_style = {'display': 'none'}
    linkage_style = {'display': 'none'}
    eps_style = {'display': 'none'}
    min_samples_style = {'display': 'none'}
    dbs_algorithm_style = {'display': 'none'}

    clusters_label_style = {'display': 'none'}
    random_state_label_style = {'display': 'none'}
    threshold_label_style = {'display': 'none'}
    branching_factor_label_style = {'display': 'none'}
    linkage_label_style = {'display': 'none'}
    eps_label_style = {'display': 'none'}
    min_samples_label_style = {'display': 'none'}
    dbs_algorithm_label_style = {'display': 'none'}

    if selected_algorithm == 'K Means':
        clusters_label_style = {'display': 'block'}
        num_clusters_style = {'display': 'block'}
        random_state_label_style = {'display': 'block'}
        random_state_style = {'display': 'block'}

    elif selected_algorithm == 'Birch':
        clusters_label_style = {'display': 'block'}
        num_clusters_style = {'display': 'block'}
        threshold_label_style = {'display': 'block'}
        threshold_style = {'display': 'block'}
        branching_factor_label_style = {'display': 'block'}
        branching_factor_style = {'display': 'block'}

    elif selected_algorithm == "Agglomerative":
        clusters_label_style = {'display': 'block'}
        num_clusters_style = {'display': 'block'}
        linkage_style = {'display': 'block'}
        linkage_label_style = {'display': 'block'}

    elif selected_algorithm == "DBSCAN":
        eps_style = {'display': 'block'}
        eps_label_style = {'display': 'block'}
        min_samples_style = {'display': 'block'}
        min_samples_label_style = {'display': 'block'}
        dbs_algorithm_style = {'display': 'block'}
        dbs_algorithm_label_style = {'display': 'block'}

    return clusters_label_style, random_state_label_style, threshold_label_style, branching_factor_label_style, linkage_label_style, eps_label_style, min_samples_label_style, dbs_algorithm_label_style, num_clusters_style, random_state_style, threshold_style, branching_factor_style, linkage_style, eps_style, min_samples_style, dbs_algorithm_style

#Getting the output and updating the results
@app.callback(
    Output("output-container", "children"),
    [Input("perform-button", "n_clicks")],
    [
        State("num_clusters", "value"),
        State("random_state", "value"),
        State("threshold", "value"),
        State("branching_factor", "value"),
        State("linkage", "value"),
        State("eps", "value"),
        State("min_samples", "value"),
        State("dbs_algorithm", "value"),
        State("Dataset", "value"),
        State("Algorithm", "value"),
        State("radioitems-input", "value"),
        State("my-toggle-switch", "on")
    ]
)
def update_output(n_clicks, num_clusters, random_state, threshold, branching_factor, linkage, eps, min_samples, dbs_algorithm, dataset, algorithm, cluster_data_on, lock_results):
    #Default project is Project_3
    project = "Project_3"
    print("cluster_data_on: ", cluster_data_on)
    cluster_data_on = 'Modulus'

    if n_clicks:
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
        result = loop.run_until_complete(async_update_output(n_clicks, num_clusters, random_state, threshold, branching_factor, linkage, eps, min_samples, dbs_algorithm, project, dataset, algorithm, cluster_data_on, lock_results))
        loop.close()

        previous_results.insert(0, result)

        return previous_results if lock_results else result

    return ""

    
#Perform clustering according to the inputs
async def async_update_output(n_clicks, num_clusters, random_state, threshold, branching_factor, linkage, eps, min_samples, dbs_algorithm, project, dataset, algorithm, cluster_data_on, lock_results):

    #Get images based on input algorithm and parameters
    if algorithm == 'K Means':
        clustering_output, scores = await resolve_kMeans_cluster(None, None, num_clusters, random_state, dataset, cluster_data_on, project)
    if algorithm == 'Birch':
        clustering_output, scores = await resolve_birch_cluster(None, None, num_clusters, threshold, branching_factor, dataset, cluster_data_on,project)
    if algorithm == 'Agglomerative':
        clustering_output, scores = await resolve_agglomerative_cluster(None, None, num_clusters, linkage, dataset, cluster_data_on, project)
    if algorithm == 'DBSCAN':
        clustering_output, scores = await resolve_dbscan_cluster(None, None, eps, min_samples, dbs_algorithm, dataset, cluster_data_on, project)

    #Extract images and scores from the output
    raw_data_image = clustering_output.get('rawData', None)
    clustered_data_image = clustering_output.get('clusteredData', None)
    clusters_fractions_image = clustering_output.get('clustersFractions', None)
    silhouette_score = scores[0]
    davis_bouldin_score = scores[1]
    calinski_harabasz_score = scores[2]

    #Output images
    raw_data_html = html.Img(src=f"data:image/png;base64, {raw_data_image}" if raw_data_image else "", style={'max-width': '100%'})
    clustered_data_html = html.Img(src=f"data:image/png;base64, {clustered_data_image}" if clustered_data_image else "", style={'max-width': '100%'})
    clusters_fractions_html = html.Img(src=f"data:image/png;base64, {clusters_fractions_image}" if clusters_fractions_image else "", style={'max-width': '100%'})

    #Download button
    downloadIcon = html.I(className='bi bi-download', style=dict(display='inline-block', ))
    
    #Return dynamically generated output using with custom IDs for pattern matching
    result = dbc.Card([
        dbc.Row(
            [
                html.Br(),
                dbc.Col([
                    html.P(f"Dataset: {dataset}"),
                    html.P(f"Algorithm: {algorithm}"),
                    html.P(f"Num Clusters: {num_clusters}"),
                    html.P(f"Random State: {random_state}"),
                    html.P(f"Cluster Data On: {cluster_data_on}"),
                ],
                width = 6),
                dbc.Col([
                    dbc.Button(children=[downloadIcon], color="info", className="download", id={'type' : 'download_button', 'index' : n_clicks}),
                    dcc.Download(id={'type' : 'download-image', 'index' : n_clicks}),
                    dbc.Button("X", color="danger", className="me-1", id={'type' : 'close_button', 'index' : n_clicks}),
                ],
                style={'text-align' : 'right'},
                width = 6),
            ]
        ),
        dbc.Row(
        [
            dbc.Col(raw_data_html),
            dbc.Col(clustered_data_html),
            dbc.Col(clusters_fractions_html),
        ],
        ),
        dbc.Row(
            [
                html.P(f"Silhouette score: {silhouette_score}" ),
                html.P(f"Davis Bouldin score: {davis_bouldin_score}" ),
                html.P(f"Calinski Harabasz score: {calinski_harabasz_score}"),
            ]
        ),
    ],
    id={
        'type' : 'output_card',
        'index' : n_clicks
    })

    #Add images to hashmap for download functionality
    previous_images[n_clicks] = [raw_data_image, clustered_data_image, clusters_fractions_image]
    cluster_details[n_clicks] = {
        'Algorithm' : algorithm,
        'Number of clusters' : num_clusters,
        'random_state' : random_state,
        'threshold' : threshold,
        'branching_factor' : branching_factor,
        'linkage' : linkage,
        'eps' : eps,
        'min_samples' : min_samples,
        'dbs_algorithm' : dbs_algorithm
    }

    return result

#Close card functionality based on pattern matching
@app.callback(
        Output({'type': 'output_card', 'index': MATCH}, 'children'),
        Input({'type': 'close_button', 'index': MATCH}, 'n_clicks'),
        State({'type': 'output_card', 'index': MATCH}, 'id')
)
def close_card(n_clicks, card_id):
    card_index = card_id['index']
    if n_clicks:
        #Remove cards from previous results array.
        #Out of bounds condition handled here
        if card_index > len(previous_results):
            previous_results.pop(0)
        else:
            previous_results.pop(-card_index)
        del previous_images[card_index]
        return ''
    raise PreventUpdate

#Download images functionality based on pattern matching
@app.callback(
        Output({'type': 'download-image', 'index': MATCH}, 'data'),
        Input({'type': 'download_button', 'index': MATCH}, 'n_clicks'),
        State({'type': 'output_card', 'index': MATCH}, 'id')
)
def download_images(n_clicks, card_id):
    card_index = card_id['index']

    if n_clicks:

        images_to_download = previous_images[card_index]

        #compression = zipfile.ZIP_DEFLATED
        # create the zip file first parameter path/name, second mode
        zf = zipfile.ZipFile(str(card_index) + '.zip', mode="w")

        file_path = 'parameters.txt'

        # Open the file in write mode
        with open(file_path, 'w') as f:
            # Iterate over the dictionary items and write them to the file
            for key, value in cluster_details[n_clicks].items():
                if cluster_details[n_clicks][key]:
                    f.write(f'{key}: {value}\n')

        try:
            #Add the dataframe to zip
            zf.write('dataframe.xlsx', arcname='dataframe.xlsx')

            #Add images to zip
            for i, image_data in enumerate(images_to_download):
                image_binary = base64.b64decode(image_data)

                zf.writestr(f"image_{i+1}.png", image_binary)
            zf.write('parameters.txt', arcname='parameters.txt')
        except FileNotFoundError:
            print("An error occurred while zipping the files")
        finally:
            zf.close()

        return dcc.send_file(str(card_index) + '.zip')
    raise PreventUpdate

#Not being used. Might remove later
#Generate a random string based on the length of the previous_array
def generate_id():
    results_length = len(previous_results)

    alphanumeric_chars = string.ascii_letters + string.digits 

    if results_length > 0:
        id_string = ''.join(random.choice(alphanumeric_chars) for _ in range(results_length))

    else:
        id_string = "output_card"
    return id_string

if __name__ == '__main__':
    app.run(debug=True)
#Dash end
