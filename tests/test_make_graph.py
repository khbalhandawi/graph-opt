import networkx as nx
import json
import os

if __name__ == "__main__":

    # Create a MultiDiGraph
    MG = nx.MultiDiGraph()

    # Add nodes to the graph
    MG.add_node('0', iatacode='AAA', longitude=-145.508333333, latitude=-17.355, subregion='Polynesia', region='Asia Pacific')
    MG.add_node('1', iatacode='BBB', longitude=100.508333333, latitude=20.355, subregion='East Asia', region='Asia Pacific')
    MG.add_node('2', iatacode='CCC', longitude=-70.508333333, latitude=30.355, subregion='North America', region='Americas')
    MG.add_node('3', iatacode='DDD', longitude=-50.508333333, latitude=40.355, subregion='South America', region='Americas')
    MG.add_node('4', iatacode='EEE', longitude=30.508333333, latitude=50.355, subregion='West Asia', region='Asia Pacific')

    # Add edges to the graph
    MG.add_edge('0', '1', edge_type='VT', from_='AAA', to='BBB', market_share_percent=60.0, weight=5000)
    MG.add_edge('0', '1', edge_type='HT', from_='AAA', to='BBB', market_share_percent=40.0, weight=3000)
    MG.add_edge('0', '2', edge_type='AT', from_='AAA', to='CCC', market_share_percent=100.0, weight=8789)
    MG.add_edge('0', '4', edge_type='CT', from_='AAA', to='EEE', market_share_percent=100.0, weight=2000)
    MG.add_edge('1', '2', edge_type='BT', from_='BBB', to='CCC', market_share_percent=100.0, weight=6000)
    MG.add_edge('3', '4', edge_type='CT', from_='DDD', to='EEE', market_share_percent=80.0, weight=4000)
    MG.add_edge('3', '4', edge_type='HT', from_='DDD', to='EEE', market_share_percent=20.0, weight=2000)
    MG.add_edge('4', '0', edge_type='VT', from_='EEE', to='AAA', market_share_percent=100.0, weight=7000)


    # Create a dictionary of airline information
    nodes_airlines_dict = {
        'AT': {
            'alliance': 'Alliance B'
        },
        'BT': {
            'alliance': 'Alliance B'
        },
        'CT': {
            'alliance': 'Alliance C'
        },
        'HT': {
            'alliance': 'Alliance A'
        },
        'VT': {
            'alliance': 'Alliance A'
        }
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the absolute directory of the script  
    data_dir = os.path.join('..','test_data')
    nx.readwrite.graphml.write_graphml(MG,os.path.join(script_dir, data_dir,'test_DG_AP_AL.graphml')) # store graph
    with open(os.path.join(script_dir, data_dir,'test_airline_nodes.json'),'w') as file:
        json.dump(nodes_airlines_dict,file,indent=4)