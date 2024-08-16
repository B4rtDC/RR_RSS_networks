# MongoDB database access
import pymongo
# domain inference
from tld import get_tld
# redirects
import requests
# network analysis
import networkit as nk
#import networkx as nx
# logging
import logging
logger = logging.getLogger()
logger.setLevel(logging.WARNING)
# storing data
import pickle
import os
import sys
from datetime import datetime
# dataframes
import pandas as pd
# plotting
from matplotlib import pyplot as plt
# computation
import numpy as np
import NEMtropy as nem # null model analysis

import tabulate

class Query:
    """Helper class to work with queries in MongoDB."""
    def __init__(self, query_dict):
        self.query_dict = query_dict

    def __repr__(self):
        return "Query: {}".format(str(self.query_dict))

    def matches(self, other_query_dict):
        """
        Check if a custom query matches a standard type.
        """
        return self._compare(self.query_dict, other_query_dict)

    def _compare(self, dict1, dict2):
        """
        helper function for matches to compare two dictionaries
        """
        for key, value in dict1.items():
            if key not in dict2:
                return False
            if isinstance(value, dict):
                if not self._compare(value, dict2[key]):
                    return False
            else:
                if value != dict2[key]:
                    return False
        return True
    
    def query(self):
        """
        return the query dictionary
        """
        return self.query_dict
    
# definitions of the queries
OR_QUERY = Query({"entities.urls": {"$exists": True}, "referenced_tweets": {"$exists": False}}) # updated to avoid overlap
RT_QUERY = Query({"referenced_tweets": {  "$exists": True},
            "referenced_tweets": {  "$elemMatch": {"type": "retweeted",
                                                   "entities.urls": {"$exists": True}}}})
RP_QUERY = Query({"referenced_tweets": {  "$exists": True},
            "referenced_tweets": {  "$elemMatch": {"type": "replied_to",
                                                   "entities.urls": {"$exists": True}}}})

# GENERATION OF THE NETWORK

def generate_user_aggregate_pipeline(Q, Nlimit=0):
    """
    generate_user_aggregate_pipeline(Q, Nlimit=0)

    Arguments:
        Q: query to filter the messages
        Nlimit: limit the number of messages to be considered

    Generates a pipeline for the aggregation framework of MongoDB to extract the users who have authored a message from a collection.
    """
    return [{"$match": Q},                                                    # only messages with urls
                    {"$limit": Nlimit if Nlimit > 0 else sys.maxsize},        # limit the number of messages
                    {"$group": {"_id": "$author.id",                          # group by user id, storing the name and the domain
                                "name": {"$first": "$author.name"}, 
                                "username": {"$first": "$author.username"}}}
                    ]

def update_external_domain_map(col, query, orig_to_external_domain_map, expand_shortened_links, rejected_domains , domain_shortening_services, Nlimit=0):
    """
    update_external_domain_map(col, query, orig_to_external_domain_map, expand_shortened_links, rejected_domains , domain_shortening_services, Nlimit=0)

    Arguments:
        col: collection to be queried
        query: query to filter the messages
        orig_to_external_domain_map: dictionary to be updated
        expand_shortened_links: boolean to indicate if the links should be expanded
        rejected_domains: list of domains to be rejected
        domain_shortening_services: list of domain shortening services
        Nlimit: limit the number of messages to be considered

    Go over all domains in a collection and expand the links if they are using a link shortening service. Each unique link will be mapped to a domain. 
    """
    if OR_QUERY.matches(query):
    #if query == OR_QUERY:
        pass
        # iterate over the tweets
        for res in col.find(query).limit(Nlimit):
            # go over all urls in the message
            for original_url in res['entities']['urls']:
                # continue if already encountered
                if original_url['expanded_url'] in orig_to_external_domain_map:
                    continue
                # parsing
                else:
                    # get the domain
                    url_domain = get_tld(original_url['expanded_url'], as_object=True).fld
                    # check if the domain is in the list of rejected domains
                    if url_domain in rejected_domains:
                        orig_to_external_domain_map[original_url['expanded_url']] = None
                    else:
                        if expand_shortened_links:
                            # check if the domain is a shortened one
                            if url_domain in domain_shortening_services:
                                # try to expand the shortened url
                                try:
                                    expanded_url = requests.head(original_url['expanded_url']).headers['location']
                                    # get the domain of the expanded url
                                    expanded_url_domain = get_tld(expanded_url, as_object=True).fld     
                                    # check if the expanded domain is in the list of rejected domains and update the map
                                    if expanded_url_domain in rejected_domains:
                                        orig_to_external_domain_map[original_url['expanded_url']] = None
                                    else:
                                        orig_to_external_domain_map[original_url['expanded_url']] = expanded_url_domain
                                except:
                                    orig_to_external_domain_map[original_url['expanded_url']] = None
                                
                                

                            else:
                                # store the domain
                                orig_to_external_domain_map[original_url['expanded_url']] = url_domain
                        else:
                            orig_to_external_domain_map[original_url['expanded_url']] = url_domain
    elif RT_QUERY.matches(query):
    #elif query == RT_QUERY: # or (query == RP_QUERY):
        # iterate over the tweets
        for res in col.find(query).limit(Nlimit):
            # go over all urls in the message
            for tweet in res["referenced_tweets"]:
                for original_url in tweet["entities"]["urls"]:
                    # continue if already encountered
                    if original_url['expanded_url'] in orig_to_external_domain_map:
                        continue
                    # parsing
                    else:
                        # get the domain
                        url_domain = get_tld(original_url['expanded_url'], as_object=True).fld
                        # check if the domain is in the list of rejected domains
                        if url_domain in rejected_domains:
                            orig_to_external_domain_map[original_url['expanded_url']] = None
                        else:
                            if expand_shortened_links:
                                # check if the domain is a shortened one
                                if url_domain in domain_shortening_services:
                                    try:
                                        # expand the shortened url
                                        expanded_url = requests.head(original_url['expanded_url']).headers['location']
                                        # get the domain of the expanded url
                                        expanded_url_domain = get_tld(expanded_url, as_object=True).fld 
                                        # check if the expanded domain is in the list of rejected domains and update the map
                                        if expanded_url_domain in rejected_domains:
                                            orig_to_external_domain_map[original_url['expanded_url']] = None
                                        else:
                                            orig_to_external_domain_map[original_url['expanded_url']] = expanded_url_domain
                                    except:
                                        orig_to_external_domain_map[original_url['expanded_url']] = None
                                else:
                                    # store the domain
                                    orig_to_external_domain_map[original_url['expanded_url']] = url_domain
                            else:
                                orig_to_external_domain_map[original_url['expanded_url']] = url_domain
    
    elif RP_QUERY.matches(query):
    #elif query == RP_QUERY:
        # we iterate of replies to a tweet containing an url
        # iterate over the tweets
        for res in col.find(query).limit(Nlimit):
            # go over all urls in the message
            for tweet in res["referenced_tweets"]:
                if tweet["type"] == "replied_to":
                    for original_url in tweet["entities"]["urls"]:
                        # continue if already encountered
                        if original_url['expanded_url'] in orig_to_external_domain_map:
                            continue
                        # parsing
                        else:
                            # get the domain
                            url_domain = get_tld(original_url['expanded_url'], as_object=True).fld
                            # check if the domain is in the list of rejected domains
                            if url_domain in rejected_domains:
                                orig_to_external_domain_map[original_url['expanded_url']] = None
                            else:
                                if expand_shortened_links:
                                    # check if the domain is a shortened one
                                    if url_domain in domain_shortening_services:
                                        try:
                                            # expand the shortened url
                                            expanded_url = requests.head(original_url['expanded_url']).headers['location']
                                            # get the domain of the expanded url
                                            expanded_url_domain = get_tld(expanded_url, as_object=True).fld 
                                            # check if the expanded domain is in the list of rejected domains and update the map
                                            if expanded_url_domain in rejected_domains:
                                                orig_to_external_domain_map[original_url['expanded_url']] = None
                                            else:
                                                orig_to_external_domain_map[original_url['expanded_url']] = expanded_url_domain
                                        except:
                                            orig_to_external_domain_map[original_url['expanded_url']] = None
                                    else:
                                        # store the domain
                                        orig_to_external_domain_map[original_url['expanded_url']] = url_domain
                                else:
                                    orig_to_external_domain_map[original_url['expanded_url']] = url_domain
        
def add_bipartite_edges(col, query, G, revusermap, revdomainmap, orig_to_external_domain_map, Nlimit):
    """
    add_bipartite_edges(col, query, G, revusermap, revdomainmap, orig_to_external_domain_map, Nlimit)

    Arguments:
        col: collection to be queried
        query: query to filter the messages
        G: bipartite graph to be updated
        revusermap: dictionary to map user ids to node ids
        revdomainmap: dictionary to map domains to node ids
        orig_to_external_domain_map: dictionary to map original urls to domains
        Nlimit: limit the number of messages to be considered

    Add edges to a bipartite graph G between users and domains. The edges are weighted by the number of times a user has shared a link to a domain.
    """
    if OR_QUERY.matches(query):
    #if query == OR_QUERY:
        # iterate over the tweets
        for res in col.find(query).limit(Nlimit):
            # go over all urls in the message
            for original_url in res['entities']['urls']:
                # get the domain
                target_domain = orig_to_external_domain_map[original_url['expanded_url']]
                if target_domain is None: # skip if the domain is in the list of rejected domains
                    continue    
                else:
                    source = revusermap[res['author']['id']]
                    target = revdomainmap[target_domain]
                    # add the edge and increase weight if already present
                    G.increaseWeight(source, target, 1)             # CHECK HER FOR WEIGTS IN ORIGINAL CASE
    elif RT_QUERY.matches(query):
    #elif query == RT_QUERY: # or (query == RP_QUERY):
        # iterate over the tweets
        for res in col.find(query).limit(Nlimit):
            # go over all urls in the message
            for tweet in res["referenced_tweets"]:
                for original_url in tweet["entities"]["urls"]:
                    # get the domain
                    target_domain = orig_to_external_domain_map[original_url['expanded_url']]
                    if target_domain is None: # skip if the domain is in the list of rejected domains
                        continue    
                    else:
                        source = revusermap[res['author']['id']]
                        target = revdomainmap[target_domain]
                        # add the edge and increase weight if already present
                        G.increaseWeight(source, target, 1)
    elif RP_QUERY.matches(query):
    #elif query == RP_QUERY:
        # iterate over the tweets
        for res in col.find(query).limit(Nlimit):
            # go over all urls in the message
            for tweet in res["referenced_tweets"]:
                if tweet["type"] == "replied_to":
                    for original_url in tweet["entities"]["urls"]:
                        # get the domain
                        target_domain = orig_to_external_domain_map[original_url['expanded_url']]
                        if target_domain is None: # skip if the domain is in the list of rejected domains
                            continue    
                        else:
                            source = revusermap[res['author']['id']]
                            target = revdomainmap[target_domain]
                            # add the edge and increase weight if already present
                            G.increaseWeight(source, target, 1)
            
def bipartite_user_domain_graph(col, directed=False, 
                                include_original=True, include_retweets=False, include_replies=False,
                                rejected_domains=set(["twitter.com"]), 
                                expand_shortened_links=False, 
                                domain_shortening_services=set(),
                                or_query=None, rt_query=None, rp_query=None,
                                Nlimit=0, **kwargs):
    """
        bipartite_user_domain_graph(col, Nlimit=0)

    This function builds a bipartite graph of users and domains

    Parameters
    ----------
        col: a mongoDB collection
        Nlimit: limit the number of tweets to be processed
        directed: if True, the graph is directed
        include_retweets: if True, retweets are included
        expand_shortened_links: if True, the shortening services are expanded
        rejected_domains: a set of domains to be rejected, by default the twitter domain is rejected
        domain_shortening_services: a set of domains that are used for shortening urls
        or_query: query to filter the original tweets, by default the OR_QUERY is used
        rt_query: query to filter the retweets, by default the RT_QUERY is used
        rp_query: query to filter the replies, by default the RP_QUERY is used
        kwargs: additional arguments to be passed


    Returns
    -------
        - networkit graph
        - dataframe holding the following node attributes: node_id, kind, description, author_id, author_name, author_username

    Notes
    -----
    - currently only works with tweets containing urls in the entities.urls field, does not work with retweets, replies, quotes or urls in the text field
    - shortened links are not expanded by default. This can be done using the requests library
    - ...
    """
    if or_query is None:
        logging.warning("No query for original tweets provided, using default query")
        or_query = OR_QUERY.query()
    if rt_query is None:
        logging.warning("No query for retweets provided, using default query")
        rt_query = RT_QUERY.query()
    if rp_query is None:
        logging.warning("No query for replies provided, using default query")
        rp_query = RP_QUERY.query()
    ## initiate the graph
    G = nk.graph.Graph(weighted=True, directed=directed)
    # initialise the node dataframe 
    df = pd.DataFrame(columns=["node_id", "kind", "description", "author_id", "author_name", "author_username", "domain_name"])

    ## Generate the network nodes from the users
    or_users = list(col.aggregate(generate_user_aggregate_pipeline(or_query, Nlimit=Nlimit))) if include_original else [] # original tweets
    rt_users = list(col.aggregate(generate_user_aggregate_pipeline(rt_query, Nlimit=Nlimit))) if include_retweets else [] # retweets
    rp_users = list(col.aggregate(generate_user_aggregate_pipeline(rp_query, Nlimit=Nlimit))) if include_replies else []  # replies
    users = or_users + rt_users + rp_users
    usermap = {number: user for user, number in zip(users, range(len(users)))}  # map node id to a user dict
    revusermap = {v["_id"]: k for k, v in usermap.items()}                      # map twitter user id to node id
    # add the user nodes to the graph
    G.addNodes(len(users))
    # set node attributes
    for i, u in enumerate(users):
        df.loc[len(df)] = [i, "user", "{} (@{})".format(u["name"], u['username']), u['_id'], u['name'], u['username'], None]

    ## Generate the network nodes from the domains
    # Generate the mapping between original and the external domains   
    orig_to_external_domain_map = dict()      
    update_external_domain_map(col, or_query, orig_to_external_domain_map, expand_shortened_links, rejected_domains, domain_shortening_services, Nlimit=Nlimit) if include_original else None
    update_external_domain_map(col, rt_query, orig_to_external_domain_map, expand_shortened_links, rejected_domains, domain_shortening_services, Nlimit=Nlimit) if include_retweets else None
    update_external_domain_map(col, rp_query, orig_to_external_domain_map, expand_shortened_links, rejected_domains, domain_shortening_services, Nlimit=Nlimit) if include_replies  else None
        
    # add the external domains to the graph
    external_domains = set(orig_to_external_domain_map.values())
    external_domains.remove(None) if None in external_domains else None # holds the set of external domains (this no longer includes None, for the rejected domains)
    domainmap = {number: domain for domain, number in zip(external_domains, range(len(users), len(users) + len(external_domains)))}  # map node id to a domain
    revdomainmap = {domain:node for node, domain in domainmap.items()}                                                               # map domain to node id
    # add the domain nodes to the graph
    G.addNodes(len(external_domains))
    for i, d in enumerate(external_domains):
        df.loc[len(df)] = [i+len(users), "domain", d, None, None, None, d]

    ## Generate the network edges
    add_bipartite_edges(col, or_query, G, revusermap, revdomainmap, orig_to_external_domain_map, Nlimit=Nlimit) if include_original else None
    add_bipartite_edges(col, rt_query, G, revusermap, revdomainmap, orig_to_external_domain_map, Nlimit=Nlimit) if include_retweets else None
    add_bipartite_edges(col, rp_query, G, revusermap, revdomainmap, orig_to_external_domain_map, Nlimit=Nlimit) if include_replies  else None

    # add the bipartite degree to the df
    df["BP_degree"] = df.apply(lambda row : G.degree(row["node_id"]), axis=1)

    ## return the graph and metadata
    return G, df
            
def add_domain_meta_info(g, df, usr_meta_col):
    """
    get_domain_meta_info(g, df, usr_meta_col)

    A helper function to add additional meteadata to the domain dataframe. In particular the number of active, closed and suspended users who intereacted what a specific domain.

    arguments:
        g: networkit graph representing the bipartite network between users and domains.
        df: pandas dataframe holding the metadata of the nodes in the network
        usr_meta_col: pymongo collection containing additional data on the users (e.g. status)

    returns: 
        the augmented dataframe

    notes:
        - for this ot be fast is is recommented to create an index on the username field in the user metadata collection (e.g. metadb_authorcol.create_index([("username", 1)], name='nameindex'))
    """
    def single_domain_meta_info(g, df, node_id, usr_meta_col):
        kind = df.loc[node_id, "kind"]
        
        # users have no specific info
        if kind == "user":
            return (None, None, None)
        elif kind == "domain":
            # get the number of active, closed and suspended users
            res = {"active": 0, "closed": 0, "suspended": 0}
            for neigh in g.iterNeighbors(node_id):
                # retrieve the user metadata from the database
                u = usr_meta_col.find_one({"username":{"$eq":df.author_username[neigh]}})
                if u is None:
                    continue
                res[u["status"]] += 1
            return res["active"], res["closed"], res["suspended"]

        else:
            raise ValueError("Unknown node type: {}".format(kind))
        
        return

    df[["active_users", "closed_users","suspended_users"]] = pd.DataFrame(df.apply(lambda row : single_domain_meta_info(g, df, row.node_id, usr_meta_col), axis=1).tolist())
    df["ratio_active_users"] = df["active_users"] / (df["active_users"] + df["closed_users"] + df["suspended_users"])
    df["ratio_suspended"] = (df["closed_users"] + df["suspended_users"]) / (df["active_users"] + df["closed_users"] + df["suspended_users"])
    return df

def add_user_meta_info(g, df, usr_meta_col):
    """
    add_user_meta_info(g, df, usr_meta_col)

    A helper function to add additional meteadata to the domain dataframe. In particular the status of the user (active, closed, suspended).

    arguments:
        g: networkit graph representing the bipartite network between users and domains.
        df: pandas dataframe holding the metadata of the nodes in the network
        node_id: the node id of the node in the network
        usr_meta_col: pymongo collection containing additional data on the users (e.g. status)

    returns: 
        the augmented dataframe
    """
    def single_user_meta_info(g, df, node_id, usr_meta_col):
        kind = df.loc[node_id, "kind"]
        # domains have no specific info
        if kind == "domain":
            return None
        elif kind == "user":
            # get the user status
            u = usr_meta_col.find_one({"username":{"$eq":df.author_username[node_id]}})
            if u is None:
                return None
            return u["status"]
        else:
            raise ValueError("Unknown node type: {}".format(kind))

        return

    df["author_status"] = df.apply(lambda row : single_user_meta_info(g, df, row.node_id, usr_meta_col), axis=1)

    return df


# SAVING: Downside of using these networkit objects: they are not serializable due to the underlying C dependency, so we cannot save them to a file. We can kind of work around this, by storing the edges with their weights and the node properties as a serializable object. We can then use this object to build the bipartite network each time we want to use it.

def write_out_node_edge_lists(filename, G, df, id_column="node_id", label_column="description", add_DTG=False):
    """write_out_node_edge_lists(G, filename, *nodeattributes):

    This function writes out the node and edge lists of a networkit graph to a csv file. Intended for use with Gephi.

    Parameters
    ----------
        filename: the name of the file to write to. Will be appended with "_nodes.csv" and "_edges.csv"
        G: a networkit graph
        df: a pandas dataframe holding the node attributes
        id_column: the name of the column in the dataframe holding the graph node ids
        label_column: the name of the column that will be used for the default Label attribute in Gephi
        add_DTG: if True, the current date and time will be appended to the filename
    """
    if add_DTG:
        globalpath = os.path.split(filename)
        filename = os.path.join(*globalpath[:-1], "{}{}".format(datetime.now().strftime("%Y-%m-%d_"), globalpath[-1]))
    if filename.endswith(".csv"):
        # strip csv if already persent in name
        filename = filename[:-4]
        
    ## write out the nodes
    # update the node id
    df["Id"] = df[id_column]
    # add the default label column
    df["Label"] = df[label_column]
    # write out 
    df.to_csv("{}_nodes.csv".format(filename), index=False, sep=";")
    
    # write out the edges
    with open(filename + "_edges.csv", "w") as f:
        f.write("source;target;weight\n")
        for u, v, w in G.iterEdgesWeights():
            f.write("{};{};{}\n".format(u, v, w))

def myserializer(fname, g, nodeproperties=None, add_DTG=False):
    """myserializer(fname, g, nodeproperties=None, add_DTG=False)

    This function serializes a networkit graph to a pickle file. 

    Parameters
    ----------
        g: a networkit graph
        nodeporoperties: a dataframe holding the node properties
        add_DTG: if True, the current date and time will be appended to the filename
    """
    if add_DTG:
        globalpath = os.path.split(fname)
        fname = os.path.join(*globalpath[:-1], "{}{}".format(datetime.now().strftime("%Y-%m-%d_"), globalpath[-1]))

    if not fname.endswith(".pkl"):
        fname += ".pkl"
    
    node_count = g.numberOfNodes()
    is_directed = g.isDirected()
    is_weighted = g.isWeighted()
    edge_it = g.iterEdgesWeights() if is_weighted else g.iterEdges()
    edges = [link for link in edge_it]
    
    # write to a pickle file
    with open(fname, "wb") as f: # "wb" because we want to write in binary mode
        if nodeproperties is None:
            pickle.dump((node_count, is_directed, is_weighted, edges), f)
        else:   
            pickle.dump((node_count, is_directed, is_weighted, edges, nodeproperties), f)

    return

def null_model_serializer(fname, model, proj:str = None, nodeproperties=None, add_DTG=False):
    """
    null_model_serializer(fname, g, nodeproperties=None, add_DTG=False)

    This function serializes a networkit graph to a pickle file, after removing the 
    edges that are not statistically significant.

    Parameters
    ----------
        fname: the name of the file to write to. Will be appended with _domainproj_edges_filtered.pkl
        g: a networkit graph
        nodeporoperties: a dataframe holding the node properties
        add_DTG: if True, the current date and time will be appended to the filename
    
    Notes
    -----
        Currently not used, but could be useful in the future.
    """
    if proj is None:
        raise ValueError("You must specify a projection type to use this function")
    if proj not in ["domain", "user"]:
        raise ValueError("Unknown projection type: {}".format(proj))
    if proj == "domain":
        projection_details = "_domainproj_filtered.pkl"
    elif proj == "user":
        projection_details = "_userproj_filtered.pkl"

    if add_DTG:
        globalpath = os.path.split(fname)
        fname = os.path.join(*globalpath[:-1], "{}{}{}".format(datetime.now().strftime("%Y-%m-%d_"), globalpath[-1], projection_details))

    if not fname.endswith(".pkl"):
        fname += ".pkl"

    raise NotImplementedError("This function is not yet completely implemented")

    return

def mydeserializer(fname):
    """
    mydeserializer(fname)

    Deserialize a networkit graph from a pickle file. Will load up the graph and the node properties if possible.   

    arguments:
        fname: filename

    returns:
        - a networkit graph
        - a dataframe holding the node properties (if any)
    """
    with open(fname, "rb") as f: # "rb" because we want to read in binary mode
        loaded = pickle.load(f)
    
    node_count, is_directed, is_weighted, edges = loaded[0:4]
    nodeprops = loaded[-1] if len(loaded) == 5 else None

    # create the graph
    g = nk.Graph(node_count, weighted=is_weighted, directed=is_directed)
    # add the edges
    for e in edges:
        g.addEdge(*e)
            
    return g, nodeprops


# PROJECTING: We can also project the bipartite network onto one of the bipartitions. This is useful if we want to analyze the network of users or the network of domains. We can also keep the weights of the edges in the projected network, or we can set them to 1.
def project_bipartite_network(G, nodeproperties, layer=None, keepweights=False):
    """
    project_bipartite_network(G, nodeproperties, layer, keepweights)

    This function projects a bipartite network onto one of the bipartitions. The attribute 'layer' is used to identify the two layers.

    Parameters
    ----------
        G: a networkit graph
        nodeproperties: a dataframe with the node properties
        layer: the layer to project onto. Should be "user" or "domain"
        keepweights: if True, the weights of the edges in the projected network are the sum of the weights of the edges in the original network that are projected onto it. If False, the weights are 1.

    Returns
    -------
        - networkit graph of the projected network
        - dataframe with the node properties of the projected network
    """

    # get the (sparse) adjacency matrix of the bipartite graph
    G_adj = nk.algebraic.adjacencyMatrix(G) if keepweights else nk.algebraic.adjacencyMatrix(nk.graphtools.toUnweighted(G))
    
    # Get the layer composition
    Nu = len(nodeproperties[nodeproperties["kind"] == "user"])
    Nd = len(nodeproperties[nodeproperties["kind"] == "domain"])
    assert Nu + Nd == G.numberOfNodes()

    # get the (sparse) adjacency matrix of the projected graph
    if layer == "domain":
        G_adj_p = G_adj[Nu:, :Nu] * G_adj[:Nu, Nu:]
    elif layer == "user":
        G_adj_p = G_adj[:Nu, Nu:] * G_adj[Nu:, :Nu]
    else:
        raise ValueError("Layer should be 'user' or 'domain'")
        
    
    # avoid self-loops
    G_adj_p.setdiag(0)
    G_adj_p.eliminate_zeros() # don't explicitely store zeros

    ## generate the projected graph
    G_p = nk.graph.Graph(n = G_adj_p.shape[0], weighted=True) # updated for weights
    # add the edges
    for (i, j) in zip(*G_adj_p.nonzero()):
        if not G_p.hasEdge(i, j): # workaround for the checkMultiEdge bug
            w = G_adj_p[i,j] if keepweights else 1
            G_p.addEdge(i, j, w) # checkMultiEdge=True throws an error ?!?


    ## generate the projected graph node properties
    df_p = nodeproperties.loc[nodeproperties["kind"] == layer,:].copy()
    df_p.loc[:,"node_id"] = df_p["node_id"] - Nu if layer == "domain" else df_p["node_id"]
    
    return G_p, df_p


def apply_null_model(filename, G, df, add_DTG=False):
    """
    apply_null_model(filename, G, df, add_DTG=False)

    Apply the bipartite null model to extract statisticall significant links
    """
    if add_DTG:
        globalpath = os.path.split(filename)
        filename = os.path.join(*globalpath[:-1], "{}{}".format(datetime.now().strftime("%Y-%m-%d_"), globalpath[-1]))
    if filename.endswith(".csv"):
        # strip csv if already persent in name
        filename = filename[:-4]

    
    # we start from the unweighted adjacency matrix of the bipartite graph
    G_adj = nk.algebraic.adjacencyMatrix(nk.graphtools.toUnweighted(G))
    Nu, Nd = df.groupby("kind").count().loc["user", "node_id"], df.groupby("kind").count().loc["domain", "node_id"]
    assert Nu + Nd == G.numberOfNodes()
    
    # Get the subset adjacency matrices
    G_adj_sub = G_adj[Nu:, :Nu]
    d_usr = np.array(np.sum(G_adj_sub, axis=0)).flatten()
    d_dom = np.array(np.sum(G_adj_sub, axis=1)).flatten()
    # Get the adjacency list (key: id in row (user), value: id in column (domain))
    # mapping of user -> domain
    adjacency_list = dict()
    for i in range(Nu):
        adjacency_list[i] = list(map(lambda x: x - Nu, G.iterNeighbors(i)))
        assert len(adjacency_list[i]) == G.degree(i)
    model = nem.BipartiteGraph(adjacency_list=adjacency_list)
    
    # sanity checks (dimensions)
    assert (model.n_cols, model.n_rows) == G_adj_sub.shape  # check that the dimensions are correct
    assert (model.cols_deg == d_dom).all()            # check that the degrees are correct
    assert (model.rows_deg == d_usr).all()            # check that the degrees are correct
    
    # compute the null model
    # compute the bipartite graph entropy
    model.solve_tool(method="fixed-point", max_steps=1000, tol=1e-8, verbose=False, initial_guess="degrees")
    
    # compute the projection onto the domain layer
    model.compute_projection(rows=False, threads_num=2, progress_bar=True)#, alpha=1e-6)
    # write out the result if applicable
    if len(model.projected_cols_adj_list) > 0:
        with open("{}_domainproj_edges_filtered.csv".format(filename), "w") as f:
            f.write("source;target\n")
            for (s, T) in model.projected_cols_adj_list.items():
                for t in T:
                    f.write("{};{}\n".format(s, t))

    # compute the projection onto the user layer
    model.compute_projection(rows=True, threads_num=2, progress_bar=True)
    # write out the result if applicable
    if len(model.projected_rows_adj_list) > 0:
        with open("{}_userproj_edges_filtered.csv".format(filename), "w") as f:
            f.write("source;target\n")
            for (s, T) in model.projected_rows_adj_list.items():
                for t in T:
                    f.write("{};{}\n".format(s, t))


# helper function to get the components of a graph, sorted by size
def get_components(G):
    comps = nk.components.ConnectedComponents(G).run()
    comps = comps.getComponents()
    comps.sort(key=lambda x : len(x), reverse=True)

    return comps

def extract_specific_components(G, comp_id):
    return





import pandas as pd
import requests
from bs4 import BeautifulSoup
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import pycountry

def detect_website_language(url):
    """
    Detects the language of a website given its URL.

    Args:
        url (str): The URL of the website to detect the language of.

    Returns:
        str: The name of the language of the website, or 'unknown' if the language could not be detected.
    """
    try:
        # Check if url scheme is present
        if not url.startswith(('http://', 'https://')):
            # Add https:// if no scheme is present
            url = 'https://' + url
            
        # Send a GET request
        response = requests.get(url)
        
        # Parse the website content with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Use langdetect to detect the language
        lang_code = detect(soup.get_text())
        
        # Convert language code to language name using pycountry
        lang = pycountry.languages.get(alpha_2=lang_code)
        return lang.name if lang else 'unknown'
    except (requests.RequestException, LangDetectException):
        # Return 'unknown' if the website language could not be detected
        return 'unknown'
