# Description: This script contains the code to study the coordinated behavior of users in the Twitter network

# import the modules to work with the database

# import the module to work with the database (Mongo)
import pymongo
# import the module to work with dataframes
import pandas as pd
# import the module to work with datetime objects
import datetime
# for nice logs
import logging
# for network analysis
#import networkit as nk
import os
import networkx as nx

# for the context analysis
from openai import OpenAI


import tiktoken # for token length control
import numpy as np # for the cosine similarity and matrix computations
import torch # for GPU support

def connect_to_mongo(host:str="localhost", port:int=27017):
    """
    Connect to a MongoDB database and return the client.
    
    Args:
        host (str): The host name.
        port (int): The port number.
        username (str): The username.
        password (str): The password.
        db_name (str): The database name.
        collection_name (str): The collection name.
    """
    # create the client
    client = pymongo.MongoClient(host, port)
    # verify proper connection
    print("Connected to server running MongoDB {} using pymongo {}. Available databases: \n - {}".format(client.server_info()['version'], 
                                                                                                        pymongo.version, str.join("\n - ", 
                                                                                                        client.list_database_names())))
    return client


# Queries to obtain the retweets
OR_QUERY = {"referenced_tweets": {  "$exists": False}} # original tweets
QT_QUERY = {"referenced_tweets": {  "$exists": True},  # quoted tweets
            "referenced_tweets": {  "$elemMatch": {"type": "quoted"}}}
RP_QUERY = {"referenced_tweets": {  "$exists": True},  # replied tweets
            "in_reply_to_user" : {  "$exists": True},
            "referenced_tweets": {  "$elemMatch": {"type": "replied_to"}}}
RT_QUERY =  {   "referenced_tweets": {  "$exists": True}, # retweets
                "referenced_tweets": {  "$elemMatch": {"type": "retweeted"}}}
DEFAULT_TIME_WINDOW = datetime.timedelta(seconds=60) # 1 minute, the time window to consider as "rapid"
DEFAULT_MIN_WEIGHT = 1 # the minimum weight of an edge to be considered in the network
DEFAULT_SIMILARITY_THRESHOLD = 0.8 # the similarity threshold to consider two tweets as similar (cosine similarity)

# Given a time window ∆t, we consider a rapid retweet as a retweet that is performed
# within ∆t seconds of the original tweet. The rapid-retweet network is then defined
# as a weighted network where the nodes represent users and the weight of an edge
# indicates the number of rapid retweets between users

# we build a function to study the sensitivity of the results to the time window (only for analysis purposes)
# the query is pre-defined or passed as an argument
# the time window is passed as an argument
# the function does a single pass over the collection and returns a dataframe.
# the dataframe contains the following columns:
#   - author_id: the id of the user who retweeted, found in the nested "author:id" field
#   - retweeted_author_id: the id of the user who was retweeted, the message is found in the list of "referenced_tweets" where "type" = "retweeted" and the id is in the nested "author:id" field
#   - retweeted_at: the timestamp of the retweet action, found in the nested "created_at" field
#   - created_at: the timestamp of the original tweet, found in the nested "referenced_tweets" where "type" = "retweeted" and the timestamp is in the nested "created_at" field

def rapid_retweet_dataframe(collection, query:dict=None):
    """
    Create a DataFrame suited for the generation of a rapid retweet network from a MongoDB collection.

    Args:
        collection (pymongo.collection.Collection): The MongoDB collection.
        query (dict, optional): the query to filter the tweets, defaults to RT_QUERY.

    Returns:
        pd.DataFrame: A DataFrame containing the retweet data (tweet_id, author_id, retweeted_author_id, retweeted_at, created_at)

    """
    if query is None:
        query = RT_QUERY
    # initialize the variables holding the data
    author_id = []
    retweeted_author_id = []
    retweeted_at = []
    retweeted_originals = []
    created_at = []
    tweet_id = []
    # iterate over the collection
    for tweet in collection.find(query):
        # get the retweeted tweep, by finding the first element in the list of referenced tweets where the type is "retweeted"
        retweeted_tweet = next((x for x in tweet["referenced_tweets"] if x["type"] == "retweeted"), None)
        # if the retweeted tweet is not None, then we can add the data to the lists
        if retweeted_tweet is not None:
            tweet_id.append(tweet["id"])
            author_id.append(tweet["author_id"])
            retweeted_author_id.append(retweeted_tweet["author_id"])
            retweeted_at.append(tweet["created_at"])
            retweeted_originals.append(retweeted_tweet["id"])
            # retweeted_tweet["created_at"] is a string, we need to convert it to a datetime object
            created_at.append(datetime.datetime.strptime(retweeted_tweet["created_at"], "%Y-%m-%dT%H:%M:%S.%fZ"))
        # if the retweeted tweet is None, then we log the error
        else:
            logging.error("No retweeted tweet found for tweet %s", tweet["id"])
    
    # create the dataframe
    df = pd.DataFrame({ "tweet_id": tweet_id,
                        "author_id": author_id, 
                        "retweeted_author_id": retweeted_author_id, 
                        "retweeted_at": retweeted_at, 
                        "created_at": created_at,
                        "retweeted_originals": retweeted_originals})
    # compute the time difference between the retweet and the original tweet
    df["time_diff"] = df["retweeted_at"] - df["created_at"]
    # add  the time difference in seconds and convert it to an integer
    df["time_diff_seconds"] = df["time_diff"].dt.total_seconds().astype(int)

    return df

def extract_rapid_retweet_network_elements(df, time_window:datetime.timedelta=None):
    """
    Extract a rapid retweet network elements (nodes, edges, density) from a DataFrame given a time_window.

    Args:
        df (pd.DataFrame): The DataFrame containing the retweet data (tweet_id, author_id, retweeted_author_id, retweeted_at, created_at).
        time_window (datetime.timedelta, optional): The time window to consider a retweet as rapid. Defaults to datetime.timedelta(seconds=60).

    Returns:
        N, E, d, p (Tuple): the number of nodes involved, the number of edges, the network density (E/N(N-1) and percentage of total retweets that are rapid retweets.
    """
    if time_window is None:
        datetime.timedelta(seconds=60)

    # filter the dataframe to keep only the rapid retweets, without modifying the original dataframe
    df_rapid = df[df["time_diff_seconds"] <= time_window.total_seconds()]

    # count the number the number of times each pair of 'author_id' and 'retweeted_author_id' appears together in df_rapid
    aggregated_df = df_rapid.groupby(['author_id', 'retweeted_author_id']).size().reset_index(name='count')

    # count number of nodes (set of unique author_id and retweeted_author_id)
    N = len(set(aggregated_df["author_id"]).union(set(aggregated_df["retweeted_author_id"])))
    # count number of edges
    E = len(aggregated_df)
    # compute the density
    d = 2* E / (N*(N-1))
    # compute the percentage of rapid retweets
    p = len(df_rapid) / len(df)
    
    return N, E, d, p

def rapid_retweet_network_data( tweet_collection, user_collection,
                                query:dict=None, 
                                time_window:datetime.timedelta=None,
                                min_weight:int=None,
                                write_to_file:bool=False,
                                file_name:str=None):
    """
    Create a rapid retweet network data from a MongoDB collection.

    Args:
        tweet_collection (pymongo.collection.Collection): The MongoDB collection containing the tweets.
        user_collection (pymongo.collection.Collection): The MongoDB collection containing the users and their metadata.
        query (dict, optional): the query to filter the tweet_collection, defaults to RT_QUERY.
        time_window (datetime.timedelta, optional): The time window to consider a retweet as rapid. Defaults to datetime.timedelta(seconds=60).
        min_weight (int, optional): The minimum weight of an edge to be included in the network. Defaults to 1.
        write_to_file (bool, optional): Whether to write the network to file. Defaults to False.
        file_name (str, optional): The name of the file to write the network to. Defaults to 'rapid_retweet_network'.

    Returns:
        E (list) The rapid retweet network edgelist (weighted and directed).
        df (pd.DataFrame): The DataFrame containing the user metadata.

    Notes:
        In the current implementation, we start from the rapid_retweet_dataframe function, then filter and then we add the user metadata.
    """
    if query is None:
        query = RT_QUERY
    if time_window is None:
        time_window = DEFAULT_TIME_WINDOW
    if min_weight is None:
        min_weight = DEFAULT_MIN_WEIGHT
    if file_name is None:
        file_name = "rapid_retweet_network"
    

    # create the raw dataframe
    df = rapid_retweet_dataframe(tweet_collection, query=query)
    # filter the dataframe to keep only the rapid retweets
    df_rapid = df[df["time_diff_seconds"] <= time_window.total_seconds()]
    # count the number the number of times each pair of 'author_id' and 'retweeted_author_id' appears together in df_rapid
    aggregated_df = df_rapid.groupby(['author_id', 'retweeted_author_id']).size().reset_index(name='count')
    # filter the dataframe to keep only the edges with a weight greater than min_weight => edgelist
    aggregated_df = aggregated_df[aggregated_df["count"] >= min_weight]
    # the column `author_id` is the user retweeting,
    # the column `retweeted_author_id` is the user being retweeted,
    # the column `count` is the weight of the edge
    # in terms of information propagation:
    # the edge (author_id, retweeted_author_id, w) means that the user author_id is retweeting the user retweeted_author_id w times
    # we consider the edges to be directed and the retweeted_author_id is the source of the information
    # weras the author_id is the target of the information

    # obtain the set of nodes
    nodes = set(aggregated_df["author_id"]).union(set(aggregated_df["retweeted_author_id"])) # set of strings
    # create a mapping between the nodes and their index (for the edgelist)
    nodemap = {node: i for i, node in enumerate(nodes)} # dict of strings to ints
    # create a mapping between the index and the nodes (for the user metadata)
    revnodemap = {i: node for i, node in enumerate(nodes)}

    # generate dataframe from list of dicts
    node_meta_df =[node_meta_helper(user_collection, i, revnodemap) for i in range(len(nodes))]
    # drop None values
    node_meta_df = [x for x in node_meta_df if x is not None]
    # convert to a DataFrame
    node_meta_df = pd.DataFrame(node_meta_df)

    # transform the aggregated_df into a list of tuples (target, source , weight)
    edges = [(nodemap[source], nodemap[target], weight) for target, source, weight in aggregated_df.values]
    # transform the list of tuples into a DataFrame
    edges = pd.DataFrame(edges, columns=["source", "target", "weight"])

    if write_to_file:
        write_out_node_edgelist(node_meta_df, edges, file_name, add_DTG=True)

    return node_meta_df, edges, df_rapid # THIS HAS BEEN MODIFIED FOR THE SEMANTIC ANALYSIS

def node_meta_helper(user_collection, i, revnodemap):
    """
    Helper function to extract the user metadata from a MongoDB collection.
    
    Args:
        user_collection (pymongo.collection.Collection): The MongoDB collection containing the users and their metadata.
        i (int): The index of the node.
        revnodemap (dict): The mapping between the index and the nodes.

    Returns:
        node_meta_info (dict): The user metadata.

    """
    node_meta_info = user_collection.find_one({"_id": {"$eq": revnodemap[i]}}, 
                                              {"_id":True, "username":True, "name":True, "status":True, "verified":True,
                                               "features.P": True, "botometer.raw_scores.english.overall": True,  "botometer.raw_scores.universal.overall": True})
    if node_meta_info is not None:
        # add the index to the node metadata
        node_meta_info["user_id"] = node_meta_info.pop("_id")
        node_meta_info["node_id"] = i
        
    else:
        logging.warning(f"User {revnodemap[i]} not found in the user collection.")
        node_meta_info = {"user_id": revnodemap[i], "node_id": i, "username": None, "name": None, "status": None}

    return node_meta_info

def write_out_node_edgelist(node_meta_df, edges, filename, id_column="node_id", label_column="username", add_DTG=False):
    """
    Write out the node and edgelist to file.

    Args:
        node_meta_df (pd.DataFrame): The DataFrame containing the user metadata.
        edges (pd.DataFrame): The DataFrame containing the edgelist.
        filename (str): The name of the file to write the network to.
        id_column (str): the name of the column in the dataframe holding the graph node ids
        label_column (str): the name of the column that will be used for the default Label attribute in Gephi
        add_DTG (bool): if True, the current date and time will be appended to the filename
    """
    if add_DTG:
        globalpath = os.path.split(filename)
        filename = os.path.join(*globalpath[:-1], "{}{}".format(datetime.datetime.now().strftime("%Y-%m-%d_"), globalpath[-1]))
    if filename.endswith(".csv"):
        # strip csv if already persent in name
        filename = filename[:-4]

    # write out the nodes
    # update the node id
    node_meta_df["Id"] = node_meta_df[id_column]
    # add the default label column
    node_meta_df["Label"] = node_meta_df[label_column]
    # write out 
    node_meta_df.to_csv("{}_nodes.csv".format(filename), index=False, sep=";")
    
    # write out the edges
    edges.to_csv(f"{filename}_edgelist.csv", index=False, sep=";")
    
    logging.info("Network written to file.")
    
    return



## NLP
client = OpenAI(
    # This is the default and can be omitted
    api_key=open('./secrets/OpenAI_API_key', 'r').read().strip(),
)

MODEL = "gpt-4"
SYSTEMPROMPT = """You are a world renowned psychologist who analyzes the stance of a person on the topic. You do this by looking at their social media messages 
to assess their attitude, beliefs, and possible emotional factors influencing their perspective on the issue."""
SYSTEMCONTENT = """
Jürgen Conings was a corporal in the Belgian Air Component and shooting instructor. He was a former elite soldier and experienced sniper. 
Over the course of his career, he took part in eleven foreign missions in Yugoslavia, Bosnia, Kosovo, Lebanon, Iraq, and Afghanistan.
Several colleagues of Conings declared that he held far-right ideas, and had threatened Marc Van Ranst, a Belgian virologist.
Until the end of 2020, Conings was a member of the right-wing populist party Vlaams Belang.
Because of his politically left-leaning ideas and presence on social media, Van Ranst is often targeted by right-wing and COVID-19 sceptics alike. 
Conings received two disciplinary sanctions because of his threats to Van Ranst in 2020. Therefore, he was demoted to weapon bearer, which secured 
him access to the armory. After his threats, his name was added to the CUTA list of the Coordination Unit for Threat Analysis, 
which features Muslim extremists and far-right and far-left individuals.

Prime Minister Alexander De Croo called it unacceptable that someone considered dangerous has access to weapons. Minister of Defence Ludivine Dedonder 
emphasised the need for measures. She was criticised by the opposition since she was already questioned about right extremism in the army one month before.
During a press conference on 25 May, Dedonder admitted together with Chief of Defence Michel Hofman that mistakes were made in the case of Conings and that 
they were being examined.
On 16 June, the Chamber of Representatives discussed the report by Dedonder regarding the Conings case. While the report was leaked before, 
it was initially not handed over to the members of parliament. The meeting was therefore postponed until 5 PM, when members had the opportunity to 
access the report. Dedonder commented on the report during the meeting, stating that there was "a structural understaffing at different services 
on all levels, a significant staff turnover, loss of knowledge and experience, limited supervision at Conings' final unit, Coronavirus measures, 
insufficient circulation of information, a new and complex structure and internal staffing problems at the military intelligence service, 
insufficient exchange of information within defence and between the different security services." Moreover, she identified a lack of a clear 
policy with regard to extremism within the army.

Conings could count on the support of several far-right people and organisations. Far-right terrorist Tomas Boutens knew Conings and expressed his support.
A Facebook group called Als 1 achter Jürgen (As 1 behind Jürgen) was started on 20 May as a support group for Conings, and gained nearly 11,500 members 
in its first 24 hours. After gaining almost 45,800 members, Facebook removed the group on 25 May, specifying that "pages which praise or support terrorists, 
like Jürgen Conings, are not allowed on Facebook or Instagram."

In Maasmechelen, close to Hoge Kempen national park, a "March for Jürgen" was held, attracting about 150 participants. One participant was filmed doing the 
Nazi salute. On 23 May, about 200 people participated in a second "March for Jürgen". On 24 May, when the national park was reopened, a walk was organised 
in the area of heathland known as Mechelse Heide, part of the national park, in support of Conings, which attracted 100 participants.
On the evening of 20 June, the day of the discovery of Conings' body, about 140 sympathisers held a vigil for Conings. 
The manhunt for Conings has given rise to several unproven and debunked conspiracy theories among his supporters, 
including the false belief that Conings was killed by the government.
"""

FOLLOW_UP_SYSTEMPROMPT = """You are a world renowned pshychologist who analyzes the stance of a person on the topic. 
You do this by looking at their social media messages to assess their attitude, beliefs, and possible emotional factors 
influencing their perspective on the issue.

You have been analyzing the stance of a person multiple times over the past few days. You combine the information from your previous analysis with the new information you have gathered today.
Each analysis start with `analysis x:`, where `x`denotes the number of the analysis. The actual analysis follows on the following line.

You combine these elements into a single analysis that you use to generate a profile of the person.
"""

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def generate_user_profile(datacol, author_id: str, model: str = "gpt-4", splitlimit:int=8000) -> str:
    """
    Generates a psychological profile of a user based on their posts.

    Args:
        datacol (pymongo.collection.Collection): The MongoDB collection containing the data
        author_id (str): The author_id of the user to generate a profile for
        model (str, optional): The model to use for the analysis. Defaults to "gpt-4".
        splitlimit (int, optional): The maximum number of tokens to use for each analysis. Defaults to 8k.

    Returns:
        str: The generated profile
    """
    print("Generating user profile using {}".format(model))
    # fetch all message from the user
    msgs = list(map(lambda x: x['text'], datacol.find({"author_id": {"$eq": author_id}}, {"_id": 0, "text":1})))
    # if no messages are found, return an empty string
    if len(msgs) == 0:
        return ""
    # combine the messages into a single string in such a way that the `num_tokens_from_string` is below 7k
    # if its above 7k, we split the messages into multiple lists until we have a list of lists where
    # each sublist has a length below 7k
    if num_tokens_from_string(" ".join(msgs), "gpt2") > splitlimit:
        logging.info("Splitting messages into multiple lists")
        # split the messages into two lists until we have a list of lists where
        # each sublist has a length below 7k
        msgs = [msgs[:len(msgs)//2], msgs[len(msgs)//2:]]
        while any(map(lambda x : num_tokens_from_string(" ".join(x), "gpt2") > splitlimit, msgs)):  
            logging.info("Splitting messages into multiple sublists")
            msgs = [msgs[0][:len(msgs[0])//2], msgs[0][len(msgs[0])//2:], msgs[1][:len(msgs[1])//2], msgs[1][len(msgs[1])//2:]]
    
    # generate a profile for each list of messages
    profiles = []
    # if we have a list of lists, we generate a profile for each sublist
    if isinstance(msgs[0], list):
        for msg in msgs:
            # analysis = openai.ChatCompletion.create(model="gpt-4",
            #                                         messages=[  { "role": "system", "content": SYSTEMPROMPT},
            #                                                     {"role": "assistant", "content": SYSTEMCONTENT},
            #                                                     {"role": "user", "content": "\n".join(msg)}
            #                                                     ]
            #                                         )
            # profiles.append(analysis['choices'][0]['message']["content"])
            analysis = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEMPROMPT},
                {"role": "assistant", "content": SYSTEMCONTENT},
                {"role": "user", "content": "\n".join(msg)}
                ]
            )

            # Extracting the assistant's message content
            #profiles.append(analysis.choices[0].message["content"])
            profiles.append(analysis.choices[0].message)

    # if we have a single list, we generate a profile for the whole list
    else:
        # analysis = openai.ChatCompletion.create(model="gpt-4",
        #                                         messages=[  { "role": "system", "content": SYSTEMPROMPT},
        #                                                     {"role": "assistant", "content": SYSTEMCONTENT},
        #                                                     {"role": "user", "content": "\n".join(msgs)}
        #                                                     ]
        #                                         )
        # profiles.append(analysis['choices'][0]['message']["content"])
        analysis = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEMPROMPT},
                {"role": "assistant", "content": SYSTEMCONTENT},
                {"role": "user", "content": "\n".join(msgs)}
            ]
        )

        # Extracting the assistant's message content
        profiles.append(analysis.choices[0].message)

    # combine the sub-profiles into a single profile
    global_analysis = "\n\n\n".join([f"analysis {i+1}: \n{response}" for i, response in enumerate(profiles)])

    # generate a follow-up analysis
    combined_response = client.chat.completions.create(model=model,
                                                    messages=[
                                                        {"role": "system", "content": FOLLOW_UP_SYSTEMPROMPT},
                                                        {"role": "user", "content": global_analysis}
                                                    ])
    return combined_response.choices[0].message


## Semantic-network analysis (the in-memory version)
## Deploying this at scale will require a redesign (e.g. using a vector database)

# we need to load the message that are not retweets in memory.
# for each message we will load the following fields:
# - author_id
# - text
# - created_at
# - all-mpnet-base-v2 embeddings

# we will also load the following fields for each user:
# - author_id
# - name
# - screen_name
# - status

# projection settings
PROJECTEDFIELDS = {"id":1, "author.id":1, "created_at":1, "_id":0}
DEFAULT_THETA = 0.8

#PROJECTEDFIELDS = {"id":1, "author.id":1, "text_en": 1, "created_at":1, "_id":0, "all-mpnet-base-v2":1}

# data loading
def find_time_range_indices(df, start_idx:int, dt:datetime.timedelta=DEFAULT_TIME_WINDOW):
    """
    Find the indices of the DataFrame that correspond to the time range
    
    Note: this requires the DataFrame to be sorted by the created_at column, so we can use binary search 
    techniques to quickly find the end index of the time range.

    Args:
        df (pd.DataFrame): The DataFrame to search in
        start_idx (int): The index of the row to start the search from
        dt (datetime.timedelta, optional): The time window to search for. Defaults to DEFAULT_TIME_WINDOW.

    Returns:
        np.ndarray: The indices of the rows in the DataFrame that are within the time window dt of the row at index start_idx
    """
    start_time = df['created_at'].iloc[start_idx]
    stop_time = start_time + dt
    end_idx = df['created_at'].searchsorted(stop_time, side='right')

    return np.arange(start_idx + 1, end_idx)

def cosine_similar_at_time(df, vector_matrix:torch.Tensor, i:int, theta=0., dt:datetime.timedelta=DEFAULT_TIME_WINDOW):
    """
    Given a dataframe, a vector matrix, an index i, a threshold theta and a time window dt,
    find the indices of the rows in the dataframe that are within the time window dt of the row at index i
    and have a cosine similarity larger than theta.

    This can be used to construct a rapid-semantic network between users.

    Args:
        df (pd.DataFrame): The DataFrame to search in
        vector_matrix (torch.Tensor): The vector matrix (on GPU holdings the embeddings, same size as the DataFrame, built after the sorting)
        i (int): The index of the row to start the search from
        theta (float, optional): The cosine similarity threshold. Defaults to 0.
        dt (datetime.timedelta, optional): The time window to search for. Defaults to DEFAULT_TIME_WINDOW.

    Returns:
        np.ndarray: The indices of the rows in the DataFrame that are within the time window dt of the row at index start_idx and that 
        have a cosine similarity larger than theta.
    """
    time_range_indices = find_time_range_indices(df, i, dt)
    
    if time_range_indices.size > 0:
        # Get the vector at index i
        vector_i = vector_matrix[i].unsqueeze(0)
        # Get all the vectors in the specified index range
        vectors_in_range = vector_matrix[time_range_indices]  
        
        # Calculate cosine similarity between vector_i and vectors_in_range
        cosine_similarities = torch.mm(vector_i, vectors_in_range.t()).squeeze()
        
        # Find the indices of cosine similarities larger than the specified threshold theta
        #above_treshold =  cosine_similarities >= theta
        #time_range_indices[]
        # convert the above_treshold tensor to a numpy array on cpu and flatten it
        #above_treshold = above_treshold.cpu().numpy().flatten()


        return time_range_indices[(cosine_similarities >= theta).cpu().numpy().flatten()]
    else:
        return np.array([]) # return an empty array if there are no indices in the time range (type consistency)

def rapid_semantic_dataframe(collection, queries=[OR_QUERY, RP_QUERY, QT_QUERY], 
                            limit=None, drop_text=True, drop_embeddings=True, dtype=None, device=None, 
                            text_column='text_en', embedding_column='all-mpnet-base-v2'):
    """
    Construct a semantic network dataframe from the data in the collection holding the tweets.

    Args:
        collection (pymongo.collection.Collection): The collection to query
        queries (list, optional): The queries to use. Defaults to [OR_QUERY, RP_QUERY, QT_QUERY].
        limit (int, optional): The maximum number of results to return. Defaults to 0 (i.e. no limit).
        drop_text (bool, optional): Whether to drop the text_en column from the dataframe. Defaults to True.
        drop_embeddings (bool, optional): Whether to drop the all-mpnet-base-v2 column from the dataframe. Defaults to True.
        dtype (torch.dtype, optional): The data type of the tensor. Defaults to torch.float32.
        device (torch.device, optional): The device to use. Defaults to torch.device('cuda' if torch.cuda.is_available() else 'cpu').
        embedding_column (str, optional): The name of the column that holds the embeddings. Defaults to 'all-mpnet-base-v2'.

    Returns:
        pd.DataFrame: A dataframe with the following columns:
            - created_at (datetime): the date of the message
            - id (str): the id of the message
            - author (str): the author of the message 
        torch.Tensor: A vector matrix (on GPU) holding the embeddings of the messages
    """
    if limit is None:
        limit = 0
    if dtype is None:
        dtype = torch.float32
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
    # Create a local copy of PROJECTEDFIELDS and update the copy to include the text_en and all-mpnet-base-v2 fields
    local_projected_fields = PROJECTEDFIELDS.copy()
    local_projected_fields.update({text_column: 1, embedding_column: 1})

    # generate cursor for each query (limiting the number of results if required)
    cursor = [collection.find(q, local_projected_fields).limit(0) for q in queries]
    # generate a list of dataframes for each query
    df = [pd.DataFrame(list(c)) for c in cursor]
    # concatenate the dataframes
    df = pd.concat(df, ignore_index=True)
    # sort the dataframe by date (ascending)
    df.sort_values(by="created_at", inplace=True)
    # flatten the author.id field
    df["author"] = df["author"].apply(lambda x: x["id"])
    # drop the rows with missing values
    df.dropna(inplace=True)
    # drop duplicate message (if any) from the id column
    df.drop_duplicates(subset="id", inplace=True)
    # transform the all-mpnet-base-v2 field into a numpy array
    df[embedding_column] = df[embedding_column].apply(np.array)

    # generate the vector matrix
    vector_matrix = np.vstack(df[embedding_column].values)
    # convert the vector matrix to a tensor on the specified device
    vector_matrix_gpu = torch.tensor(vector_matrix, dtype=torch.float32, device=device)
    # normalize the vector matrix
    vector_matrix_gpu = vector_matrix_gpu / vector_matrix_gpu.norm(dim=1, keepdim=True)
    
    if drop_text:
        df.drop(columns=text_column, inplace=True)
    if drop_embeddings:
        df.drop(columns=embedding_column, inplace=True)
    # reset the index
    df.reset_index(drop=True, inplace=True)
    
    return df, vector_matrix_gpu

def rapid_semantic_network_data(tweet_collection, user_collection, queries=[OR_QUERY, RP_QUERY, QT_QUERY], 
                                limit=None, drop_text=True, drop_embeddings=True, dtype=None, device=None, 
                                text_column='text_en', embedding_column='all-mpnet-base-v2', 
                                theta=None, time_window=None, min_weight:int=1, allow_self_loops:bool=False,
                                write_to_file:bool=False, file_name:str=None, add_DTG:bool=True):
    """
    Create a rapid semantic network data from a MongoDB collection.

    Args:
        tweet_collection (pymongo.collection.Collection): The collection holding the tweets.
        user_collection (pymongo.collection.Collection): The collection holding the users.
        queries (list, optional): The queries to use. Defaults to [OR_QUERY, RP_QUERY, QT_QUERY].
        limit (int, optional): The maximum number of results to return. Defaults to 0 (i.e. no limit).
        drop_text (bool, optional): Whether to drop the text_en column from the dataframe. Defaults to True.
        drop_embeddings (bool, optional): Whether to drop the all-mpnet-base-v2 column from the dataframe. Defaults to True.
        dtype (torch.dtype, optional): The data type of the tensor. Defaults to torch.float32.
        device (torch.device, optional): The device to use. Defaults to torch.device('cuda' if torch.cuda.is_available() else 'cpu').
        text_column (str, optional): The name of the column that holds the text. Defaults to 'text_en'.
        embedding_column (str, optional): The name of the column that holds the embeddings. Defaults to 'all-mpnet-base-v2'.
        theta (float, optional): The threshold for the cosine similarity. Defaults to 0.8.
        time_window (datetime.timedelta, optional): The time window to consider a semantic tweet as rapid. Defaults to datetime.timedelta(seconds=60).
        min_weight (int, optional): The minimum weight of an edge. Defaults to 1.
        allow_self_loops (bool, optional): Whether to allow self-loops. Defaults to False.
        write_to_file (bool, optional): Whether to write the data to file. Defaults to False.
        file_name (str, optional): The name of the file to write the data to. Defaults to 'rapid_semantic_network'.
        add_DTG (bool, optional): Whether to add the current date and time to the filename. Defaults to True.

    Returns:
        node_meta_df (pd.DataFrame): The DataFrame containing the user metadata.
        semantic_network_df (pd.DataFrame): The DataFrame containing the semantic network data (weighted edgelist).

    Notes:
        In the current implementation, we start from the rapid_retweet_dataframe function, then filter and then we add the user metadata.
    """
    if limit is None:
        limit = 0
    if dtype is None:
        dtype = torch.float32
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if time_window is None:
        time_window = DEFAULT_TIME_WINDOW
    if theta is None:
        theta = DEFAULT_THETA
    
    # obtain the rapid semantic dataframe
    df, vector_matrix_gpu = rapid_semantic_dataframe(tweet_collection, queries=queries, limit=limit, drop_text=drop_text, 
                                                    drop_embeddings=drop_embeddings, dtype=dtype, device=device, 
                                                    text_column=text_column, embedding_column=embedding_column)
    # compute the rapid cosine similarity indices
    df["cosine_similar_index"] = [cosine_similar_at_time(df, vector_matrix_gpu, i, dt=time_window, theta=theta) for i in range(len(df))]
    # obtain the rows where we have rapid cosine similarity indices
    simidx = df["cosine_similar_index"].apply(lambda x: len(x) > 0)
    # generate the edgelist
    edges = []

    # Iterate through the input dataframe
    for _, row in df[simidx].iterrows():
        source = row["author"]
        target_indices = row["cosine_similar_index"]

        # Iterate through the target indices
        for target_index in target_indices:
            # Get the target author
            target = df['author'].iloc[target_index]
            # Add the source-target pair to the list
            edges.append((source, target))

    # Convert the list of source-target pairs to a dataframe
    semantic_network_df = pd.DataFrame(edges, columns=['source', 'target'])
    # Count the number of times each source-target pair appears in the dataframe
    semantic_network_df = semantic_network_df.groupby(['source', 'target']).size().reset_index(name='weight')
    # filter the dataframe to keep only the edges with a weight greater than min_weight => edgelist
    semantic_network_df = semantic_network_df[semantic_network_df["weight"] >= min_weight]
    # drop the self-loops
    if not allow_self_loops:
        semantic_network_df = semantic_network_df[semantic_network_df["source"] != semantic_network_df["target"]]

    # Get unique authors from source and target columns
    nodes = set(semantic_network_df['source']) | set(semantic_network_df['target'])
    # create a mapping between the nodes and their index (for the edgelist)
    nodemap = {node: i for i, node in enumerate(nodes)} # dict of strings to ints
    # create a mapping between the index and the nodes (for the user metadata)
    revnodemap = {i: node for i, node in enumerate(nodes)}

    # generate dataframe from list of dicts
    node_meta_df =[node_meta_helper(user_collection, i, revnodemap) for i in range(len(nodes))]
    # drop None values
    node_meta_df = [x for x in node_meta_df if x is not None]
    # convert to a DataFrame
    node_meta_df = pd.DataFrame(node_meta_df)

    # generate dataframe from list of dicts
    node_meta_df = [node_meta_helper(user_collection, i, revnodemap) for i in range(len(nodes))]
    # drop None values
    node_meta_df = [x for x in node_meta_df if x is not None]
    # convert to a DataFrame
    node_meta_df = pd.DataFrame(node_meta_df)

    # transform the original edges (str) into their index (int)
    semantic_network_df["source"] = semantic_network_df["source"].apply(lambda x: nodemap[x])
    semantic_network_df["target"] = semantic_network_df["target"].apply(lambda x: nodemap[x])
    
    # write to file (if required)
    if write_to_file:
        if file_name is None:
            file_name = "rapid_semantic_network"
        write_out_node_edgelist(node_meta_df, semantic_network_df, file_name, add_DTG=add_DTG)

    return node_meta_df, semantic_network_df


# helper function for the parameter study plots
def extract_rapid_semantic_network_elements(df, vector_matrix_gpu, 
                                            time_window:datetime.timedelta=None, theta:float=None,
                                            allow_self_loops:bool=False, min_weight:int=1):
    """
    Extract a rapid retweet network elements (nodes, edges, density) from a DataFrame given a time_window.

    Args:
        df (pd.DataFrame): The rapid semantic dataframe (created_at, tweet_id, author_id).
        vector_matrix_gpu (torch.Tensor): The torch.Tensor containing the embeddings of the tweets.
        time_window (datetime.timedelta, optional): The time window to consider a retweet as rapid. Defaults to datetime.timedelta(seconds=60).
        theta (float, optional): The cosine similarity threshold to consider a retweet as rapid. Defaults to 0.8.

    Returns:
        N, E, d, p (Tuple): the number of nodes involved, the number of edges, the network density (E/N(N-1) and percentage of messages that are rapid semantic tweets.
    """
    if time_window is None:
        time_window = datetime.timedelta(seconds=60)
    if theta is None:
        theta = DEFAULT_THETA

    # compute the rapid cosine similarity indices
    df["cosine_similar_index"] = [cosine_similar_at_time(df, vector_matrix_gpu, i, dt=time_window, theta=theta) for i in range(len(df))]
    # obtain the rows where we have rapid cosine similarity indices
    simidx = df["cosine_similar_index"].apply(lambda x: len(x) > 0)
    # generate the edgelist
    edges = []

    # Iterate through the input dataframe
    for _, row in df[simidx].iterrows():
        source = row["author"]
        target_indices = row["cosine_similar_index"]

        # Iterate through the target indices
        for target_index in target_indices:
            # Get the target author
            target = df['author'].iloc[target_index]
            # Add the source-target pair to the list
            edges.append((source, target))

    # Convert the list of source-target pairs to a dataframe
    semantic_network_df = pd.DataFrame(edges, columns=['source', 'target'])
    # Count the number of times each source-target pair appears in the dataframe
    semantic_network_df = semantic_network_df.groupby(['source', 'target']).size().reset_index(name='weight')
    # filter the dataframe to keep only the edges with a weight greater than min_weight => edgelist
    semantic_network_df = semantic_network_df[semantic_network_df["weight"] >= min_weight]
    # drop the self-loops
    if not allow_self_loops:
        semantic_network_df = semantic_network_df[semantic_network_df["source"] != semantic_network_df["target"]]

    # Get unique authors from source and target columns
    nodes = set(semantic_network_df['source']) | set(semantic_network_df['target'])
    # number of nodes
    N = len(nodes) 
    # number of edges
    E = len(semantic_network_df) 
    # compute the density
    d = 2* E / (N*(N-1))
    # compute the percentage of rapid semantic tweets
    p = len(df[simidx]) / len(df)
    
    
    return N, E, d, p

def calculate_indegree_distribution(RRE):
    """"
    Helper function to compute the indegree distribution for a given time window. 
    
    The indegree represents the intensity of the message amplificaton across multiple users.
    """
    # Generate the actual network
    G = nx.DiGraph()
    
    # Add the edges
    for _, row in RRE.iterrows():
        G.add_edge(row.source, row.target, weight=row.weight)
    
    # Calculate the indegree for each node
    indegree = dict(G.in_degree())
    
    # Calculate the indegree distribution
    indegree_distribution = [list(indegree.values()).count(x) for x in range(max(indegree.values()) + 1)]
    
    # Normalize the distribution
    total_nodes = len(G.nodes)
    indegree_distribution_normalized = np.array(indegree_distribution) / total_nodes
    
    # Get horizontal values
    indegree_values = np.arange(len(indegree_distribution))
    
    return indegree_values, indegree_distribution_normalized

def compute_weight_distribution(RRE):
    """"
    Helper function to compute the weight distribution for a given time window. 
    
    The weight represents the intensity of the message amplificaton in a single user.
    """
    # Get the weight values
    weights = RRE['weight'].values
    
    # Compute the weight distribution
    weight_distribution = np.bincount(weights)
    
    # Normalize the distribution
    total_weights = np.sum(weight_distribution)
    weight_distribution_normalized = weight_distribution / total_weights
    
    # Get horizontal values
    weight_values = np.arange(len(weight_distribution))
    
    return weight_values, weight_distribution_normalized

# store the user profiles in a file
import json

# print a text, in the notebook, but limit the output to 100 charachters per line
def print_wrapped_text(text, max_chars_per_line=100):
    words = text.split()
    current_line = ""

    for word in words:
        if len(current_line + word) + 1 <= max_chars_per_line:
            current_line += word + " "
        else:
            print(current_line.strip())
            current_line = word + " "

    # Print the last line if it's not empty
    if current_line.strip():
        print(current_line.strip())


# embedding models
from sentence_transformers import SentenceTransformer, LoggingHandler

# regex for mentions and urls removal
import re
MENTION_URL_REGEX = re.compile(r"http\S+|@\S+")
SPACE_REGEX = re.compile(r"\xa0")
NEWLINE_REGEX = re.compile(r"\n")
AMPERSAND_REGEX = re.compile(r"&amp;")

# all embedding models (not all are used)
EMBEDDING_MODEL_NAMES_EN = ['sentence-transformers/all-mpnet-base-v2',
                            'sentence-transformers/all-distilroberta-v1']
EMBEDDING_MODEL_NAMES_MULTI = ['sentence-transformers/paraphrase-multilingual-mpnet-base-v2']

# if you have a path holding the models, set it here, if not specified, the models will be downloaded
MODEL_PATHS = '/SSD-data/LLMs' 
# generate dict from comprehension
EMBEDDING_MODELS_EN = {name.split('/')[-1]: SentenceTransformer(name, cache_folder=MODEL_PATHS) for name in EMBEDDING_MODEL_NAMES_EN}
EMBEDDING_MODELS_MULTI = {name.split('/')[-1]: SentenceTransformer(name, cache_folder=MODEL_PATHS) for name in EMBEDDING_MODEL_NAMES_MULTI}

# create progress bar
from tqdm import tqdm
tqdm.pandas()
import time
# obtain proper queries
from coordinated_behavior import OR_QUERY, RP_QUERY, QT_QUERY 
# combination of queries
QUERIES = [OR_QUERY, RP_QUERY, QT_QUERY]        # combined query for non-retweets
EMBEDDING_COMPUTE_FILTER = {"_id":1, "text":1, "id":1}  # filter used for embedding computation
LIMIT = 10                                      # limit for the number of tweets to be embedded for each query

def tweet_preprocessor(text, remove_url=True, remove_mention=True, remove_newline=True, replace_ampersand=True):
    """
    Preprocess the tweet text

    Args:
        text (str): the tweet text
        remove_url (bool, optional): whether to remove urls. Defaults to True.
        remove_mention (bool, optional): whether to remove mentions. Defaults to True.
        remove_newline (bool, optional): whether to remove newlines. Defaults to True.

    Returns:
        str: the preprocessed tweet text
    """
    cleaned = text
    if remove_url:
        cleaned = MENTION_URL_REGEX.sub('', cleaned)
    if remove_mention:
        cleaned = SPACE_REGEX.sub(' ', cleaned)
    if remove_newline:
        cleaned = cleaned.replace('\n', cleaned)
    if replace_ampersand:
        cleaned = AMPERSAND_REGEX.sub('and', cleaned)
    
    return cleaned

def openai_embedding(text, model="text-embedding-ada-002", minlength=5, max_retries=3):
    """
    Compute the embedding of a text using openai api if the text is longer than minlength

    Args:
        text (str): the text to be embedded
        model (str, optional): the model to be used. Defaults to "text-embedding-ada-002".
        minlength (int, optional): the minimum length of the text to be embedded. Defaults to 5.
        max_retries (int, optional): the maximum number of retries in case of a failure. Defaults to 3.

    Returns:
        list: the embedding of the text (1536 dimensions) or a specific error message
    """
    if len(text) < minlength:
        return None
    else:
        for i in range(max_retries):
            try:
                return client.embeddings.create(input = [text], model=model).data[0].embedding
            except Exception as e:
                logging.error(f"Failed to get embeddings from OpenAI for text: {text}. Attempt: {i+1}. Error: {str(e)}")
                if i < max_retries - 1:  # i is zero indexed
                    time.sleep(1)  # wait a bit before trying again
                else:
                    return "Embedding error"

def generate_text_embeddings(collection, queries=None, limit=None, update_collection=True, 
                             clean_text=True, remove_url=True, remove_mention=True,
                             open_models=None, openai_models=[], openai_minlength=5):
    """
    Generate a dataframe of tweets from the database 

    Args:
        collection (pymongo.collection.Collection): the collection to be queried
        queries (list, optional): the queries to be used. Defaults to QUERIES.
        limit (int, optional): the limit of the number of tweets to be queried. Defaults to LIMIT.
        update_collection (bool, optional): whether to update the collection with the embeddings. Defaults to True.
        clean_text (bool, optional): whether to clean the text. Defaults to True.
        remove_url (bool, optional): whether to remove urls. Defaults to True.
        remove_mention (bool, optional): whether to remove mentions. Defaults to True.
        open_models (dict, optional): the open source models to be used. Defaults to EMBEDDING_MODELS_MULTI.
        openai_models (list, optional): the openai models to be used. Defaults to ["text-embedding-ada-002"].
        openai_minlength (int, optional): the minimum length of the text to be embedded using openai. Defaults to 5.

    
    Returns:
        pd.DataFrame: the dataframe of tweets with their embeddings
    """
    if queries is None:
        queries = QUERIES
    if limit is None:
        limit = LIMIT
    if open_models is None:
        open_models = EMBEDDING_MODELS_MULTI
    if len(openai_models) == 0:
        logging.warning("No OpenAI models specified.")

    ## Get the data
    logging.info("Fetching the data")
    # generate cursor for each query (limiting the number of results if required)
    cursor = [collection.find(q, EMBEDDING_COMPUTE_FILTER).limit(limit) for q in queries]
    # generate a list of dataframes for each query
    df = [pd.DataFrame(list(c)) for c in cursor]
    # concatenate the dataframes
    df = pd.concat(df, ignore_index=True)

    ## Drop duplicates
    logging.info("Dropping duplicates")
    df.drop_duplicates(subset=["_id"], inplace=True)

    ## Text preprocessing
    logging.info("Preprocessing the text")
    if clean_text:
        df["clean_text"] = df.progress_apply(lambda x: tweet_preprocessor(x["text"], remove_url=remove_url, remove_mention=remove_mention), axis=1)
    else:
        df["clean_text"] = df["text"]

    model_names = []
    ## Open source embeddings
    for model_name, model in open_models.items():
        logging.info("Computing embeddings for model: {}".format(model_name))
        model_names.append(model_name)
        # compute the embeddings for a particular model
        res = model.encode(df["clean_text"].tolist(), show_progress_bar=True)
        # transform the result array (30, 768) to a list of (768,) arrays
        df[model_name] = [r for r in res]

    ## OpenAI embeddings
    for model_name in openai_models:
        logging.info("Obtaining embeddings for model: {}".format(model_name))
        model_names.append(model_name)
        # compute the embeddings for a particular model
        df[model_name] = df["clean_text"].progress_apply(lambda x: openai_embedding(x, model=model_name, minlength=openai_minlength))
    
    ## Update the collection
    if update_collection:
        logging.info("Updating the collection")
        # update the collection
        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            # get the model names and embeddings
            values_dict = {model_name: row[model_name].tolist() if isinstance(row[model_name], np.ndarray)  else row[model_name] for model_name in model_names}
            # update the document in the database
            collection.update_one({"_id": row["_id"]}, {"$set": values_dict}, upsert=True)


    return df 


