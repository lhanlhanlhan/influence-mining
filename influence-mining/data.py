import copy
import csv
import json
import matplotlib.pyplot as plt
import networkx as nx
import random

class Data:
    def __init__(self):
        pass


def read_influence(path="../data"):
    names = {}
    connections = {}
    artist_genre = {}
    with open("{}/influence_data.csv".format(path), 'r', encoding='utf-8') as file_handler:
        influence_file = csv.reader(file_handler)
        next(influence_file)
        for influence_entry in influence_file:
            influencer_id = int(influence_entry[0])
            influencer_name = influence_entry[1]
            influencer_type = influence_entry[2]
            influencer_active_year = int(influence_entry[3])
            follower_id = int(influence_entry[4])
            follower_name = influence_entry[5]
            follower_type = influence_entry[6]
            follower_active_year = int(influence_entry[7])

            if influencer_id not in connections:
                connections[influencer_id] = []
            names[influencer_id] = influencer_name
            names[follower_id] = follower_name

            connections[influencer_id].append(follower_id)
            artist_genre[influencer_id] = influencer_type
            artist_genre[follower_id] = follower_type
    return connections, artist_genre, names


def data_preprocess(path="../data"):
    """Preliminary process of files"""
    print("Prepossessing dataset from '{}' ...".format(path))

    """Read: Music file"""
    with open("{}/full_music_data.csv".format(path), 'r', encoding='utf-8') as file_handler:
        music_file = csv.reader(file_handler)
        next(music_file)
        for music_entry in music_file:
            artist_list = list(map(int, json.loads(music_entry[1])))
            music_features = map(float, music_entry[2:-1])
            # music_features = map(float, music_entry[2:7])
            year = int(music_entry[16])
            date = music_entry[17]
            title = music_entry[18]
            # print(artist_list, year, date, title)
            # print(music_features)

    """Read: Influence file"""
    connections, artist_genre, _ = read_influence(path)

    """Read: Artist data file"""
    names = {}
    artists = {}
    with open("{}/data_by_artist.csv".format(path), 'r', encoding='utf-8') as file_handler:
        artist_file = csv.reader(file_handler)
        next(artist_file)
        for artist_entry in artist_file:
            artist_name = artist_entry[0]
            artist_id = int(artist_entry[1])
            feature_a = artist_entry[2:-1]
            artist_features = feature_a
            # artist_features = artist_entry[2:7]

            if artist_id not in artists:
                artists[artist_id] = artist_features
                names[artist_id] = artist_name

    """写出每一个 artist 的跟随者人数"""
    followers_count = {}
    for src_artist_id in artists:
        if src_artist_id not in connections:
            continue
        des_artist_ids = connections[src_artist_id]
        followers_count[src_artist_id] = len(des_artist_ids)
        # for des_artist_id in des_artist_ids:
        #     if des_artist_id not in followers_count:
        #         followers_count[des_artist_id] = 0
        #     followers_count[des_artist_id] += 1
    followers_count = sorted(followers_count.items(),
                             key=lambda x: x[1],
                             reverse=True)
    """Write: Follower Files"""
    with open("{}/artists.follow.csv".format(path), 'w') as file:
        for inf_id, count in followers_count:
            file.write("\"{}\",{}\n".format(names[inf_id], count))
    followers_count = [inf_id for inf_id, count in followers_count]

    targets = set(followers_count[:int(len(followers_count) * 0.5)])
    # print(followers_count)

    """Write: Artist Profile file"""
    with open("{}/artists.prof".format(path), 'w') as file:
        for artist_id in artists:
            if artist_id not in targets:
                continue
            """暂时将 artist_genre 映射到给定的数据中"""
            genre = 'Not_known'
            if artist_id in artist_genre:
                genre = artist_genre[artist_id].replace(' ', '_')
            """这个是每个 artist 的特征"""
            artist_features_str = '\t'.join(artists[artist_id])
            file.write("{}\t{}\t{}\n".format(artist_id,
                                             artist_features_str,
                                             genre))

    """Write: Connection file"""
    with open("{}/artists.conn".format(path), 'w') as file:
        for src in connections:
            if src not in artists:
                continue
            if src not in targets:
                continue
            for dest in connections[src]:
                if dest not in artists:
                    continue
                if dest not in targets:
                    continue
                file.write("{}\t{}\n".format(src, dest))

    make_graph(connections, artists)


def make_graph(connections, nodes):
    print("Generating graph...")
    G = nx.DiGraph()
    G.add_nodes_from([id for id in nodes])

    for src in connections:
        for dest in connections[src]:
            G.add_edge(src, dest)

    pos = nx.spring_layout(G)
    dd = nx.degree(G)

    for u, v, d in G.edges(data=True):
        d['w'] = random.random()

    edges, weights = zip(*nx.get_edge_attributes(G, 'w').items())

    print("Graph nodes: {}".format(len(G.nodes)))
    print("Graph edges: {}".format(len(G.edges)))

    plt.switch_backend('agg')
    plt.figure(figsize=(20, 20))
    nx.draw(G, pos=pos, with_labels=False,
            node_color='b',
            node_size=[v * 100 for v in dd.values()],
            edgelist=edges,
            width=2.0,
            edge_colour=weights,
            edge_cmap=plt.cm.Blues)
    plt.axis('off')
    print("Drawing graph...")
    plt.savefig("tu.pdf")
