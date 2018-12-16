# -------------------------------------------- IMPORTS ------------------------------------------------------

import csv
import numpy as np
from collections import Counter, defaultdict
from scrapy.selector import Selector
from bs4 import BeautifulSoup
import requests
import pickle
import matplotlib.pyplot as plt
import networkx as nx
from scrapy.utils.markup import remove_tags
import re
from networkx.algorithms import community
import xmltodict
import gensim
import logging
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
import stop_words
import tabulate

# -------------------------------------------- GLOBALS ------------------------------------------------------

letter_regex = re.compile(r'[^a-zA-Z\']')
space_regex = re.compile(r' +')


# -------------------------------------------- PART 1 -------------------------------------------------------


def split_book_name(regular_expression, string):
    """
    This function splits the books name using the syntax used in the site.
    """
    result = regular_expression.match(string)
    if result is None:  # Special case for Bran the Builder
        regular = re.compile(r'(.*) \((.*)')
        result = regular.match(string)
        if result is None:  # Characters that don't mention the way they are mentioned in the books.
            return string, None
    return result.group(1), result.group(2)


def clear_book_data(regular_expression, book_list):
    """
    This function clears the book list from strings to tuples of book and where the character is mentioned.
    """
    return list(map(lambda book_string: split_book_name(regular_expression, book_string), book_list))


def clear_html(html_text, tag):
    """
    This function gets the html data with the given tag and clears it.
    :param html_text: The html text.
    :param tag: The html tag.
    :return: A list of clean data
    """
    dirty_data = Selector(text=html_text).xpath('//tr[th = "{}"]/td'.format(tag)).extract()
    if not dirty_data:
        return []
    data_list = dirty_data[0].split('<br>')
    data_list = list(filter(lambda string: bool(string), map(str.lower ,map(str.strip, map(remove_tags, data_list)))))
    data_list = list(map(lambda x: x.replace("&amp;", '&'),data_list))
    data_list = list(map(lambda item: re.sub(r" ?\[[^]]+\]", "", item), data_list))
    return data_list


def create_data_and_return_characters():
    """
    This function creates the data and stores it in data.p
    """
    try:
        characters = pickle.load(open("./data/data.p", "rb"))
    except:
        url = 'http://awoiaf.westeros.org/index.php/List_of_characters'
        page = requests.get(url)
        soup = BeautifulSoup(page.text, "lxml") # parse HTML page
        main_list_element = soup.find_all('div',{'id':'mw-content-text'})[0] # retrieve HTML element that contains the list of characters
        list_of_elements = main_list_element.find_all('li') # find all list elements
        characters = dict()
        for list_element in list_of_elements:
            character_link = list_element.find_all('a')[0]  # get HTML element with link & character name
            characters[character_link.text] = character_link.attrs['href']
        characters_inverse = dict()
        for key in characters.keys():
            characters_inverse[characters[key]] = key
        regular_expression = re.compile(r'(.*) \((.*)\)')
        for character_name in characters.keys():
            print(character_name)
            character_url = characters[character_name]
            character_html = requests.get('http://awoiaf.westeros.org' + character_url)
            soup = BeautifulSoup(character_html.content, "lxml") # parse HTML page
            character = {'name':character_name, 'url':character_url, 'html_content':character_html.text}
            character['culture'] = clear_html(character_html.text, 'Culture')
            character['title'] = clear_html(character_html.text, 'Title') + clear_html(character_html.text, 'Other Titles')
            character["allegiance"] = clear_html(character_html.text, 'Allegiance')
            character['books'] = clear_book_data(regular_expression, clear_html(character_html.text, 'Book(s)')) + clear_book_data(regular_expression, clear_html(character_html.text, 'Books'))
            character['text'] = soup.find('div',{'id':'mw-content-text'}).text
            character['links'] = Counter([characters_inverse[link.get('href')] for link in soup.find_all('a') if link.get('href') in characters_inverse])
            character['clean_text'] = soup.text
            characters[character_name] = character
        pickle.dump(characters, open("./data/data.p", "wb"))
    return characters


# -------------------------------------------- PART 2 -------------------------------------------------------


def most_mentioned(characters):
    """
    This function prints the most mentioned character.
    """
    mentioned = Counter()
    for character in characters:
        mentioned += characters[character]['links']
    most_mentioned_character = mentioned.most_common(1)
    table = []
    for character in most_mentioned_character:
        table.append([character[0], character[1]])
    print(tabulate.tabulate(table, headers=['Character name','Amount of times mentioned'], tablefmt='orgtbl'))
    return most_mentioned_character


def get_connected_amount_counter_incoming_links(characters):
    """
    This function returns a counter containing the amount of times each character is connected.
    """
    connected_amount = Counter()
    for character in characters:
        for name in (characters[character]['links']).keys():
            connected_amount[name] += 1
    return connected_amount


def get_connected_amount_counter_outgoing_links(characters):
    """
    This function returns a counter containing the amount of times each character is connected.
    """
    connected_amount = Counter()
    for character in characters:
        for name in (characters[character]['links']).keys():
            connected_amount[character] += 1
    return connected_amount


def get_connected_amount_counter_all_links(characters):
    """
    This function returns a counter containing the amount of times each character is connected.
    """
    connected_amount = Counter()
    for character in characters:
        for name in (characters[character]['links']).keys():
            connected_amount[name] += 1
            connected_amount[character] += 1
    return connected_amount


def most_connected(characters, amount_of_characters_to_print=10):
    """
    This function prints the given amount of characters most connected.
    """
    functions = {'All links':get_connected_amount_counter_all_links,
                 'Incoming links':get_connected_amount_counter_incoming_links,
                 'Outgoing links': get_connected_amount_counter_outgoing_links}
    for func in functions.keys():
        most_connected_characters = (functions[func])(characters).most_common(amount_of_characters_to_print)
        table = []
        for character in most_connected_characters:
            table.append([character[0], character[1]])
        print((' '*14) + func)
        print('-'*47)
        print(tabulate.tabulate(table, headers=['Character name','Amount of connections'], tablefmt='orgtbl'))
        print()
        print()


def plot_degree_distribution_histogram(characters):
    """
    This function plots a degree distribution histogram of the characters.
    """
    connected_amount = get_connected_amount_counter_incoming_links(characters)
    degree_histogram = [0] * (max(connected_amount.values()) + 1)
    for character in characters:
        degree_histogram[connected_amount[character]] += 1
    f, (ax1, ax2) = plt.subplots(1, 2)
    f.suptitle('Degree distribution')
    ax1.set_title("Degree distribution histogram")
    ax1.bar(list(range(len(degree_histogram))), degree_histogram, align='center')
    ax1.set_xlabel("Amount of connections")
    ax1.set_ylabel("Degree of characters")
    ax2.set_title("Log Degree distribution histogram")
    ax2.bar(np.log(np.arange(len(degree_histogram)) + 1), np.log(np.array(degree_histogram)+1), align='center')
    ax2.set_xlabel("Log amount of connections")
    ax2.set_ylabel("Log degree of characters")
    plt.show()


def test_unidirectionality(characters):
    """
    This function prints out the amount of total links between characters, and the amount of one directional links.
    """
    amount_of_links = 0
    amount_of_one_direction_links= 0
    for character in characters:
        for neighbor in characters[character]['links']:
            amount_of_links += 1
            if character not in characters[neighbor]['links']:
                amount_of_one_direction_links += 1
    print('Out of {} links between characters, There are {} one directional links.'.format(amount_of_links, amount_of_one_direction_links))


def plot_culture(characters):
    """
    This function plots a pie of the sizes of the cultures.
    """
    cultures = Counter()
    for character in characters:
        culture = characters[character]['culture']
        if len(culture) != 0:
            cultures[culture[0]] += 1
        else:
            cultures['No culture'] += 1
    cultures['Non significant'] = 0
    for culture in cultures.keys():
        if culture == 'Non significant':
            continue
        elif cultures[culture] <= 10:
            cultures['Non significant'] += cultures[culture]
    labels = []
    sizes = []
    for culture in cultures.keys():
        if cultures[culture] <= 10:
            continue
        labels.append(culture)
        sizes.append(cultures[culture])
    labels = [x for _, x in sorted(zip(sizes, labels))]
    sizes.sort()
    explodeTuple = [0.3] * len(sizes)
    explodeTuple = tuple(explodeTuple)
    plt.figure("Culture distribution - All")
    plt.pie(sizes,explode=explodeTuple, labels=labels, autopct='%1.1f%%')
    labels = labels[:-1]
    sizes = sizes[:-1]
    explodeTuple = explodeTuple[:-1]
    plt.figure("Culture distribution - Excluding no culture and non significant")
    plt.pie(sizes,explode=explodeTuple, labels=labels, autopct='%1.1f%%')
    plt.show()


def book_table(characters):
    """
    This function prints a table of characters and in what book do the appear.
    """
    books_characters = dict()
    for character in characters:
        for book in characters[character]['books']:
            if book[0] == 'game of thrones':
                book = ('a game of thrones', book[1])
            elif book[0] == "'the world of ice & fire":
                book = ('the world of ice & fire', book[1])
            elif book[0] == "a dance with dragons (mentioned) the winds of winter":
                books_characters['a dance with dragons'].append(character)
                books_characters['the winds of winter'].append(character)
                continue
            elif book[0] == "the world of ice and fire":
                book = ('the world of ice & fire', book[1])
            if book[0] not in books_characters.keys():
                books_characters[book[0]] = []
            books_characters[book[0]].append(character)
    books = ['Characters'] + list(books_characters.keys())
    total = ['Total Amount of characters'] + ([0]*(len(books) -1))
    total_appear = ['Total Amount of characters appearing'] + ([0]*(len(books) -1))
    total_mentioned = ['Total Amount of characters mentioned'] + ([0]*(len(books) -1))
    total_appendix = ['Total Amount of characters appendixed'] + ([0]*(len(books) -1))
    total_not_labeled = ['Total Amount of characters not labeled'] + ([0]*(len(books) -1))
    table = [total, total_appear, total_mentioned, total_appendix, total_not_labeled]
    break_row = ['-' * (len(total[0]) +2)]
    for book in books[1:]:
        break_row.append('-' * (len(book) + 2))
    table.append(break_row)
    for character in characters:
        table.append([character] + ['X'] * (len(books) - 1))
        if character == 'Rafford':
            table[-1][books.index('a dance with dragons')] = 'mentioned'
            table[-1][books.index('the winds of winter')] = 'appears'
        for book in books_characters.keys():
            if character in books_characters[book]:
                for character_book in characters[character]['books']:
                    if character_book[0] == 'game of thrones':
                        character_book = ('a game of thrones', character_book[1])
                    elif character_book[0] == "'the world of ice & fire":
                        character_book = ('the world of ice & fire', character_book[1])
                    elif character_book[0] == "the world of ice and fire":
                        character_book = ('the world of ice & fire', character_book[1])
                    if character_book[0] == book:
                        table[-1][books.index(book)] = character_book[1]
                        break
                if character_book[1] == 'appears':
                    table[1][books.index(book)] += 1
                elif character_book[1] == 'mentioned':
                    table[2][books.index(book)] += 1
                elif character_book[1] == 'appendix':
                    table[3][books.index(book)] += 1
                else:
                    table[4][books.index(book)] += 1
                table[0][books.index(book)] += 1
    print(tabulate.tabulate(table, headers=books, tablefmt='orgtbl'))
    return


def title_table(characters):
    """
    This function prints a table of the characters that have one of the titles we chose to show.
    (ser, commander, lord, king, prince, lady, queen, knight).
    """
    title_characters = dict()
    for character in characters:
        for title in characters[character]['title']:
            if 'ser' in title:
                title = 'ser'
            elif 'commander' in title:
                title = 'commander'
            elif 'lord' in title:
                title = 'lord'
            elif 'king' in title:
                title = 'king'
            elif 'prince' in title:
                title = 'prince'
            elif 'lady' in title:
                title = 'lady'
            elif 'queen' in title:
                title = 'queen'
            elif 'knight' in title:
                title = 'knight'
            else:
                continue
            if title not in title_characters.keys():
                title_characters[title] = []
            title_characters[title].append(character)
    table = []
    for title in title_characters.keys():
        table.append([str.upper(title)+"'s"] + ['Amount={}'.format(len(title_characters[title]))] + title_characters[title])
    print(tabulate.tabulate(table, tablefmt='orgtbl'))
    return


def allegiance_table(characters):
    """
    This function prints a table of the strongest houses and the characters that are aliened with that house.
    """
    threshold = 10
    allegiance_characters = dict()
    for character in characters:
        for allegiance in characters[character]['allegiance']:
            if allegiance not in allegiance_characters.keys():
                allegiance_characters[allegiance] = []
            allegiance_characters[allegiance].append(character)
    table = []
    allegiance_characters_list = []
    sizes = []
    for allegiance in allegiance_characters.keys():
        allegiance_characters_list.append([allegiance] + allegiance_characters[allegiance])
        sizes.append(len(allegiance_characters[allegiance]))
    allegiance_characters_list = [x for _, x in sorted(zip(sizes, allegiance_characters_list))]
    allegiance_characters_list.reverse()
    for allegiance in allegiance_characters_list:
        if (len(allegiance) - 1) < threshold:
            break
        table.append([str.upper(allegiance[0])] + ['Amount={}'.format(len(allegiance) - 1)] + allegiance[1:])
    print(tabulate.tabulate(table, tablefmt='orgtbl'))
    return


def traitor_table(characters):
    """
    This function prints a table of the characters that are betraying their houses and aliened with a
    different house.
    """
    table = []
    threshold = 10
    allegiance_characters = dict()
    for character in characters:
        for allegiance in characters[character]['allegiance']:
            if allegiance not in allegiance_characters.keys():
                allegiance_characters[allegiance] = []
            allegiance_characters[allegiance].append(character)
    for allegiance in allegiance_characters.keys():
        if len(allegiance_characters[allegiance]) < threshold:
            continue
        for character in allegiance_characters[allegiance]:
            family_name = character.split()
            if len(family_name) <= 1:
                continue
            family_name = family_name[-1].lower()
            for allegiance_name in allegiance_characters.keys():
                if family_name in allegiance_name and family_name not in allegiance:
                    table.append([character, allegiance, allegiance_name])
    print(tabulate.tabulate(table,headers=['Character name', 'Part of allegiance', 'betraying allegiance'], tablefmt='orgtbl'))
    return


def shed_more_light(characters):
    """
    This function runs the functions that are supposed to shed more light on the data.
    """
    plot_culture(characters)
    print("BOOK TABLE")
    print()
    book_table(characters)
    print()
    print()
    print("TITLE TABLE")
    print()
    title_table(characters)
    print()
    print()
    print("ALLEGIANCE TABLE")
    print()
    allegiance_table(characters)
    print()
    print()
    print("TRAITOR TABLE")
    print()
    traitor_table(characters)


# -------------------------------------------- PART 3 -------------------------------------------------------

def create_edges(characters):
    """
    creates edges for nx graph
    """
    network_links = Counter() # uses a Counter obj to create edges
    for source in characters:
        for dest in characters[source]['links'].keys():
            if source != dest:
                if source > dest:  # swap
                    swapped_source, swapped_dest = dest, source
                else:
                    swapped_source, swapped_dest = source, dest
                network_links[(swapped_source, swapped_dest)] += characters[source]['links'][swapped_dest]
    return network_links


def create_graph(characters, min_weight=1):
    """
    creates nx graph based on input characters
    """
    network_links = create_edges(characters)
    graph = nx.Graph()
    for link, weight in network_links.items():
        if weight >= min_weight:
            source = link[0]
            dest = link[1]
            graph.add_edge(source, dest, weight=weight)
    graph = graph.to_undirected()  # turn to undirected
    print('Graph created with Nodes: {0:d}, Links: {1:d}'.format(graph.number_of_nodes(), graph.number_of_edges()))
    return graph

def remove_nodes_by_degree(graph, min_degree):
    """
    method intended to iteratively remove nodes up until min_degree
    """
    proceed = True
    while proceed:
        nodes_to_remove = [node for node, degree in graph.degree(graph.nodes) if degree < min_degree]
        if len(nodes_to_remove) == 0 or len(graph.nodes) <= 100:
            proceed = False

        else:
            # print('Removing {0:d} nodes'.format(len(nodes_to_remove)))
            graph.remove_nodes_from(nodes_to_remove)
            # print('Remaining Nodes: {0:d}, Links: {1:d}'.format(graph.number_of_nodes(), graph.number_of_edges()))
    return graph


def keep_nodes_by_name(graph, names_to_keep):
    """
    method to set a graph nodes by inputted names_to_keep, returns graph
    """
    proceed = True
    while proceed:
        nodes_to_remove = [node for node in graph.nodes() if node not in names_to_keep]
        if len(nodes_to_remove) == 0:
            proceed = False
        else:
            # print('Removing {0:d} nodes'.format(len(nodes_to_remove)))
            graph.remove_nodes_from(nodes_to_remove)
            # print('Remaining Nodes: {0:d}, Links: {1:d}'.format(graph.number_of_nodes(), graph.number_of_edges()))
    return graph


def remove_nodes_by_degree_not_iterative(graph, min_degree):
    """
    A non iterative method to set a min_degree
    """
    # with 17 as min_degree this gives us the top 100 chars with highest initial degree
    nodes_to_remove = [node for node, degree in graph.degree(graph.nodes) if degree < min_degree]
    #print('Removing {0:d} nodes'.format(len(nodes_to_remove)))
    graph.remove_nodes_from(nodes_to_remove)
    #print('Remaining Nodes: {0:d}, Links: {1:d}'.format(graph.number_of_nodes(), graph.number_of_edges()))
    return graph


def find_comms(graph):
    """
    find communities within a graph, returns a list of sets, with each set representing a ccmmunities
    """
    communities = [comminuity for comminuity in
                   community.label_propagation.label_propagation_communities(graph)]
    #print('Found {0:d} communities using label propagation algorithm'.format(len(communities)))
    return communities


def find_colors(communities):
    """
    generate color set based on inputted communities
    """
    return plt.cm.get_cmap('hsv', len(communities))  # colors(n) will generate a color for community

def find_all_chars_chars_and_create_csvs(graph):
    """ based on entered graph generate 5 csv's according to 5 graph-assisting algorithms
    """
    def calc_pagerank(graph):
        return nx.algorithms.pagerank(graph)

    def calc_betweenes(graph):
        return nx.algorithms.betweenness_centrality(graph)

    def calc_degree(graph):
        return nx.algorithms.degree_centrality(graph)

    def calc_closeness(graph):
        return nx.algorithms.closeness_centrality(graph)

    def calc_weighted_degree(graph):
        return graph.degree(weight='weight')

    def find_central_chars_by_centrailty(graph, node_func, centrailty_type):
        """
        based on entered graph and algorthim function (node_func) generate a csv containing the top
        100 characters, the csv will be named by entered  centrailty_type param
        """
        nodes = dict(node_func(graph)) # generate the nodes based on algorithm func

        # generate an in order importance character list
        node_names = np.array(list(nodes.keys()))
        node_values = np.array(list(nodes.values()))
        sort_ortder = np.argsort(node_values)[::-1]
        ret = []
        for i in sort_ortder[:100]: # cut to top 100 char's
            ret.append(node_names[i])

        # insert ret to csv
        with open(centrailty_type, 'w', encoding="utf-8", newline='') as csvfile:  # creating appropriate csv
            filewriter = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            for row in ret:
                filewriter.writerow([str(row)])
        print('Based on ' + centrailty_type[:-4] + " most central characters are " + str(ret))

    find_central_chars_by_centrailty(graph, calc_pagerank, 'page_rank.csv')
    find_central_chars_by_centrailty(graph, calc_betweenes, 'betweenes_rank.csv')
    find_central_chars_by_centrailty(graph, calc_degree, 'degree_rank.csv')
    find_central_chars_by_centrailty(graph, calc_closeness, 'closeness_rank.csv')
    find_central_chars_by_centrailty(graph, calc_weighted_degree, 'weighted_rank.csv')

def generate_comms(characters, centrality_csv, threshold):
    """
    Utilty func to generate a graph and communities based on characters data, a csv with mains characters
    and a threshold of minimum degree for each edge.
    """
    graph_characters = []
    with open(centrality_csv, 'r', encoding="utf-8", newline='') as csvfile:
        filereader = csv.reader(csvfile, quoting=csv.QUOTE_ALL)
        for row in filereader:
            graph_characters.append(row[0])

    graph = create_graph(characters, threshold)
    graph = keep_nodes_by_name(graph, graph_characters)
    comm = find_comms(graph)
    return comm


def generate_new_graph_and_csvs(min_weight):
    """ this function creates a new graph base on inputted based on min_degree paramter as well creates
    the top charchters csv's"""
    charchters = create_data_and_return_characters()
    graph = create_graph(charchters, min_weight)
    find_all_chars_chars_and_create_csvs(graph)


def graph_by_centrality(centrality_csv, characters, edges_threshold, color_by_culture, min_degree):
    """
    Part 3 main function. Ment to create an nx graph based on input csv, generete colored comms or color nodes by
    culture based on color_by_culture param.
    :param centrality_csv: The csv containing the main story charcters
    :param characters: the data characters
    :param edges_threshold: the minimum threshold each edge in the graph must have
    :param color_by_culture: a bool param, if true color by culture, else color by comm
    :param min_degree: int param to set reduced graph to a min degree
    :return:
    """
    graph_characters = []

    ## open csv and get main charcters
    with open(centrality_csv, 'r', encoding="utf-8", newline='') as csvfile:
        filereader = csv.reader(csvfile, quoting=csv.QUOTE_ALL)
        for row in filereader:
            graph_characters.append(row[0])

    graph = create_graph(characters, edges_threshold) # create graph based on threshold
    graph = remove_nodes_by_degree(graph, min_degree) # remove nodes to hold min degree
    graph = keep_nodes_by_name(graph, graph_characters) # only keep nodes from csv
    node_colors = [None] * graph.number_of_nodes() # holder for node colors
    node_sizes = [None] * graph.number_of_nodes() # holder for node size
    node_names = [node for node in graph.nodes] # all node names

    cultures = defaultdict(list)
    for name in graph_characters: # collect all cultures from nodes
        if characters[name].get('culture') and len(characters[name]['culture']) > 0:
            cur_culture = characters[name]['culture'][0]
            cultures[cur_culture].append(name)
    if color_by_culture:  #color by culture
        color_by = 'culture'
        colors = find_colors(cultures)
        for community_id, culture in enumerate(cultures):
            community_color = colors(community_id)
            for node in cultures[culture]:
                if node in node_names:
                    node_colors[node_names.index(node)] = community_color
                    value = max(200,min(1000, 1500 * (8 / (graph_characters.index((node)) + 1))))
                    node_sizes[node_names.index(node)] = value
    else: # color by community
        color_by = 'comm'
        comm = find_comms(graph)
        colors = find_colors(comm)
        for community_id, community in enumerate(comm):
            community_color = colors(community_id)
            for node in community:
                if node in node_names:
                    node_colors[node_names.index(node)] = community_color
                    value = max(200,min(1000, 1500*(8/(graph_characters.index((node))+1))))
                    node_sizes[node_names.index(node)] = value

    plt.figure(figsize=(30,18)) # set figure size

    # fail safe - set black to all nodes without culture of comm
    for i in range(len(node_colors)):
        if not node_colors[i]:
            node_colors[i] = (0,0,0) # node default color is black
        if not node_sizes[i]:
            node_sizes[i] = 100 # node default size is 100
    nx.draw(graph, pos=nx.nx_pydot.graphviz_layout(graph), node_size=node_sizes, node_color=node_colors, with_labels=True)
    plt.draw()
    # plt.show() - uncomment this to show graph
    plt.savefig((color_by+str(min_degree)+centrality_csv + str(edges_threshold) + ".jpg")) # save fig

def test_graph_by_name(characters):
    """
    generate many types of graphs, with different thresholds and min degrees
    """
    for csv in ['betweenes_rank.csv', 'page_rank.csv', 'closeness_rank.csv', 'weighted_rank.csv', 'degree_rank.csv']:
        for theshold in range(3, 8):
            for min_degree in range(1,5):
                graph_by_centrality(csv, characters, theshold, True, min_degree) # color by culture
                graph_by_centrality(csv, characters, theshold, False, min_degree) # color by comm


def compare_algorithms():
    csvs = ['betweenes_rank.csv', 'closeness_rank.csv', 'degree_rank.csv', 'page_rank.csv', 'weighted_rank.csv']
    csv_dict = dict()
    csv_names = []
    for csv_file in csvs:
        csv_name = ' '.join((csv_file.split('.')[0]).split('_'))
        csv_names.append(csv_name)
        csv_dict[csv_name] = []
        with open(csv_file, 'r', encoding='utf-8', newline='') as csvfile:
            filereader = csv.reader(csvfile, quoting=csv.QUOTE_ALL)
            for row in filereader:
                csv_dict[csv_name].append(row[0])
    for csv1_index in range(len(csv_names)):
        csv2_index = csv1_index + 1
        while csv2_index < len(csv_names):
            csv1_diff = []
            csv2_diff = []
            for character in csv_dict[csv_names[csv1_index]]:
                if character not in csv_dict[csv_names[csv2_index]]:
                    csv1_diff.append(character)
            for character in csv_dict[csv_names[csv2_index]]:
                if character not in csv_dict[csv_names[csv1_index]]:
                    csv2_diff.append(character)
            print('There are',len(csv1_diff), 'differences between', "'" + csv_names[csv1_index] + "'", 'and', "'" + csv_names[csv2_index] + "'", 'and they are:')
            print('Only in', "'" + csv_names[csv1_index] + "':", ', '.join(csv1_diff))
            print('Only in', "'" + csv_names[csv2_index] + "':", ', '.join(csv2_diff))
            print()
            csv2_index += 1


# -------------------------------------------- PART 4 -------------------------------------------------------


def download_all_pages():
    """
    This function downloads all pages using the site map or loads the data that was already downloaded and returns it.
    """
    try:
        pages = pickle.load(open("./data/all pages.p", "rb"))
    except:
        pages = dict()
        with open('sitemap-westeros-mw_-NS_0-0.xml', 'r') as sitemap:
            xml_response = xmltodict.parse(sitemap.read())
            # Needs to be a positive number.
            num_iter = int(input("How many pages to download before asking to change IP?"))
            i = 0
            for xml in xml_response['urlset']['url']:
                if i == num_iter and i != 0:
                    num_iter = int(input("Change your ip address and then enter nummber of pages to download before asking again"))
                    i = 0
                url = xml['loc']
                pages[url] = requests.get(url).text
        pickle.dump(pages, open("./data/all pages.p", "wb"))
    return pages


def remove_stop_words(sentence, english_stop_words):
    """
    This function removes stop words from a sentence.
    """
    sentence_words = sentence.split(' ')
    result_words = [words for words in sentence_words if words not in english_stop_words]
    return ' '.join(result_words)


def get_clean_text(page):
    """
    This function gets a page and returns only the clean text from it.
    """
    page = page.replace('>', '> ')
    page = page.replace('\n', '.')
    x = BeautifulSoup(page, 'lxml')
    [s.extract() for s in x('script')]
    clean_text = x.text.lower()
    clean_text = clean_text.split('.')  # Splitting into sentences
    clean_text = list(map(lambda y: letter_regex.sub(' ', y), clean_text))
    clean_text = list(map(lambda y: space_regex.sub(' ', y), clean_text))
    clean_text = list(map(lambda y: y.strip(), clean_text))
    clean_text = list(filter(lambda y: bool(y), clean_text))
    english_stop_words = stop_words.get_stop_words('en')
    clean_text = list(map(lambda y: remove_stop_words(y, english_stop_words), clean_text))
    return list(map(lambda y: gensim.utils.simple_preprocess(y), clean_text))


def get_all_pages_clean_text():
    """
    This function returns all pages after cleaning the text.
    """
    try:
        pages = pickle.load(open("./data/all pages clean text.p", "rb"))
    except:
        pages = download_all_pages()
        for key in enumerate(pages.keys()):
            pages[key] = get_clean_text(pages[key])
        pickle.dump(pages, open("./data/all pages clean text.p", "wb"))
    return pages


def amount_of_pages_downloaded():
    """
    This function prints the amount of pages downloaded.
    """
    all_pages = download_all_pages()
    print("The amount of pages download is {} pages.".format(len(all_pages)))


def combine_sentences(sentences_of_words):
    """
    This function returns a list of the sentences given.
    """
    combined_sentences = []
    for sentence in sentences_of_words:
        combined_sentences += sentence
    return combined_sentences


def text_length(clean_page_text):
    """
    This function returns the length of all text given in sentences.
    """
    return len(combine_sentences(clean_page_text))


def average_text_length():
    """
    This function prints the average text length on a page.
    """
    all_pages_clean_text = get_all_pages_clean_text()
    text_length_list = list(map(text_length, all_pages_clean_text.values()))
    print("The average text length in a page is {}.".format(sum(text_length_list) / float(len(all_pages_clean_text))))


def plot_text_length_histogram():
    """
    This function plots the text length histogram.
    """
    histogram_counter = Counter()
    all_pages_clean_text = get_all_pages_clean_text()
    text_length_list = list(map(text_length, all_pages_clean_text.values()))
    for length in text_length_list:
        histogram_counter[length] += 1
    histogram = []
    min_value = min(histogram_counter.keys())
    max_value = max(histogram_counter.keys())
    for i in range(min_value, max_value + 1):
        histogram.append(histogram_counter[i])
    f, (ax1, ax2) = plt.subplots(1, 2)
    f.suptitle('Text length distribution')
    ax1.set_title("Text length distribution histogram")
    ax1.bar(list(range(min_value, max_value + 1)), histogram, align='center')
    ax1.set_xlabel("Text length")
    ax1.set_ylabel("Amount of pages")
    histogram = []
    for i in range(max_value + 1):
        histogram.append(histogram_counter[i])
    ax2.set_title("Log text length distribution histogram")
    ax2.bar(np.log(np.arange(len(histogram)) + 1), np.log(np.array(histogram)+1), align='center')
    ax2.set_xlabel("Log text length")
    ax2.set_ylabel("Log amount of pages")
    plt.show()


def amount_of_words():
    """
    This function returns the amount of words found on all pages.
    """
    all_pages_clean_text = get_all_pages_clean_text()
    words = []
    for clean_text in all_pages_clean_text.values():
        words += combine_sentences(clean_text)
    return len(words)


def amount_of_unique_words():
    """
    This function returns the amount of unique words found on all pages.
    """
    all_pages_clean_text = get_all_pages_clean_text()
    words = set()
    for clean_text in all_pages_clean_text.values():
        words = words.union(combine_sentences(clean_text))
    return len(words)


def amount_of_words_found():
    """
    This function prints the amount of words and the amount of unique words found on all pages.
    """
    print("The amount of words found is: {}.".format(amount_of_words()))
    print("The amount of unique words found is: {}.".format(amount_of_unique_words()))


def plot_distribution_of_words(print_common_words=False):
    """
    This function plots the distribution of the words.
    """
    all_pages_clean_text = get_all_pages_clean_text()
    words = []
    for clean_text in all_pages_clean_text.values():
        words += combine_sentences(clean_text)
    distribution = Counter()
    distribution_amount = Counter()
    for word in words:
        distribution[word] += 1
    if print_common_words:
        common = []
        size = []
        for k in distribution.keys():
            if distribution[k] > 2000:
                common.append(k)
                size.append(distribution[k])
        common = [x for _, x in sorted(zip(size, common))]
        common.reverse()
        size.sort()
        size.reverse()
        for i in range(len(common)):
            print(common[i], size[i])
    for amount in distribution.values():
        distribution_amount[amount] += 1
    histogram = []
    min_value = min(distribution_amount.values())
    max_value = max(distribution_amount.values())
    for i in range(min_value, max_value + 1):
        histogram.append(distribution_amount[i])
    f, (ax1, ax2) = plt.subplots(1, 2)
    f.suptitle('Word distribution')
    ax1.set_title("Word distribution histogram")
    ax1.set_ylim([0, 500])
    ax1.bar(list(range(min_value, max_value + 1)), histogram, align='center')
    ax1.set_xlabel("Word amount of appearance")
    ax1.set_ylabel("Amount of words")
    histogram = []
    for i in range(max_value + 1):
        histogram.append(distribution_amount[i])
    ax2.set_title("Log text length distribution histogram")
    ax2.bar(np.log(np.arange(len(histogram)) + 1), np.log(np.array(histogram)+1), align='center')
    ax2.set_xlabel("Log word amount of appearance")
    ax2.set_ylabel("Log amount of words")
    plt.show()
    return distribution


def create_text_data():
    """
    This function returns text data for training a word2vec model.
    """
    all_pages_clean_text = get_all_pages_clean_text()
    all_sentences = []
    for sentence in all_pages_clean_text.values():
        all_sentences += sentence
    return all_sentences


def get_word2vec_trained_model():
    """
    This function returns a trained word2vec model.
    """
    try:
        model = pickle.load(open("./data/word2vec trained model.p", "rb"))
    except:
        text_data = create_text_data()
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        model = gensim.models.Word2Vec(text_data, min_count=2, window=10, size=150)
        pickle.dump(model, open("./data/word2vec trained model.p", "wb"))
    return model


# -------------------------------------------- PART 5 -------------------------------------------------------

def illustrate_word2vec_preserve_semantics(word_list, amount_of_words=10):
    """
    This function prints words that shows that the word2vec learned the semantics.
    """
    table = []
    header = ['Positive words', 'Negative words']
    for i in range(amount_of_words):
        header.append("Word {}".format(i))
        header.append("Similarity {}".format(i))
    model = get_word2vec_trained_model()
    for word in word_list:
        row = []
        if type(word) == list:
            if len(word) == 2 and type(word[1]) == list:
                row.append(', '.join(word[0]))
                row.append(', '.join(word[1]))
                similar_words = model.wv.most_similar(positive=word[0], negative=word[1], topn=amount_of_words)
            else:
                row.append(', '.join(word))
                row.append('----')
                similar_words = model.wv.most_similar(positive=word, topn=amount_of_words)
        else:
            row.append(word)
            row.append('----')
            similar_words = model.wv.most_similar(positive=[word], topn=amount_of_words)
        for similar_word in similar_words:
            row.append(similar_word[0])
            row.append(similar_word[1])
        table.append(row)
    print(tabulate.tabulate(table,headers=header, tablefmt='orgtbl'))


def test_tsne(word_list, print_similar=False):
    """
    This function illustrates that tsne can project data nicely into 2D.
    :return:
    """
    word_vectors = []
    colors = []
    print_data = ''
    model = get_word2vec_trained_model()
    for i, word in enumerate(word_list):
        word_vectors.append(np.array(model.wv[word]))
        colors.append(i)
        if print_similar:
            print_data += word
            print_data += ": "
        for similar_word in model.wv.most_similar(positive=[word], topn=10):
            if print_similar:
                print_data += similar_word[0]
                print_data += ", "
            word_vectors.append(model.wv[similar_word[0]])
            colors.append(i)
        if print_similar:
            print_data += "\n"
    if print_similar:
        print(print_data)
    distance_matrix = pairwise_distances(np.array(word_vectors), np.array(word_vectors), metric='cosine', n_jobs=-1)
    embedded_words = TSNE(perplexity=5, metric="precomputed").fit_transform(distance_matrix)
    embedded_words = np.array(embedded_words)
    f, ax = plt.subplots()
    ax.scatter(embedded_words[:, 0], embedded_words[:, 1], c=colors)
    for i, word_vector in enumerate(word_vectors):
        ax.annotate(model.most_similar(positive=[word_vector], topn=1)[0][0], (embedded_words[i, 0], embedded_words[i, 1]))
    plt.show()


def project_most_prominent_characters_into_2D_by_culture(centrality_csv):
    """
    This function plots the most prominent characters into 2D and colors them by culture
    """
    model = get_word2vec_trained_model()
    characters = create_data_and_return_characters()
    characters_names = []
    characters_vectors = []
    characters_cultures = []
    cultures = {}
    color = 0
    with open(centrality_csv, 'r', encoding="utf-8", newline='') as csvfile:
        filereader = csv.reader(csvfile, quoting=csv.QUOTE_ALL)
        for row in filereader:
            if len(characters[row[0]]['culture']) == 0:
                characters[row[0]]['culture'] = ["No culture"]
            characters_names.append(row[0])
            characters_vectors.append(model.wv[row[0].lower().split()[0]])
            if characters[row[0]]['culture'][0] not in cultures.keys():
                cultures[characters[row[0]]['culture'][0]] = color
                color += 1
            characters_cultures.append(cultures[characters[row[0]]['culture'][0]])
    distance_matrix = pairwise_distances(np.array(characters_vectors), np.array(characters_vectors), metric='cosine', n_jobs=-1)
    perplexity = 10
    embedded_words = TSNE(perplexity=perplexity, metric="precomputed").fit_transform(distance_matrix)
    embedded_words = np.array(embedded_words)
    f, ax = plt.subplots()
    ax.scatter(embedded_words[:, 0], embedded_words[:, 1], c=characters_cultures)
    for i, character in enumerate(characters_names):
        ax.annotate(character, (embedded_words[i, 0], embedded_words[i, 1]), size=7)
    plt.savefig('./project_culture_output/' + str(centrality_csv.split('.')[0])+' '+ str(perplexity))
    plt.close()


def project_most_prominent_characters_into_2D_by_community(centrality_csv):
    """
    This function plots the most prominent characters into 2D and colors them by communities.
    """
    model = get_word2vec_trained_model()
    characters_names = []
    characters_vectors = []
    characters_community = []
    threshold = 7
    communities = generate_comms(create_data_and_return_characters(), centrality_csv, threshold)
    for i, community in enumerate(communities):
        for character in community:
            characters_names.append(character)
            characters_vectors.append(model.wv[character.lower().split()[0]])
            characters_community.append(i)
    distance_matrix = pairwise_distances(np.array(characters_vectors), np.array(characters_vectors), metric='cosine', n_jobs=-1)
    perplexity = 10
    embedded_words = TSNE(perplexity=perplexity, metric="precomputed").fit_transform(distance_matrix)
    embedded_words = np.array(embedded_words)
    f, ax = plt.subplots()
    ax.scatter(embedded_words[:, 0], embedded_words[:, 1], c=characters_community)
    for i, character in enumerate(characters_names):
        ax.annotate(character, (embedded_words[i, 0], embedded_words[i, 1]))
    plt.savefig('./project_community_output/' + str(centrality_csv.split('.')[0])+' '+ str(perplexity))
    plt.close()
