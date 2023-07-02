# coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions
to this file that are not part of the classes that we want.
"""

import heapq as hq
import os
import pickle
import math
import copy
import random
import time



class PriorityQueue(object):
    """
    A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    (Hint: take a look at the module heapq)

    Attributes:
        queue (list): Nodes added to the priority queue.
        
        
    Given queue node tuple = (priority, value)
    
    Modified queue node tuple = (priority, count, value)
    
    """

    def __init__(self):
        """Initialize a new Priority Queue."""

        self.queue = []
        self.count=0
        

    def pop(self):
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """
        noodle = hq.heappop(self.queue)
        
        return (noodle[0],noodle[-1])

    def remove(self, node):
        """
        Remove a node from the queue.

        Hint: You might require this in ucs. However, you may
        choose not to use it or to define your own method.

        Args:
            node (tuple): The node to remove from the queue.
        """
        
        for thingy in self.queue:
            if thingy[0] == node[0] and thingy[-1] == node[-1]:
                poppit = thingy
                break
        self.queue.remove(poppit)
        hq.heapify(self.queue)
        
    def get_node(self,key):
        
        #returns formatted node without removal
        
        for n in self.queue:
            try:
                if n[-1][0] == key:
                    return (n[0],n[-1])
            except:
                if n[-1] == key:
                    return (n[0],n[-1])
    
    def heuristic_mod(self,goal):
        
        # modifies heuristic scores if both paths to a specific goal have been found
        # ONLY USED FOR n-directional A*
        
        # node format (tuple) -- (current node score + min(heuristic), counter, (current node, previous node, (ranked heuristic tuple) ) )
        # ranked heuristic tuple -- ( (min heuristic, current goal, heuristic goal),...
        #                              (max heuristic, current goal, heuristic goal) )
        
        
        for node in copy.copy(self.queue):
            # heur_tuple=()
            # heur_list=[]
            
            flag = 0
            
            
            heur_tuple = node[-1][2]
            heur_list = list(heur_tuple)
            
            original_heuristic = node[-1][2][0][0]
            
            for heur in heur_tuple:
                if goal in heur:
                    heur_list.remove(heur)
                    flag+=1
            heur_list.sort()
            heur_tuple = tuple(heur_list)
            
            
            if flag > 0 :
                score = node[0] - original_heuristic + heur_tuple[0][0]
                rank = node[1]
                node_info = (node[-1][0], node[-1][1], copy.copy(heur_tuple))
                self.queue.remove(node)
                self.queue.append((score,rank,node_info))
         
        hq.heapify(self.queue)
            
                
            
            
        
        
    
    def __iter__(self):
        """Queue iterator."""

        return iter(sorted(self.queue))

    def __str__(self):
        """Priority Queue to string."""

        return 'PQ:%s' % self.queue

    def append(self, node):
        """
        Append a node to the queue.

        Args:
            node: Comparable Object to be added to the priority queue.
        """
        
        
        self.count+=1
        
        noodle = (node[0], copy.copy(self.count), node[1])
        
        hq.heappush(self.queue, noodle)
        
        
    def __contains__(self, key):
        """
        Containment Check operator for 'in' 

        Args:
            key: The key to check for in the queue. (ALSO CALLED A VALUE IN OTHER DOCUMENTATION)

        Returns:
            True if key is found in queue, False otherwise.
        """
        for n in self.queue:
            try:
                if n[-1][0] == key:
                    return True
            except:
                if n[-1] == key:
                    return True
        return False
        
        

    def __eq__(self, other):
        """
        Compare this Priority Queue with another Priority Queue.

        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self.queue == other.queue

    def size(self):
        """
        Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def top(self):
        """
        Get the top item in the queue.

        Returns:
            The first item stored in teh queue.
        """

        return (self.queue[0][0],self.queue[0][-1])

def breadth_first_search(graph, start, goal):
    """
    Warm-up exercise: Implement breadth-first-search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    if start == goal:
        return []
    
    prev_layer = [start]
    next_layer = []
    
    # dict format -- node : predecessor
    found_nodes = {}
    found_nodes[start]=None
    
    while len(prev_layer)>0:
        for node in prev_layer:
            
            neighbors = []
            neighborino = graph.neighbors(node)
            
            while True:
                try:
                    neighbors.append(next(neighborino))
                    
                except StopIteration:
                    break
            
            neighbors.sort()
            
            for q in neighbors:
                
                #eliminates repeats
                if q not in found_nodes:
                    found_nodes[q]=node
                    next_layer.append(q)           
                
                if q == goal:
                    best_path = [q]
                    while found_nodes[best_path[-1]] != None:
                        best_path.append(found_nodes[best_path[-1]])
                    
                    
                    best_path.reverse()
                    
                    return best_path
                
        
        prev_layer = copy.copy(next_layer)
        next_layer = []
    return []
                            



def uniform_cost_search(graph, start, goal):
    """
    Warm-up exercise: Implement uniform_cost_search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    
    use:
        graph.get_edge_weight(node_1, node_2)
        graph.neighbors(node)
        
        queue searches by lowest total cost
        
        Given PriorityQueue node tuple = (priority, value)
    
    
    """
    '''
    if start == goal:
        return []
    
    #dictionary to trace back to origin
    # dict format -- node : (current node score, predecessor)
    found_nodes = {}
    found_nodes[start]=None
    
    #frontier node format
    # tuple -- (current node score, current node)
    search_tree = PriorityQueue()
    
    score = 0
    origin = (score,start)
    
        
    while origin[1] != goal:       
        
        hideyho = graph.neighbors(origin[1]) #get frontier iter() list thingy
          
        while True:
            try:
                q = next(hideyho) #gets the next frontier
                score = origin[0] + graph.get_edge_weight(origin[1], q)
                
                
                #write node to traceback dictionary
                if q not in found_nodes:
                    found_nodes[q]=(score,origin[1])
                    search_tree.append((score,q))
                 
                elif found_nodes[q] != None and score < found_nodes[q][0]:
                    found_nodes[q]=(score,origin[1])
                                    
            except StopIteration:
                break
                    
        origin = search_tree.pop()
    
    if goal in found_nodes:
        best_path = [goal]
        while found_nodes[best_path[-1]] != None:
            best_path.append(found_nodes[best_path[-1]][1])
        
        
        best_path.reverse()
        
        return best_path
    
    else:
        return[]
    
    attempt #2
     -----------------------------------------------------------------------'''
    print ('In the function')
    
    if start == goal:
        return []
    
    #dictionary to trace back to origin
    # dict format -- { node : previous }
    explored = {}
    
    #frontier node format
    # tuple -- ( current node score, (current node, previous node) ) 
    
    #MIDTERM
    # tuple -- ( current node distance score, (current node, previous node, energy score))
    frontier = PriorityQueue()
    
    # node = (0, (start,None))
    node = (0, (start,None,0))
    energy = 0
    frontier.append(node)
    
    while True:       
        
        try:
            node = frontier.pop()
        except:
            return []
        
        testee = node[1]
        explored[testee[0]] = testee[1]
        
        #goal test
        if testee[0] == goal and testee[2] <= 365: # testee[2] added for midterm
            
            score_final = node[0]
            energy_final = testee[2]
            
            best_path = [goal]
            
            while explored[best_path[-1]] != None:
                best_path.append(explored[best_path[-1]]) #adds "prev node" to list
            best_path.reverse()
            return best_path, score_final, energy_final
        
        #get frontier list
        # hideyho = graph.neighbors(testee[0])
        hideyho = graph[testee[0]]
               
        # while True:
        #     try:
                
        #         # city_dict format: {city:iter[(neighbor, distance, energy), (neighbor, distance,energy)]}
                
        #         #get the next frontier
        #         q = next(hideyho) 
        #         score = node[0] + graph.get_edge_weight(testee[0], q)
                
                
        #         #write node to traceback dictionary
        #         try:
        #             test_node = frontier.get_node(q)
                    
        #             if score < test_node[0]:
        #                 frontier.remove(test_node)
        #                 frontier.append((score, (q, testee[0])))
                
        #         except:
        #             if q not in explored:
        #                 frontier.append((score, (q, testee[0])))
            
                               
        #     except StopIteration:
        #         break
        
        for neighbor in hideyho:
              
            # city_dict format: {city:iter[(neighbor, distance, energy), (neighbor, distance,energy)]}
            
            score = node[0] + neighbor[1]
            energy = node[1][2] + neighbor[2]
            if neighbor[0] == 'SEA' or neighbor[0] == 'SNF' or neighbor[0] == 'SND' or neighbor[0] == 'PTD':
                print(neighbor[0], ' ',energy)
            
            #write node to traceback dictionary
            try:
                test_node = frontier.get_node(neighbor[0])
                
                if score < test_node[0]:
                    frontier.remove(test_node)
                    frontier.append((score, (neighbor[0], testee[0], energy)))
            
            except:
                if neighbor[0] not in explored:
                    frontier.append((score, (neighbor[0], testee[0], energy)))
            
                               
               
        
    
    
        
    
    



def null_heuristic(graph, v, goal):
    """
    Null heuristic used as a base line.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """
    Warm-up exercise: Implement the euclidean distance heuristic.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node
    
    graph.nodes[n]['pos'] - returns (x,y) position of node
    
      
    """
    
    
    
    
    if type(v)==str:
        origin = graph.nodes[v]['pos']
    else:
        origin = v['pos']
    
    
    if type(goal) == str:
        dest = graph.nodes[goal]['pos']
        
    else:
        dest = goal['pos']
    
    dist = ((dest[1]-origin[1])**2 + (dest[0]-origin[0])**2)**0.5
    
    return int(dist)
    
    
    
    


def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Warm-up exercise: Implement A* algorithm.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    
    
    Cut and paste of UCS, then add heuristic
    
    """
    
    if start == goal:
        return []
    
    #dictionary to trace back to origin
    # dict format -- node : (raw score, predecessor)
    found_nodes = {}
    found_nodes[start]=None
    
    #frontier node format
    # tuple -- (current node score, current node)
    search_tree = PriorityQueue()
    
    score = 0
    origin = (score,start)
    
        
    while origin[1] != goal:       
        
        hideyho = graph.neighbors(origin[1]) #get frontier iter() list thingy
        
         
        while True:
            try:
                q = next(hideyho) #gets the next frontier
                score = origin[0] + graph.get_edge_weight(origin[1], q)
                added_dist = heuristic(graph,q,goal)
                               
                #write node to traceback dictionary
                if q not in found_nodes:
                    found_nodes[q]=(score,origin[1])
                    search_tree.append((score+added_dist,q))
                    
                 
                elif found_nodes[q] != None and score < found_nodes[q][0]:
                    search_tree.append((score+added_dist,q))
                    search_tree.remove((found_nodes[q][0]+added_dist, q))
                    found_nodes[q]=(score,origin[1])
            
                        
            except StopIteration:
                break
                    
        origin = search_tree.pop()
        
        #take out the heuristic score
        whyyy = origin[0] - heuristic(graph,origin[1],goal)
        pernt = origin[1]
        origin = (whyyy,pernt)
    
    
    
    if goal in found_nodes:
        
        
        best_path = [goal]
        while found_nodes[best_path[-1]] != None:
            best_path.append(found_nodes[best_path[-1]][1])
            
        
        best_path.reverse()
        return best_path
    
    else:
        return[]



def bidirectional_ucs(graph, start, goal):
    """
    Exercise 1: Bidirectional Search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    #see attempt #2 from ucs
    
    if start == goal:
        return []
    
    #dictionary to trace back to origin
    # dict format -- node : frontier node format
    explored_start = {}
    explored_goal = {}
    
    #frontier node format
    # tuple -- (current node score, (current node, previous node))
    frontier_start = PriorityQueue()
    frontier_goal = PriorityQueue()
    
    node = (0, (start,None))
      
    frontier_start.append(node)
    frontier_goal.append((0,(goal,None)))
    
    #flag for end condition
    ira_pohl = 0
    
    #flag for alternating beginnings
    current_path = -1
    
    while True:       
        
        try:
            if current_path<0:
                node = frontier_start.pop()
                explored_start[node[1][0]] = node
                # print(node, ' : explored_start')
                
                if node[1][0] in explored_goal:
                    ira_pohl = 1
            else:
                node = frontier_goal.pop()
                explored_goal[node[1][0]] = node
                # print(node, ' : explored_goal')
                
                if node[1][0] in explored_start:
                    ira_pohl=1
        except:
            return []
        
       
        # IRA POHL check
        # reference: https://docs.google.com/document/d/14Wr2SeRKDXFGdD-qNrBpXjW8INCGIfiAoJ0UkZaLWto/pub
        # reference: Pohl, Ira. "Bi-Directional Search", Thomas J. Watson Research Center, IBM Corp.
        
        while ira_pohl == 1:
            
            #check_list_x format -- node : score
            
            check_list_goal = {}
            check_list_start = {}
            
            if current_path < 0: #means last loop was for goal lists, explored set of this vs union of start
               
                #loops through nodes
                for goal_node in explored_goal.values():
                    
                    nood = goal_node[1][0]
                    
                    if nood in explored_start or frontier_start.__contains__(nood):
                        check_list_goal[nood] = goal_node[0]
                        
                        if nood in explored_start:
                            check_list_start[nood] = explored_start[nood][0]
                        else:
                            explored_start[nood] = frontier_start.get_node(nood)
                            check_list_start[nood] = frontier_start.get_node(nood)[0]
                
            else: #last loop was for start, explore union of goals
                
                for goal_node in explored_start.values():
                    
                    nood = goal_node[1][0]
                    
                    if nood in explored_goal or frontier_goal.__contains__(nood):
                        check_list_start[nood] = goal_node[0]
                        
                        if nood in explored_goal:
                            check_list_goal[nood] = explored_goal[nood][0]
                        else:
                            explored_goal[nood] = frontier_goal.get_node(nood)
                            check_list_goal[nood] = frontier_goal.get_node(nood)[0]
                
            
            if len(check_list_goal.keys()) == 0 or len(check_list_start.keys()) == 0:
                break
                
            locost= float('inf')
            locost_node = ''
            for val in check_list_goal:
                #print('start:',val in explored_start,' goal:', val in explored_goal)
                try:
                    chaching = check_list_goal[val] + check_list_start[val]
                    if chaching < locost:
                        locost = chaching
                        locost_node = val
                except:
                    bwamp=0
            
            
            best_path_start = [locost_node]
            best_path_goal = [locost_node]
            
            while explored_start[best_path_start[-1]][1][1] != None:
                best_path_start.append(explored_start[best_path_start[-1]][1][1])
                
            
            while explored_goal[best_path_goal[-1]][1][1] != None:
                best_path_goal.append(explored_goal[best_path_goal[-1]][1][1])
                
                
        
        
            best_path_start.reverse()
            best_path = best_path_start + best_path_goal[1:]
            #print(best_path)
            return best_path    
                    
        
        
        
        
        #MEAT N POTATOES
        
        #dummy
        testee=node[1]
        
        #get frontier list
        hideyho = graph.neighbors(testee[0]) 
           
        while True:
            try:
                
                #get the next frontier
                q = next(hideyho) 
                score = node[0] + graph.get_edge_weight(testee[0], q)
                
                # print(q,' ', score)
                
                #write node to traceback dictionary
                if current_path<0: #start origin
                    
                    try:
                        test_node = frontier_start.get_node(q)
                        
                        if score < test_node[0]:
                            frontier_start.remove(test_node)
                            frontier_start.append((score, (q, testee[0])))
                    
                    except:
                        if q not in explored_start:
                            frontier_start.append((score, (q, testee[0])))
                
                else: #goal origin
                    
                    try:
                        test_node = frontier_goal.get_node(q)
                        
                        if score < test_node[0]:
                            frontier_goal.remove(test_node)
                            frontier_goal.append((score, (q, testee[0])))
                    
                    except:
                        if q not in explored_goal:
                            frontier_goal.append((score, (q, testee[0])))
                 
            except StopIteration:
                break
            
        current_path *= (-1)
    





def bidirectional_a_star(graph, start, goal,
                         heuristic=euclidean_dist_heuristic):
    """
    Exercise 2: Bidirectional A*.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    #copy bi-directional UCS, add heuristic.
    
    if start == goal:
        return []
    
    #dictionary to trace back to origin
    # dict format -- node : frontier node format
    explored_start = {}
    explored_goal = {}
    
    #frontier node format
    # tuple -- (current node score, (current node, previous node))
    frontier_start = PriorityQueue()
    frontier_goal = PriorityQueue()
    
    node = (0, (start,None))
      
    frontier_start.append(node)
    frontier_goal.append((0,(goal,None)))
    
    #flag for end condition
    ira_pohl = 0
    
    #flag for alternating beginnings
    current_path = -1
    
    while True:       
        
        try:
            if current_path<0:
                node = frontier_start.pop()
                
                #take out the heuristic score
                whyyy = node[0] - heuristic(graph, node[1][0], goal)
                pernt = node[1]
                node = (whyyy,pernt)
                
                explored_start[node[1][0]] = node
                # print(node, ' : explored_start')
                
                if node[1][0] in explored_goal:
                    ira_pohl = 1
            else:
                node = frontier_goal.pop()
                
                #take out the heuristic score
                whyyy = node[0] - heuristic(graph, node[1][0], start)
                pernt = node[1]
                node = (whyyy,pernt)
                
                explored_goal[node[1][0]] = node
                # print(node, ' : explored_goal')
                
                if node[1][0] in explored_start:
                    ira_pohl=1
        except:
            return []
        
        
        # IRA POHL check
        # reference: https://docs.google.com/document/d/14Wr2SeRKDXFGdD-qNrBpXjW8INCGIfiAoJ0UkZaLWto/pub
        # reference: Pohl, Ira. "Bi-Directional Search", Thomas J. Watson Research Center, IBM Corp.
        
        while ira_pohl == 1:
            
            #check_list_x format -- node : score
            
            check_list_goal = {}
            check_list_start = {}
            
            if current_path < 0: #means last loop was for goal lists, explored set of this vs union of start
               
                #loops through nodes
                for goal_node in explored_goal.values():
                    
                    nood = goal_node[1][0]
                    
                    if nood in explored_start or frontier_start.__contains__(nood):
                        check_list_goal[nood] = goal_node[0]
                        
                        if nood in explored_start:
                            check_list_start[nood] = explored_start[nood][0]
                        else:
                            #like pulling teeth, including breaking it down
                            huge_node = frontier_start.get_node(nood)
                            whyyy = huge_node[0] - heuristic(graph, huge_node[1][0], goal)
                            pernt = huge_node[1]
                            huge_node = (whyyy,pernt)
                                                        
                            explored_start[nood] = huge_node
                            check_list_start[nood] = huge_node[0]
                
            else: #last loop was for start, explore union of goals
                
                for goal_node in explored_start.values():
                    
                    nood = goal_node[1][0]
                    
                    if nood in explored_goal or frontier_goal.__contains__(nood):
                        check_list_start[nood] = goal_node[0]
                        
                        if nood in explored_goal:
                            check_list_goal[nood] = explored_goal[nood][0]
                        else:
                            #OMFG
                            huge_node = frontier_goal.get_node(nood)
                            whyyy = huge_node[0] - heuristic(graph, huge_node[1][0], start)
                            pernt = huge_node[1]
                            huge_node = (whyyy,pernt)
                            
                            
                            explored_goal[nood] = huge_node
                            check_list_goal[nood] = huge_node[0]
                
            
            if len(check_list_goal.keys()) == 0 or len(check_list_start.keys()) == 0:
                break
                
            locost= float('inf')
            locost_node = ''
            for val in check_list_goal:
                #print('start:',val in explored_start,' goal:', val in explored_goal)
                try:
                    chaching = check_list_goal[val] + check_list_start[val]
                    if chaching < locost:
                        locost = chaching
                        locost_node = val
                except:
                    bwamp=0
            
            
            best_path_start = [locost_node]
            best_path_goal = [locost_node]
            
            while explored_start[best_path_start[-1]][1][1] != None:
                best_path_start.append(explored_start[best_path_start[-1]][1][1])
                
            
            while explored_goal[best_path_goal[-1]][1][1] != None:
                best_path_goal.append(explored_goal[best_path_goal[-1]][1][1])
                
                
        
        
            best_path_start.reverse()
            best_path = best_path_start + best_path_goal[1:]
            #print(best_path)
            return best_path    
                    
        
        
        
        
        #MEAT N POTATOES
        
        #dummy
        testee=node[1]
        
        #get frontier list
        hideyho = graph.neighbors(testee[0]) 
           
        while True:
            try:
                
                #get the next frontier
                q = next(hideyho) 
                score = node[0] + graph.get_edge_weight(testee[0], q)
                
                # print(q,' ', score)
                
                #write node to traceback dictionary
                if current_path<0: #start origin
                    
                    added_dist = heuristic(graph,q,goal)
                    
                    try:
                        test_node = frontier_start.get_node(q)
                        
                        if score < (test_node[0] - added_dist):
                            frontier_start.remove(test_node)
                            frontier_start.append((score + added_dist, (q, testee[0])))
                    
                    except:
                        if q not in explored_start:
                            frontier_start.append((score + added_dist, (q, testee[0])))
                
                else: #goal origin
                    
                    added_dist = heuristic(graph,q,start)
                    
                    try:
                        test_node = frontier_goal.get_node(q)
                        
                        if score < (test_node[0] - added_dist):
                            frontier_goal.remove(test_node)
                            frontier_goal.append((score + added_dist, (q, testee[0])))
                    
                    except:
                        if q not in explored_goal:
                            frontier_goal.append((score + added_dist, (q, testee[0])))
            
            except StopIteration:
                break
            
        current_path *= (-1)


def tridirectional_search(graph, goals):
    """
    Exercise 3: Tridirectional UCS Search
        
    See README.MD for exercise description.
    
    Based on uniform_cost_search() above, and bi-directional UCS
    Make into  n-dimensional search, specifically 2 or 3 but maybe expandable
    Depends on end state complexity, need to analyze different numbers of nodes
    
    what if a goal is on the path to somewhere else?
    this may be a dumb question, and accounted for by finding shortest paths
    
    =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    
    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    
    """
    
    # problem children 
    goals_tup = copy.copy(goals)
    goals=list(goals)
    
    # all duplified
    if goals.count(goals[0]) == len(goals):
        return []
    
    # check for and eliminate repeats
    cheggitout = copy.copy(goals)
    for single in goals:
        while cheggitout.count(single) > 1:
            cheggitout.remove(single)
    goals = cheggitout
    
    
    # DECLARATIONS
    
    # node format (tuple) -- (current node score, (current node, previous node) )
    # explored format (dict) -- { goal[n] : {current node : node format} }
    # frontier format (dict) -- { goal[n] : PriorityQueue( node format ) }
        
    explored_dict = {}
    frontier_dict = {}
    for n in goals:
        explored_dict[n] = {}
        frontier_dict[n] = PriorityQueue()
        frontier_dict[n].append( (0,(n,None)) )
    
    # known_path_dict format (dict) -- { (current,previous) : [score, best path between nodes] }
    # known_path_dict format (dict) -- { (previous,current) : [score, best path between nodes reversed] }
    # used for testing and traced paths in ira pohl function
    
    known_path_dict={}
    
    #flag for end condition etc
    
    num_goals = len(goals)
    ira_pohl = 0
    ira_pohl_list = [n for n in range(1,num_goals+1)]
    
    #start at the beginning
    
    current_path = goals[0]
    check_path = ()
    
    # first pop, not the last. sets up the "lowest cost frontier" method of origin selection at end of function
    
    node = frontier_dict[current_path].pop()
    
    while True:       
        
        
        
        #check to see if the popped node intersects with another path
        
        explored_dict[current_path][node[1][0]] = node
              
        for checkit in goals_tup:
                           
            if checkit != current_path and node[1][0] in explored_dict[checkit] and not ((current_path,checkit) in known_path_dict or (checkit,current_path) in known_path_dict):  #needs to be n dimensional
                check_path = (current_path,checkit)
                ira_pohl += 1
                break
        
        
        
        #make damn sure it doesn't expand unnecessary frontier nodes
        #flag for ira pohl goal removal due to both paths from node being found
        #buried at the end of ira_pohl and before the neighbor call
        goal_removed_check = 0
        
        
        
        # IRA POHL check
        
        # When an explored node appears in another list, it triggers this check function, delineated by the 'if' statement below
        
        # reference: https://docs.google.com/document/d/14Wr2SeRKDXFGdD-qNrBpXjW8INCGIfiAoJ0UkZaLWto/pub
        # reference: Pohl, Ira. "Bi-Directional Search", Thomas J. Watson Research Center, IBM Corp.
        
                     
        if ira_pohl in ira_pohl_list: # known_path_dict handles len(goals)-1 paths
            
            ira_pohl_list.remove(ira_pohl) # runs only once per found intersection
            
            #check_lists format (dict) -- {goal[n]: {node : score} }
            check_list_dict={}           
                        
            for n in check_path:
                check_list_dict[n]={}
            
            current = check_path[0]
            previous = check_path[1]
            
            for goal_node in explored_dict[previous].values():
                
                nood = goal_node[1][0]
                    
                if nood in explored_dict[current] or frontier_dict[current].__contains__(nood):
                    check_list_dict[previous][nood] = goal_node[0]
                    
                    if nood in explored_dict[current]:
                        check_list_dict[current][nood] = explored_dict[current][nood][0]
                    else:
                        explored_dict[current][nood] = frontier_dict[current].get_node(nood)
                        check_list_dict[current][nood] = frontier_dict[current].get_node(nood)[0]
            
            
            
            
            # break out of IRA POHL if the lists are empty  
            if len(check_list_dict[previous].keys()) != 0 and len(check_list_dict[current].keys()) != 0:
                    
                
                    
                #get the lowest score middle node
                
                locost= float('inf')
                locost_node = ''
                for val in check_list_dict[previous]:
                    try:
                        chaching = check_list_dict[previous][val] + check_list_dict[current][val]
                        if chaching < locost:
                            locost = chaching
                            locost_node = val
                    except:
                        bwamp=0
                
                
                best_path_current = [locost_node]
                best_path_previous = [locost_node]
                
                
                #populate the path from middle
                
                while explored_dict[current][best_path_current[-1]][1][1] != None:
                    best_path_current.append(explored_dict[current][best_path_current[-1]][1][1])
                   
                while explored_dict[previous][best_path_previous[-1]][1][1] != None:
                    best_path_previous.append(explored_dict[previous][best_path_previous[-1]][1][1])
                    
                
                # generate path scores
                                
                score_current = explored_dict[current][best_path_current[0]][0]
                score_previous = explored_dict[previous][best_path_previous[0]][0]
                score_path = score_current + score_previous
                
                
                   
                #stitch paths together
                
                known_path = []
                reversi = []
                
                best_path_current.reverse()
                known_path = best_path_current + best_path_previous[1:]
                reversi = copy.copy(known_path)
                reversi.reverse()
                
                
                known_path.insert(0,score_path)
                reversi.insert(0,score_path)
                
                
                
                known_path_dict[(current,previous)] = known_path
                known_path_dict[(previous,current)] = reversi
                
                # removes a goal if both paths to that goal are found
                # only if there are other paths to find
                
                if len(goals)>2:
                    goalie = copy.copy(goals)
                    for i in goals:
                        chount = 0
                        for j in known_path_dict.keys():
                            if i in j:
                                chount+=1
                        if chount == 4:
                            
                            
                            #also at end of frontier mods below and before ira pohl
                            
                            if i == current_path:
                                goal_removed_check = 1
                            
                            goalie.remove(i)
                    goals = goalie
                    
                    
                
                
                #END STATE TEST - check if it's time to return the node list
                
                if num_goals == 2:
                               
                    return known_path[1:]
                
                elif ira_pohl == num_goals:
                    
                    
                    #list of lists of paths
                    #set up for sorting the best ones out using the score added at the beginning
                                       
                    all_path_lists = []
                    for silly in known_path_dict.values():
                        all_path_lists.append(silly) 
                    all_path_lists.sort()
                    all_path_lists = all_path_lists[:4]
                    
                    for paths in all_path_lists:
                        paths.pop(0)
                    
                    
                    for r in all_path_lists:
                        for s in all_path_lists:
                            if r[-1] == s[0] and r[0] != s[-1]: 
                                best_path = r + s[1:]
                            elif r[0] == s[-1] and r[-1] != s[0]:
                                best_path = s + r[1:]
                                
                                return best_path 
                    
                    
                    
                                
                        
                    
                    
                    
        
        if goal_removed_check == 0:
            
            #MEAT N POTATOES - construct the frontier and heapify
            
            #dummy
            testee=node[1]
            
            #get frontier list
            hideyho = graph.neighbors(testee[0]) 
            
            while True:
                try:
                    
                    #get the next frontier
                    q = next(hideyho) 
                    score = node[0] + graph.get_edge_weight(testee[0], q)
                    
                    
                    #write node to traceback dictionary
                    try:
                        test_node = frontier_dict[current_path].get_node(q)
                        
                        if score < test_node[0]:
                            frontier_dict[current_path].remove(test_node)
                            frontier_dict[current_path].append((score, (q, testee[0])))
                    except:
                        if q not in explored_dict[current_path]:
                            frontier_dict[current_path].append((score, (q, testee[0])))
                    
                     
                except StopIteration:
                    break
        
        
        #choose the next node path to check. lowest score priority
        
        node_list_temp = []
        next_node = ()
        bwamp=0
        for goooal in goals:
                
            try:
                node_temp = frontier_dict[goooal].pop()
            except:
                
                bwamp=1
            
            node_list_temp.append((node_temp[0], goooal, node_temp))
        
        node_list_temp.sort()
        next_node = node_list_temp.pop(0)
                  
        
        current_path = next_node[1]
        node = next_node[2]
          
        #then append the remaining nodes back to their respective frontiers
        
        for v in node_list_temp:
            
            frontier_dict[v[1]].append(v[2])
        
         
        
        
    
    
    


def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic, landmarks=None):
    """
    Exercise 4: Upgraded Tridirectional Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.
        landmarks: Iterable containing landmarks pre-computed in compute_landmarks()
            Default: None

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    
    
    # problem children 
    goals_tup = copy.copy(goals)
    goals=list(goals)
    
    # all duplified
    if goals.count(goals[0]) == len(goals):
        return []
    
    # check for and eliminate repeats
    goal_temp = copy.copy(goals)
    for single in goals:
        while goal_temp.count(single) > 1:
            goal_temp.remove(single)
    goals = copy.copy(goal_temp)
    
    
    # DECLARATIONS
    
    # node format (tuple) -- (current node score + min(heuristic), (current node, previous node, (ranked heuristic tuple) ) )
    # ranked heuristic tuple -- ( (min heuristic, current goal, heuristic goal),...
    #                              (max heuristic, current goal, heuristic goal) )
    
    # note: ranked heuristic is sorted and modified as list, then converted to tuple
    
    # explored format (dict) -- { goal[n] : {current node : node format} }
    # frontier format (dict) -- { goal[n] : PriorityQueue( node format ) }
        
    explored_dict = {}
    frontier_dict = {}
    
    
    #start_list (list) -- used to determine which goal to being search from
    start_list = []
    
    #compute landmarks
    #landmarks format (list) -- [(landmark,{every node : score from landmark})...x4]
    
    
    # landmarks = 1
    
    landmark_list=[]
    landmark_dict={}
    if landmarks != None:
        landmark_list = compute_landmarks(graph)
        for g in landmark_list:
            landmark_dict[g[0]]=g[1]
    
    
    
    for n in goals:
        explored_dict[n] = {}
        frontier_dict[n] = PriorityQueue()
        
        #build heuristic list
        goal_temp = copy.copy(goals)
        goal_temp.remove(n)
        heur_list = []
        heur_tuple = ()
        
        
        
        
            
        
        if landmarks == None:
            for m in goal_temp:
                heur_list.append((heuristic(graph,n,m), n, m))
        else:
            for m in goal_temp:
                l_temp = []
                for i in list(landmark_dict.keys()):
                    l_temp.append(abs(landmark_dict[i][m]-landmark_dict[i][n]))
                            
                heur_list.append((max(l_temp), n, m))
        
        
        heur_list.sort()
        
        if landmarks!= None:
            heur_list.reverse()
        
        heur_tuple = tuple(heur_list)
        frontier_dict[n].append( (heur_tuple[0][0], (n, None, copy.copy(heur_tuple)) ) )
        
        start_list.append(heur_list[0])
    
    start_list.sort()
        
    
    # known_path_dict format (dict) -- { (current,previous) : [score, best path between nodes] }
    # known_path_dict format (dict) -- { (previous,current) : [score, best path between nodes reversed] }
    # used for testing and traced paths in ira pohl function
    
    known_path_dict={}
    
    #flag for end condition, useless idiots, etc
    
    num_goals = len(goals)
    ira_pohl = 0
    ira_pohl_list = [n for n in range(1,num_goals+1)]
    
    
    
    
    
    #start at the beginning
    
    current_path = start_list[0][1]
    check_path = ()
    
    
    # first pop, not the last. sets up the "lowest cost frontier" method of origin selection at end of function
    
    node = (0, frontier_dict[current_path].pop()[1])
    
        
    while True:       
        
        
        
        #check to see if the popped node intersects with another path
        
        explored_dict[current_path][node[1][0]] = node
              
        for checkit in goals_tup:
                           
            if checkit != current_path and node[1][0] in explored_dict[checkit] and not ((current_path,checkit) in known_path_dict or (checkit,current_path) in known_path_dict):  #needs to be n dimensional
                check_path = (current_path,checkit)
                ira_pohl += 1
                break
        
        
        
        # IRA POHL check
        
        # When an explored node appears in another list, it triggers this check function, delineated by the 'if' statement below
        
        # reference: https://docs.google.com/document/d/14Wr2SeRKDXFGdD-qNrBpXjW8INCGIfiAoJ0UkZaLWto/pub
        # reference: Pohl, Ira. "Bi-Directional Search", Thomas J. Watson Research Center, IBM Corp.


        
        
        # flag for ira pohl goal removal due to both paths from node being found
        # buried at the end of ira_pohl and before the neighbor call
        goal_removed_check = 0
        
             
        if ira_pohl in ira_pohl_list: # known_path_dict handles len(goals)-1 paths
            
            ira_pohl_list.remove(ira_pohl) # runs only once per found intersection
            
                  
            #check_lists format (dict) -- {goal[n]: {node : score} }
            check_list_dict={}           
                        
            for n in check_path:
                check_list_dict[n]={}
            
            current = check_path[0]
            previous = check_path[1]
            
            for goal_node in explored_dict[previous].values():
                
                nood = goal_node[1][0]
                    
                if nood in explored_dict[current] or frontier_dict[current].__contains__(nood):
                    check_list_dict[previous][nood] = goal_node[0]
                    
                    if nood in explored_dict[current]:
                        check_list_dict[current][nood] = explored_dict[current][nood][0]
                    else:
                        
                        # node format (tuple) -- (current node score + min(heuristic), (current node, previous node, (ranked heuristic tuple) ) )
                        
                        huge_node = frontier_dict[current].get_node(nood)
                        
                        #take out the heuristic score
                        whyyy = huge_node[0] - huge_node[1][2][0][0]
                        pernt = huge_node[1]
                        huge_node = (whyyy,pernt)
                        
                                                    
                        explored_dict[current][nood] = huge_node
                        check_list_dict[current][nood] = huge_node[0]
            
            
            
            # break out of IRA POHL if the lists are empty  
            if len(check_list_dict[previous].keys()) != 0 and len(check_list_dict[current].keys()) != 0:
                
                
                #get the lowest score middle node
                
                locost= float('inf')
                locost_node = ''
                for val in check_list_dict[previous]:
                    try:
                        chaching = check_list_dict[previous][val] + check_list_dict[current][val]
                        if chaching < locost:
                            locost = chaching
                            locost_node = val
                    except:
                        bwamp=0
                
                
                best_path_current = [locost_node]
                best_path_previous = [locost_node]
                
                
                #populate the path from middle
                
                while explored_dict[current][best_path_current[-1]][1][1] != None:
                    best_path_current.append(explored_dict[current][best_path_current[-1]][1][1])
                   
                while explored_dict[previous][best_path_previous[-1]][1][1] != None:
                    best_path_previous.append(explored_dict[previous][best_path_previous[-1]][1][1])
                    
                
                # generate path scores
                                
                score_current = explored_dict[current][best_path_current[0]][0]
                score_previous = explored_dict[previous][best_path_previous[0]][0]
                score_path = score_current + score_previous
                
                
                   
                #stitch paths together
                
                known_path = []
                reversi = []
                
                best_path_current.reverse()
                known_path = best_path_current + best_path_previous[1:]
                reversi = copy.copy(known_path)
                reversi.reverse()
                
                
                known_path.insert(0,score_path)
                reversi.insert(0,score_path)
                
                
                known_path_dict[(current,previous)] = known_path
                known_path_dict[(previous,current)] = reversi
                
                
                # removes a goal if both paths to that goal are found
                # only if there are other paths to find
                
                if len(goals)>2:
                    goalie = copy.copy(goals)
                    for i in goals:
                        chount = 0
                        for j in known_path_dict.keys():
                            if i in j:
                                chount+=1
                        if chount == 4:
                            
                            # checks if the current path is the one removed, to not add extra neighbor call to removed goal
                            if i == current_path:
                                goal_removed_check = 1
                                           
                            goalie.remove(i)
                            
                            # removes heuristic scores for removed goal, then adds the next best one back in and remakes heap
                            #only call for frontiers that haven't been removed
                            for k in goalie:
                                frontier_dict[k].heuristic_mod(i)
                          
                            
                    goals = goalie
                    
                    
                    
                
                
                #END STATE TEST - check if it's time to return the node list
                
                if num_goals == 2:
                               
                    return known_path[1:]
                
                elif ira_pohl == num_goals:
                    
                    #list of lists of paths
                    #set up for sorting the best ones out using the score added at the beginning
                    
                    all_path_lists = []
                    for silly in known_path_dict.values():
                        all_path_lists.append(silly) 
                    all_path_lists.sort()
                    all_path_lists = all_path_lists[:4]
                    
                    for paths in all_path_lists:
                        paths.pop(0)
                    
                    for r in all_path_lists:
                        for s in all_path_lists:
                            if r[-1] == s[0] and r[0] != s[-1]: 
                                best_path = r + s[1:]
                            elif r[0] == s[-1] and r[-1] != s[0]:
                                best_path = s + r[1:]
                                
                                return best_path 
                                
                        
                    
                    
                    
        
        if goal_removed_check == 0:
            
            #MEAT N POTATOES - construct the frontier and heapify
            
            #dummy
            testee=node[1]
            
            #get frontier list
            hideyho = graph.neighbors(testee[0]) 
            
            while True:
                try:
                    
                    #get the next frontier
                    q = next(hideyho) 
                    score = node[0] + graph.get_edge_weight(testee[0], q)
                    
                    
                    #build heuristic list
                    goal_temp = copy.copy(goals)
                    goal_temp.remove(current_path)
                    heur_list = []
                    heur_tuple = ()
                    
                    
                    
                    
                    
                    # for m in goal_temp:
                    #     heur_list.append((heuristic(graph,q,m), current_path, m))
                    
                        
                    # heur_list.sort()
                    # heur_tuple = tuple(heur_list)
                    
                    # added_dist = heur_tuple[0][0]
                    
                    
                    if landmarks == None:
                        for m in goal_temp:
                            heur_list.append((heuristic(graph,n,m), current_path, m))
                    else:
                        for m in goal_temp:
                            l_temp = []
                            for i in list(landmark_dict.keys()):
                                l_temp.append(abs(landmark_dict[i][m]-landmark_dict[i][n]))
                            
                            heur_list.append((max(l_temp), current_path, m))
        
        
                    heur_list.sort()
        
                    if landmarks != None:
                        heur_list.reverse()
                    
                    heur_tuple = tuple(heur_list)
                    
                    added_dist = heur_tuple[0][0]
                    
                    
                    
                    
                    
                    
                    #write node to traceback dictionary
                    try:
                        test_node = frontier_dict[current_path].get_node(q)
                        
                        if score < (test_node[0] - added_dist):
                            frontier_dict[current_path].remove(test_node)
                            frontier_dict[current_path].append((score+added_dist, (q, testee[0], copy.copy(heur_tuple))))
                    except:
                        if q not in explored_dict[current_path]:
                            frontier_dict[current_path].append((score+added_dist, (q, testee[0], copy.copy(heur_tuple))))
                    
                     
                except StopIteration:
                    break
        
        
        
        
        
        #pop the next node from a frontier
        
        
        #choose the next node path to check. lowest score priority
        #node_temp (tuple) -- (score, goal, (node format) )
        
        node_list_temp = []
        
        
        for goooal in goals:
                   
            try:
                node_temp = frontier_dict[goooal].pop()
            except:
                bwamp=0
            
                        
            node_list_temp.append((node_temp[0], goooal, node_temp))
        
        node_list_temp.sort()
        next_node = node_list_temp.pop(0)
        
        current_path = next_node[1]
        node = next_node[2]
        
        
        
        
        #then append the remaining nodes back to their respective frontiers
        
        for v in node_list_temp:
            frontier_dict[v[1]].append(v[2])
        
        node_temp = ()
        
        #subtract heuristic score from popped node
        #find smallest heuristic = shortest path to any goal
               
        #take out the heuristic score
        whyyy = node[0] - node[1][2][0][0]
        pernt = node[1]
        node = (whyyy,pernt)
        
        
        
        
        
        
        
        
        

def return_your_name():
    """Return your name from this function"""
    return ("Richard Walsh Praetorius")


def compute_landmarks(graph):
    """
    Feel free to implement this method for computing landmarks. We will call
    tridirectional_upgraded() with the object returned from this function.

    Args:
        graph (ExplorableGraph): Undirected graph to search.

    Returns:
    List with not more than 4 computed landmarks. 
    """
    
    distances = [(0,'a','a')]
    
    
    node_list = list(graph.nodes)
    random.seed(time.time())
    loopcheck = 0
    
    while loopcheck < len(node_list)*10:
        x=random.choice(node_list)
        y=random.choice(node_list)
           
        distances.append((euclidean_dist_heuristic(graph,x,y),x,y))
        loopcheck+=1
    
    
    distances.sort()
    distances.reverse()
    
    this_list = []
    best_list =[]
       
    for n in distances:
        for l in n:
            if type(l)!=int and l not in this_list:
                this_list.append(l)
    
    
    # landmark format (list) -- [(landmark,{every node : score})...x4]
    
    
    for goal in this_list[:4]:
        dist_dict = {}
        
        for start in node_list:
            # temp_list = []
            # temp_score=0
            # temp_list = a_star(graph,start,goal)
            # for j in range(len(temp_list)-1):
            #     temp_score += graph.get_edge_weight(temp_list[j], temp_list[j+1])
            # dist_dict[start]= temp_score
            
            dist_dict[start]=euclidean_dist_heuristic(graph,start,goal)
            
                
                
                
                
        best_list.append( (goal, copy.copy(dist_dict)) )
    
    return best_list


def custom_heuristic(graph, v, goal):
    """
       Feel free to use this method to try and work with different heuristics and come up with a better search algorithm.
       Args:
           graph (ExplorableGraph): Undirected graph to search.
           v (str): Key for the node to calculate from.
           goal (str): Key for the end node to calculate to.
       Returns:
           Custom heuristic distance between `v` node and `goal` node
       """
    pass


# Extra Credit: Your best search method for the race
def custom_search(graph, start, goal, data=None):
    """
    Race!: Implement your best search algorithm here to compete against the
    other student agents.

    If you implement this function and submit your code to Gradescope, you'll be
    registered for the Race!

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        data :  Data used in the custom search.
            Will be passed your data from load_data(graph).
            Default: None.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError


def load_data(graph, time_left):
    """
    Feel free to implement this method. We'll call it only once 
    at the beginning of the Race, and we'll pass the output to your custom_search function.
    graph: a networkx graph
    time_left: function you can call to keep track of your remaining time.
        usage: time_left() returns the time left in milliseconds.
        the max time will be 10 minutes.

    * To get a list of nodes, use graph.nodes()
    * To get node neighbors, use graph.neighbors(node)
    * To get edge weight, use graph.get_edge_weight(node1, node2)
    """

    # nodes = graph.nodes()
    return None
 
 
def haversine_dist_heuristic(graph, v, goal):
    """
    Note: This provided heuristic is for the Atlanta race.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Haversine distance between `v` node and `goal` node
    """

    #Load latitude and longitude coordinates in radians:
    vLatLong = (math.radians(graph.nodes[v]["pos"][0]), math.radians(graph.nodes[v]["pos"][1]))
    goalLatLong = (math.radians(graph.nodes[goal]["pos"][0]), math.radians(graph.nodes[goal]["pos"][1]))

    #Now we want to execute portions of the formula:
    constOutFront = 2*6371 #Radius of Earth is 6,371 kilometers
    term1InSqrt = (math.sin((goalLatLong[0]-vLatLong[0])/2))**2 #First term inside sqrt
    term2InSqrt = math.cos(vLatLong[0])*math.cos(goalLatLong[0])*((math.sin((goalLatLong[1]-vLatLong[1])/2))**2) #Second term
    return constOutFront*math.asin(math.sqrt(term1InSqrt+term2InSqrt)) #Straight application of formula
