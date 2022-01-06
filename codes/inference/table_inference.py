import numpy as np
import gurobipy as gp
from gurobipy import GRB
from auxil_ftns import *
import math

sys.path.append(os.path.join(os.getcwd(), "code_commons"))
#sys.path.append(os.path.join(os.getcwd(), '../code_commons'))
from config import lscore_extra, nscore_regul, ep_dist, mu_d


class variable_manager:
    def __init__(self, gpm, data_table, candidate_location, \
                horz_adjacency_matrix, horz_adjacency_score, vert_adjacency_matrix, \
                    vert_adjacency_score, num_extra_nodes = 0, \
                    H=100, W=100, img=None, test_scan=False):
        self.gpm = gpm
        self.num_labels = 9

        self.num_candidates = candidate_location.shape[0]
        self.num_extra_nodes = num_extra_nodes
        self.num_nodes = self.num_candidates + num_extra_nodes

        margin = 100
        self.x_min = max(0, np.min( candidate_location[:,0] ) - margin)
        self.x_max = np.max( candidate_location[:,0] ) + margin

        self.y_min = max(0, np.min( candidate_location[:,1] ) - margin)
        self.y_max = np.max( candidate_location[:,1] ) + margin

        self.H, self.W = H, W
        self.input_image = img

        self.test_scan = test_scan

        if num_extra_nodes == 0:
            self.data_table = data_table
            self.candidate_location = candidate_location
            self.horz_adjacency_matrix = horz_adjacency_matrix
            self.horz_adjacency_score = horz_adjacency_score
            self.vert_adjacency_matrix = vert_adjacency_matrix
            self.vert_adjacency_score = vert_adjacency_score
        else:
            self.data_table = -1.0 * np.ones( shape=[self.num_nodes, self.num_labels], dtype = data_table.dtype)
            self.data_table[:self.num_candidates,:] = data_table

            self.candidate_location = np.zeros( shape=[self.num_nodes, 2], dtype = candidate_location.dtype)
            self.candidate_location[:self.num_candidates,:] = candidate_location

            self.horz_adjacency_matrix = np.ones( shape=[self.num_nodes, self.num_nodes], dtype = horz_adjacency_matrix.dtype )
            np.fill_diagonal( self.horz_adjacency_matrix, 0 )
            self.horz_adjacency_matrix[:self.num_candidates,:self.num_candidates] = horz_adjacency_matrix

            self.vert_adjacency_matrix = np.ones( shape=[self.num_nodes, self.num_nodes], dtype = horz_adjacency_matrix.dtype )
            np.fill_diagonal( self.vert_adjacency_matrix, 0 )
            self.vert_adjacency_matrix[:self.num_candidates,:self.num_candidates] = vert_adjacency_matrix

#            self.horz_adjacency_score = 0.15 * np.ones( shape=[self.num_nodes, self.num_nodes], dtype = horz_adjacency_score.dtype ) # recognition
#            self.horz_adjacency_score = 0. * np.ones( shape=[self.num_nodes, self.num_nodes], dtype = horz_adjacency_score.dtype ) # detection
            self.horz_adjacency_score = lscore_extra * np.ones( shape=[self.num_nodes, self.num_nodes], dtype = horz_adjacency_score.dtype ) # detection
            self.horz_adjacency_score[:self.num_candidates,:self.num_candidates] = horz_adjacency_score

#            self.vert_adjacency_score = 0.15 * np.ones( shape=[self.num_nodes, self.num_nodes], dtype = vert_adjacency_score.dtype ) # recognition
#            self.vert_adjacency_score = 0. * np.ones( shape=[self.num_nodes, self.num_nodes], dtype = vert_adjacency_score.dtype ) # detection
            self.vert_adjacency_score = lscore_extra * np.ones( shape=[self.num_nodes, self.num_nodes], dtype = vert_adjacency_score.dtype ) # detection
            self.vert_adjacency_score[:self.num_candidates,:self.num_candidates] = vert_adjacency_score


        self.my_dict = {}

    @staticmethod
    def decoding(str):
        l = []
        for x in str.split('#'):
            if x.isdigit():
                l.append( int(x) )
            else:
                l.append( x )
        return l


    @staticmethod
    def get_names(elements):
        str = elements[0]
        for e in elements[1:]:
            if isinstance( e, int ) is True:
                tmp = '{:04d}'.format(e)
            else:
                tmp = e

            str = str + '#' + tmp
        return str

    def get_node(self, num, create = True ):
        name = self.get_names( ['node', num ] )
        if name in self.my_dict.keys():
            return self.my_dict[name]
        elif create == False:
            return None
        else:
            e = self.gpm.addVar(vtype=GRB.BINARY, name = name+"#e")
            w = self.gpm.addVar(vtype=GRB.BINARY, name = name+"#w")
            s = self.gpm.addVar(vtype=GRB.BINARY, name = name+"#s")
            n = self.gpm.addVar(vtype=GRB.BINARY, name = name+"#n")

            m = self.gpm
            m0 = m.addVar(vtype=GRB.BINARY, name=name+"#90000#x")

            m5 = m.addVar(vtype=GRB.BINARY, name=name+"#90101#8")
            m6 = m.addVar(vtype=GRB.BINARY, name=name+"#90110#2")
            m7 = m.addVar(vtype=GRB.BINARY, name=name+"#90111#5")

            m9 = m.addVar(vtype=GRB.BINARY, name=name+"#91001#6")
            m10 = m.addVar(vtype=GRB.BINARY, name=name+"#91010#0")
            m11 = m.addVar(vtype=GRB.BINARY, name=name+"#91011#3")
            m13 = m.addVar(vtype=GRB.BINARY, name=name+"#91101#7")
            m14 = m.addVar(vtype=GRB.BINARY, name=name+"#91110#1")
            m15 = m.addVar(vtype=GRB.BINARY, name=name+"#91111#4")
#            if num>=self.num_candidates:
#                m0.ub=1
#                m0.lb=1
            m.addConstr( e == m9 + m10 + m11 + m13 + m14 + m15, name+'#0' )
            m.addConstr( w == m5 + m6 + m7 + m13 + m14 + m15 , name+'#1')
            m.addConstr( s == m6 + m7 + m10 + m11 + m14 + m15 , name+'#2')
            m.addConstr( n == m5 + m7 + m9 + m11 + m13 + m15 , name+'#3')

            m.addConstr( 1 == m0 + m5 + m6 + m7 + m9 + m10 + m11 + m13 + m14 + m15 , name+'#4')
#            m.addConstr( (m0) * (e+w+s+n) == 0 , name+'#5')

            if num >= self.num_candidates:
                x = self.gpm.addVar( self.x_min, self.x_max, vtype=GRB.CONTINUOUS, name=name+"#X")
                y = self.gpm.addVar( self.y_min, self.y_max, vtype=GRB.CONTINUOUS, name=name+"#Y")
            else:
                x = self.candidate_location[num,0]
                y = self.candidate_location[num,1]


            variables = [ [e, w, s, n ], [m10, m14, m6, m11, m15, m7, m9, m13, m5, m0 ], [x,y] ]
            self.my_dict[name] = variables
            return self.my_dict[name]

    def get_edge(self, node1, node2, directions, create = True  ):
        """ add edge to self.my_dict, a link between node1-node2 in a specific direction """
        #if len(directions) == 1:
        rst = []
        for direction in directions:
            edge_name = self.get_names( ['edge', node1, node2, direction ] ) # edge#0001#0002#E
            if edge_name in self.my_dict.keys():
                rst.append( self.my_dict[edge_name] )
            else:
                m = self.gpm
                if create is True:
                    l = m.addVar(vtype=GRB.BINARY, name=edge_name  )
                    self.my_dict[edge_name] = l
                else:
                    l = None
                rst.append( l )

        return rst

    def get_node_score(self,num):
        """ get Vp """
        ewsn, labels, _ = self.get_node(num)

        dt = self.data_table[num]
        ics = invalid_config_score = -np.max(dt) if num<self.num_candidates else 0
        obj = 0
        for i in range(self.num_labels):
            obj = obj + dt[i] * labels[i]
#            obj = obj - 0.2 * labels[i] # regulazation # table detection
            obj = obj - nscore_regul * labels[i] # regulazation # table detection

        obj = obj + ics * labels[self.num_labels] # ics*m0

        return obj

    def add_single_link_consts(self):
        num_nodes = self.num_nodes
        """
        a = self.get_node(0, create=False)
        b = self.get_edge(0,1, directions='EWSN', create=False)
        c = self.get_edge(1,0, directions='EWSN', create=False)
        b, c
        """
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                ewsn1 = self.get_edge(i,j, directions='EWSN', create=False)
                ewsn2 = self.get_edge(j,i, directions='EWSN', create=False)
                name = self.get_names(['edge', i, j])

                if ewsn1[0] != None:
                    self.gpm.addConstr(  ewsn1[0] == ewsn2[1] , name+'#0')
                if ewsn1[1] != None:
                    self.gpm.addConstr(  ewsn1[1] == ewsn2[0] , name+'#1')
                if ewsn1[2] != None:
                    self.gpm.addConstr(  ewsn1[2] == ewsn2[3] , name+'#2')
                if ewsn1[3] != None:
                    self.gpm.addConstr(  ewsn1[3] == ewsn2[2] , name+'#3')

                ewsni = [i for i in ewsn1 if i]
                ewsnj = [i for i in ewsn2 if i]

                if len(ewsni) != 0:
                    self.gpm.addConstr( gp.quicksum( e for e in ewsni ) <= 1, name+'#4')
                if len(ewsnj) != 0:
                    self.gpm.addConstr( gp.quicksum( e for e in ewsnj ) <= 1, name+'#5')

    def add_linkpair_constrs( self, link1, link2 ):
        """ add constraint to link1, link2 which have intersection """
        ewsn1 = self.get_edge( link1[0], link1[1], directions='EWSN', create=False)
        ewsn2 = self.get_edge( link2[0], link2[1], directions='EWSN', create=False)

        ewsni = [i for i in ewsn1 if i]
        ewsnj = [i for i in ewsn2 if i]

        self.gpm.addConstr( gp.quicksum( e for e in ewsni ) * gp.quicksum( e for e in ewsnj ) == 0) #Inactivate at least one link (link1 or link2)

    def get_link_score( self, num ):
        """ Get pairwise objective function for node num """
        ha = self.horz_adjacency_matrix
        hs = self.horz_adjacency_score
        va = self.vert_adjacency_matrix
        vs = self.vert_adjacency_score

        num_nodes = ha.shape[0]

        h_neighbor = ha[ num ]
        v_neighbor = va[ num ]

        E = []
        W = []
        S = []
        N = []

        Ei = []
        Wi = []
        Si = []
        Ni = []

        for i in range(num_nodes):
            if h_neighbor[i] == 1:
                e, w = self.get_edge( num, i, directions='EW')
                if e is not None:
                    E.append( e )
                    Ei.append(i)
                if w is not None:
                    W.append( w )
                    Wi.append(i)

            if v_neighbor[i] == 1:
                s, n = self.get_edge( num, i, directions= 'SN')
                if s is not None:
                    S.append( s )
                    Si.append(i)
                if n is not None:
                    N.append( n )
                    Ni.append(i)

        ewsn, _, pos = self.get_node( num )
        name = self.get_names(['node', num])

        self.gpm.addConstr( ewsn[0] == gp.quicksum( e for e in E ), name+'#e')
        self.gpm.addConstr( ewsn[1] == gp.quicksum( w for w in W ), name+'#w')
        self.gpm.addConstr( ewsn[2] == gp.quicksum( s for s in S ), name+'#s')
        self.gpm.addConstr( ewsn[3] == gp.quicksum( n for n in N ), name+'#n')

#        calculate_ep = max(np.max(self.horz_adjacency_score), np.max(self.vert_adjacency_score))
#        ep_obj = np.max(self.data_table)/calculate_ep
#        print(ep_obj)
        ep = ep_dist if num < self.num_candidates else 0.
        obj = 0
        total_num = 0
        horz_e = np.zeros(len(E))
        horz_w = np.zeros(len(W))
        vert_s = np.zeros(len(S))
        vert_n = np.zeros(len(N))
        i = 0
        for e, idx in zip(E,Ei):
            horz_e[i] = abs(self.candidate_location[idx][0] - self.candidate_location[num][0])
            horz_dist = (self.candidate_location[idx][0]-self.candidate_location[num][0])/(self.x_max - self.x_min)
            vert_dist = (self.candidate_location[idx][1]-self.candidate_location[num][1])/(self.y_max - self.y_min)
            dist =  math.sqrt(horz_dist**2 + vert_dist**2)
            if idx < self.num_candidates:
                obj = obj + e * hs[num, idx ] - e*ep*dist
            else:
                obj = obj + e * hs[num, idx]
            i += 1
        i = 0
        for w, idx in zip(W,Wi):
            horz_w[i] = abs(self.candidate_location[num][0] - self.candidate_location[idx][0])
            horz_dist = (self.candidate_location[idx][0]-self.candidate_location[num][0])/(self.x_max - self.x_min) #*2
            vert_dist = (self.candidate_location[idx][1]-self.candidate_location[num][1])/(self.y_max - self.y_min) #*2
            #dist =  horz_dist**2 + vert_dist**2
            dist = math.sqrt(horz_dist**2 + vert_dist**2)
            if idx < self.num_candidates:
                obj = obj + w * hs[num, idx ] - w*ep*dist
            else:
                obj = obj + w * hs[num, idx]
            i += 1
        i = 0
        for s, idx in zip(S,Si):
            vert_s[i] = abs(self.candidate_location[idx][1] - self.candidate_location[num][1])
            horz_dist = (self.candidate_location[idx][0]-self.candidate_location[num][0])/(self.x_max - self.x_min) #*2
            vert_dist = (self.candidate_location[idx][1]-self.candidate_location[num][1])/(self.y_max - self.y_min) #*2
            #dist =  horz_dist**2 + vert_dist**2 #math.sqrt(horz_dist**2 + vert_dist**2)
            dist = math.sqrt(horz_dist**2 + vert_dist**2)
            if idx < self.num_candidates:
                obj = obj + s * vs[num, idx ] - s*ep*dist
            else:
                obj = obj + s * vs[num, idx]
            i += 1
        i = 0
        for n, idx in zip(N,Ni):
            vert_n[i] = abs(self.candidate_location[num][1] - self.candidate_location[idx][1])
            horz_dist = (self.candidate_location[idx][0]-self.candidate_location[num][0])/(self.x_max - self.x_min) #*2
            vert_dist = (self.candidate_location[idx][1]-self.candidate_location[num][1])/(self.y_max - self.y_min) #*2
            #dist =  horz_dist**2 + vert_dist**2 #math.sqrt(horz_dist**2 + vert_dist**2)
            dist = math.sqrt(horz_dist**2 + vert_dist**2)
            if idx < self.num_candidates:
                obj = obj + n * vs[num, idx ] - n*ep*dist
            else:
                obj = obj + n * vs[num, idx]
            i += 1

#        delta = 12#self.character_height-2 #12
#        epsilon = 10#self.character_height-4 #10
        if self.test_scan:
            delta = mu_d #14 #mu_d #self.character_height-2 #12 ctdar19 #mu
            epsilon = 10 #mu_s#self.character_height-4 #10 ctdar19 #lambda
        else:
            delta = mu_d #14
            epsilon = 6

        for e, idx in zip(E, Ei):
            _, _, e_pos = self.get_node( idx ) #[x,y]
            self.gpm.addConstr( ( e_pos[0] - pos[0] - delta ) * e >= 0  )
            if num >= self.num_candidates or idx >= self.num_candidates:
                self.gpm.addConstr( -epsilon*e <= e*(e_pos[1] - pos[1] ) )
                self.gpm.addConstr( e*(e_pos[1] - pos[1] - epsilon ) <= 0 )

        for w, idx in zip(W, Wi):
            _, _, w_pos = self.get_node( idx )
            self.gpm.addConstr( (  pos[0]  - w_pos[0] - delta) * w >= 0  )
            if num >= self.num_candidates or idx >= self.num_candidates:
                self.gpm.addConstr(  epsilon*w >= w*(w_pos[1] - pos[1]) )
                self.gpm.addConstr( w*(w_pos[1] - pos[1] + epsilon) >= 0 )

        for s, idx in zip(S, Si):
            _, _, s_pos = self.get_node( idx )
            self.gpm.addConstr( ( s_pos[1] - pos[1] - delta ) * s >= 0  )
            if num >= self.num_candidates or idx >= self.num_candidates:
                self.gpm.addConstr( -epsilon*s <= s*(s_pos[0] - pos[0] ) )
                self.gpm.addConstr( s*(s_pos[0] - pos[0] - epsilon) <= 0 )

        for n, idx in zip(N, Ni):
            _, _,  n_pos = self.get_node( idx )
            self.gpm.addConstr( (  pos[1]  - n_pos[1] - delta) * n >= 0  )
            if num >= self.num_candidates or idx >= self.num_candidates:
                self.gpm.addConstr(  epsilon*n >= n*(n_pos[0] - pos[0]) )
                self.gpm.addConstr( n*(n_pos[0] - pos[0] + epsilon) >= 0 )

        return obj


    def get_active_edges( self ):
        gpm = self.gpm
        active_edges = []
        num_real_candidates = self.num_candidates
        for v in gpm.getVars():
            if v.x == 1:
                #print('%s %g' % (v.varName, v.x)
                rst = self.decoding( v.varName )
                if rst[0] == 'edge':
                    node1 = rst[1]
                    node2 = rst[2]

                    if node1 > node2:
                        continue # work only on l_pq when p=<q
                    if node1 < num_real_candidates:
                        pos1 = self.candidate_location[ node1 ]
                    else:
                        _, _, pos = self.get_node( node1 )
                        pos1 = [pos[0].x, pos[1].x ]

                    if node2 < num_real_candidates:
                        pos2 = self.candidate_location[ node2 ]
                    else:
                        _, _, pos = self.get_node( node2 )
                        pos2 = [pos[0].x, pos[1].x ]

                    active_edges.append( [ node1, node2, pos1, pos2] )
        return active_edges

    def set_constrs_except_extra_pos(self):
        Vars = self.gpm.getVars()
        for v in Vars:
            if 'X' in v.varName or 'Y' in v.varName:
                continue
            value = v.x
            v.lb = value
            v.ub = value

    def remove_extra_nodes(self, obj):
        Vars = self.gpm.getVars()
        num_real_candidates = self.num_candidates
        self.num_extra_nodes = 0

        for v in Vars:
            rst = self.decoding(v.varName)
            if rst[0]=='node':
                if rst[1] >= self.num_candidates: # extra node!
                    self.gpm.remove(v)
                    obj.remove(v)
            else: #'edge'
                if rst[1] >= self.num_candidates or rst[2] >= self.num_candidates:
                    self.gpm.remove(v)
                    obj.remove(v)
        return obj


    def get_inference_results( self ):
        final_candidate_labels = []
        final_candidate_location = []
        final_extra_labels = []
        final_extra_location = []
        final_candidate_names = [] # for visualization
        num_real_candidates = self.num_candidates
        num_extra_candidates = self.num_extra_nodes
        for i in range(num_real_candidates + num_extra_candidates):
            _, junction_types, pos = self.get_node(i)
            values = np.array( [ x.x for x in junction_types], dtype = np.float32 )
            if i < num_real_candidates:
                #if np.argmax( values ) < num_labels:
                final_candidate_labels.append( np.argmax( values ) )
                final_candidate_location.append( np.array( self.candidate_location[i], dtype = np.float32) )
                final_candidate_names.append(i)
            else:
                #if np.argmax( values ) < num_labels:
                final_extra_labels.append( np.argmax( values ) )
                final_extra_location.append( np.array( [pos[0].x, pos[1].x], dtype = np.float32) )
                final_candidate_names.append(i)

        final_extra_location = np.array( final_extra_location )
        final_candidate_location = np.array( final_candidate_location )


        counter = 0
        labels = []
        locations = []
        node_map = {}

        for i in range(num_real_candidates + num_extra_candidates):
            _, junction_types, pos = self.get_node(i)
            values = np.array( [ x.x for x in junction_types], dtype = np.float32 ) ##original-np.int32
            if i < num_real_candidates:
                label = np.argmax( values )
                #print( i, "-th node at", candidate_location[i]," type", label )
                if  label >= 0 and label < self.num_labels:
                    locations.append( self.candidate_location[i]  )
                    labels.append( label )
                    node_map[ str(i) ] = counter
                    counter = counter+1
            else:
                label = final_extra_labels[ i - num_real_candidates]
                #print( i, "-th node at", pos[0].x, pos[1].x, " type", label )
                if label >= 0 and label < self.num_labels:
                    locations.append( np.array([pos[0].x, pos[1].x], dtype = np.float32)  )
                    if label != 0:
                        labels.append( -label )
                    else:
                        labels.append(-0.1)
                    node_map[ str(i) ] = counter
                    counter = counter + 1

        horz_adj = np.zeros( shape = [counter,counter], dtype=np.int32 )
        vert_adj = np.zeros( shape = [counter,counter], dtype=np.int32 )


        for v in self.gpm.getVars():
            if v.x >0.9:
                #print('%s %g' % (v.varName, v.x))
                rst = self.decoding( v.varName )
                if rst[0] == 'node':# and len(rst) == 4:
                    #print( rst[1], rst[3] )
                    #print(rst)
                    pass
                if rst[0] == 'edge':
                    key1 = str(rst[1])
                    key2 = str(rst[2])
                    if key1 in node_map.keys() and key2 in node_map.keys():
                        idx1 = node_map[ key1 ]
                        idx2 = node_map[ key2 ]
                        if rst[-1] == 'E' or rst[-1] == 'W':
                            horz_adj[idx1,idx2] = horz_adj[idx2,idx1] = 1
                            #print( rst )
                        if rst[-1] == 'S' or rst[-1] == 'N':
                            vert_adj[idx1,idx2] = vert_adj[idx2,idx1] = 1
                            #print( rst )
        """"""

        locations = np.array( locations )
        labels = np.array( labels )


        dict = {}
        dict["candidate_labels"] = final_candidate_labels
        dict["candidate_location"] =final_candidate_location
        dict["extra_labels"] = final_extra_labels
        dict["extra_location"] = final_extra_location
        dict["horz_adj"] = horz_adj
        dict["vert_adj"] = vert_adj
        dict["junction_location"] = locations
        dict["labels"] = labels
        dict["names"] = final_candidate_names

        return dict



def junction_squeeze( x, y ):
    thres = 15
    newx = []
    newy = []
    length = len(x)
    for i in range(length):
        for j in range(i+1,length):
            pt1 = y[i]
            pt2 = y[j]
            if np.sum( np.square( pt1 - pt2 ) ) < thres:
                bRedundant = ((x[i] == 8 and x[j] == 0) or  (x[i] == 0 and x[j] == 8) \
                    or (x[i] == 2 and x[j] == 6) or  (x[i] == 6 and x[j] == 2))
                if  bRedundant == True:
                    x[i] = 4
                    x[j] = -1

                    print('****************************')

        """
        if x[i] >= 0:
            newx.append( x[i] )
            newy.append( y[i] )
        """
    return x, y
    return newx, newy
