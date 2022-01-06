import numpy as np
import os
from xml.etree.ElementTree import Element, SubElement, dump
from xml.etree import ElementTree
from collections import OrderedDict
import json

import sys
# sys.path.append(os.path.join(os.getcwd(), "code_commons"))
sys.path.append(os.path.join(os.getcwd(), '../code_commons'))
from config import mu_s
#COUNTS_UP = {0: 0, 1: 0, 2: 0, 3: 1, 4: 2, 5: 1, 6: 1, 7: 2, 8: 1}
#COUNTS_DOWN = {0: 1, 1: 2, 2: 1, 3: 1, 4: 2, 5: 1, 6: 0, 7: 0, 8: 0}
#COUNTS_LEFT = {0: 0, 1: 1, 2: 1, 3: 0, 4: 2, 5: 2, 6: 0, 7: 1, 8: 1}
#COUNTS_RIGHT = {0: 1, 1: 1, 2: 0, 3: 2, 4: 2, 5: 0, 6: 1, 7: 1, 8: 0}

# UDLR
COUNTS = [[0,1,0,1], [0,2,1,1], [0,1,1,0], [1,1,0,2], [2,2,2,2], [1,1,2,0], [1,0,0,1], [2,0,1,1], [1,0,1,0]]

class Output(object):
    def __init__(self, start_location, test_scan=False):
        self.TL_pos = start_location
        self.rowsS_pos = []
        self.rowsE_pos = []
        self.colsS_pos = []
        self.colsE_pos = []
        self.test_scan = test_scan
        self.cell = [] #[starty, startx, TL, TR, BL, BR]

    def add_cell(self, TL, TR, BL, BR):
        startx = min(TL[0], BL[0])
        starty = min(TL[1], TR[1])
        endx = max(BR[0], TR[0])
        endy = max(BR[1], BL[1])

        self.rowsS_pos.append(starty)
        self.rowsE_pos.append(endy)
        self.colsS_pos.append(startx)
        self.colsE_pos.append(endx)

        self.cell.append([starty, startx, endy, endx, TL, BL, BR, TR])

    def indexing_rows_cols(self):
        self.rowsS_pos2id, self.rowsS_id2pos, self.rowsS_proto, self.rowsSNum = self._indexing_lines(self.rowsS_pos)
        self.rowsE_pos2id, self.rowsE_id2pos, self.rowsE_proto, self.rowsENum = self._indexing_lines(self.rowsE_pos)
        self.colsS_pos2id, self.colsS_id2pos, self.colsS_proto, self.colsSNum = self._indexing_lines(self.colsS_pos)
        self.colsE_pos2id, self.colsE_id2pos, self.colsE_proto, self.colsENum = self._indexing_lines(self.colsE_pos)

    def _indexing_lines(self, candidates):
        """ Find rows and cols, giving indices """
        # candidates = [v[0] for v in positions] #x for cols, y for rows
        candidates = list(dict.fromkeys(candidates))
        candidates = sorted(list(set(candidates)))

        pos2id = {}
        id2pos = {}
        prototype = {}

        current_id = 0

        temp_candidates = []
        for i in range(len(candidates)):
            temp_candidates.append(candidates[i])
            if i<len(candidates)-1 and (candidates[i+1] - candidates[i]) > 7: #7 is a hyperparameter
                pos2id[candidates[i]] = current_id
                id2pos[current_id] = temp_candidates
                current_id += 1
                temp_candidates = []
            elif i==len(candidates)-1:
                pos2id[candidates[i]] = current_id
                temp_candidates.append(candidates[i])
                id2pos[current_id] = temp_candidates
            else: # share col_id
                pos2id[candidates[i]] = current_id
                temp_candidates.append(candidates[i])

        for k, v in id2pos.items():
            proto = np.mean(v)
            prototype[k] = proto

        return pos2id, id2pos, prototype, current_id+1

    def indexing_cells(self):
        self.cells = {}
        for cell in self.cell:
            startrow = self.rowsS_pos2id[cell[0]]
            endrow = self.rowsE_pos2id[cell[2]]
            startcol = self.colsS_pos2id[cell[1]]
            endcol = self.colsE_pos2id[cell[3]]

            cell_proto = [np.array([self.colsS_proto[startcol], self.rowsS_proto[startrow]]),\
                          np.array([self.colsS_proto[startcol], self.rowsE_proto[endrow]]),\
                          np.array([self.colsE_proto[endcol], self.rowsE_proto[endrow]]),\
                          np.array([self.colsE_proto[endcol], self.rowsS_proto[startrow]])]

            cellname = 'cell#{:03d}#{:03d}#{:03d}#{:03d}'.format(startrow, startcol, endrow, endcol)

            if self.test_scan:
                self.cells[cellname] = cell[4:]
            else:
                self.cells[cellname] = cell_proto

    def set_TR(self, loc):
        self.TR_pos = loc
    def set_BL(self, loc):
        self.BL_pos = loc
    def set_BR(self, loc):
        self.BR_pos = loc
    def re_set_TL(self, loc):
        self.TL_pos = loc


def _construct_cell(locations, labels, horz_adj, vert_adj, node_id, cell_set, TR_id_prev=None, BL_id_prev=None, BR_id_prev=None):
    TL = locations[node_id]
    TL_type = abs(int(round(labels[node_id])))

    TR_set = 0
    pivot_id = node_id
    while not TR_set:
        TR_ids = np.where(horz_adj[pivot_id]==1)[0].tolist()
        TR_id = [x for x in TR_ids if locations[x][0]>locations[pivot_id][0]]
        if len(TR_id)==0:
            cell_set=0
            TR=-1
            TR_type=-1
            TR_set = 1
        else:
            TR_id = TR_id[0]
            TR_type = abs(int(round(labels[TR_id])))
            if COUNTS[TR_type][1]:
                TR_set = 1
                TR = locations[TR_id]
            else:
                pivot_id = TR_id

    BL_set = 0
    pivot_id = node_id
    while not BL_set:
        BL_ids = np.where(vert_adj[pivot_id]==1)[0].tolist()
        BL_id = [x for x in BL_ids if locations[x][1]>locations[pivot_id][1]]
        if len(BL_id)==0:
            cell_set=0
            BL = -1
            BL_type = -1
            BL_set = 1
        else:
            BL_id = BL_id[0]
            BL_type = abs(int(round(labels[BL_id])))
            if COUNTS[BL_type][3]:
                BL_set = 1
                BL = locations[BL_id]
            else:
                pivot_id = BL_id

    if cell_set:
        BR_horz_set = 0
        pivot_id = BL_id
        while not BR_horz_set:
            BR_ids = np.where(horz_adj[pivot_id]==1)[0].tolist()
            BR_id = [x for x in BR_ids if locations[x][0]>locations[pivot_id][0]]
            if len(BR_id) == 0:
                BR_id_horz = -1
                BR_horz_set = 1
            else:
                BR_id_horz = BR_id[0]
                BR_type_horz = abs(int(round(labels[BR_id_horz])))
                if COUNTS[BR_type_horz][0]:
                    BR_horz_set = 1
                else:
                    pivot_id = BR_id_horz

        BR_vert_set = 0
        pivot_id = TR_id
        while not BR_vert_set:
            BR_ids = np.where(vert_adj[pivot_id]==1)[0].tolist()
            BR_id = [x for x in BR_ids if locations[x][1]>locations[pivot_id][1]]
            if len(BR_id) == 0:
                BR_id_vert = -1
                BR_vert_set = 1
            else:
                BR_id_vert = BR_id[0]
                BR_type_vert = abs(int(round(labels[BR_id_vert])))
                if COUNTS[BR_type_vert][2]:
                    BR_vert_set = 1
                else:
                    pivot_id = BR_id_vert

        if BR_id_horz == -1 and BR_id_vert == -1:
            cell_set = 0
            BR_id = -1
            BR = -1
            BR_type = -1
        else:
            if BR_id_horz == BR_id_vert:
                BR_id = BR_id_horz
                BR = locations[BR_id]
                BR_type = BR_type_horz
            else:
                # print('DOES NOT MATCH')
                cell_set = 0
                BR_id = -1
                BR = -1
                BR_type = -1
    else:
        BR_id = -1
        BR = -1
        BR_type=-1


    if cell_set:
        if not(locations[BR_id][0] > locations[BL_id][0] and locations[BR_id][1] > locations[TR_id][1]):
            print('need this condition!!(BR, cell_set)')
            cell_set = 0
        if abs(locations[BR_id][1]-locations[BL_id][1])>mu_s or abs(locations[BR_id][0]-locations[TR_id][0])>mu_s:
            print('need more finding BR')
            cell_set = 2

    return cell_set, TL, TL_type, TR, TR_id, TR_type, BL, BL_id, BL_type, BR, BR_id, BR_type


def each_table(nNodes, locations, labels, horz_adj, vert_adj, test_scan=False, tab_loc=None):
    # Set start_node      locations=[x,y]
    start_node_candidates_idx = np.where(locations[:,1] <= locations[:,1].min()+10)[0].tolist() #y
    start_node_candidates = locations[start_node_candidates_idx,:]
    start_node_idx = np.argmin(start_node_candidates[:,0]) #x
    start_node = start_node_candidates[start_node_idx]
    output_table = Output(start_node, test_scan)

    # Get all connected nodes
    def collect_nodes(index, nodes, horz_adj, vert_adj, current_nodes):
        # print(indices)
        # connected_nodes_horz = connected_nodes_vert = []
        # for index in indices:
        connected_nodes_horz = np.where(horz_adj[index] == 1)[0].tolist()
        connected_nodes_vert = np.where(vert_adj[index] == 1)[0].tolist()
        if np.sum(horz_adj)==0 and np.sum(vert_adj)==0:
            return nodes
        else:
            connected_nodes = [v for v in connected_nodes_horz] + [v for v in connected_nodes_vert] + [index]
            if len(connected_nodes) == 1:
                return connected_nodes
            remain_nodes = [v for v in nodes if v not in connected_nodes]
            horz_adj[:, index] = 0
            horz_adj[index, :] = 0
            vert_adj[:, index] = 0
            vert_adj[index, :] = 0
            ch = 0
            for next_index in connected_nodes:
                if next_index != index:
                    current_nodes += connected_nodes
                    current_nodes = list(set(current_nodes))
                    current_nodes += collect_nodes(next_index, remain_nodes, horz_adj, vert_adj, current_nodes)
        return current_nodes

    horz_adj_temp = np.copy(horz_adj)
    vert_adj_temp = np.copy(vert_adj)
    whole_start_node_idx = start_node_candidates_idx[start_node_idx]
    whole_nodes = collect_nodes(whole_start_node_idx, [v for v in range(len(locations)) if v!=whole_start_node_idx], horz_adj_temp, vert_adj_temp, [])
    whole_nodes = list(set(whole_nodes))
    if len(whole_nodes) == 1:
        remain_idx = [v for v in range(len(locations)) if v not in whole_nodes]
        nNodes = len(remain_idx)
        locations = locations[remain_idx, :]
        labels = np.asarray([labels[v] for v in remain_idx])
        horz_adj = horz_adj[remain_idx,:]
        horz_adj = horz_adj[:,remain_idx]
        vert_adj = vert_adj[remain_idx,:]
        vert_adj = vert_adj[:,remain_idx]

        return None, nNodes, locations, labels, horz_adj, vert_adj
    # checked = np.ones((nNodes), np.int32) * (-1) #len(whole_nodes)
    checked = np.ones((nNodes, 4), np.int32) * (-1)

    for node_id in whole_nodes:
        cell_set = 1
        if np.sum(checked[node_id] != 0) == 0:
            continue

        cell_set, TL, TL_type, TR, TR_id, TR_type, BL, BL_id, BL_type, BR, BR_id, BR_type = _construct_cell(locations, labels, horz_adj, vert_adj, node_id, cell_set)

        if cell_set:
            output_table.add_cell(TL, TR, BL, BR)

            if np.sum(checked[node_id]) == -4:
                checked[node_id] = COUNTS[TL_type]
            checked[node_id][1] -= 1
            checked[node_id][3] -= 1
            if np.sum(checked[TR_id]) == -4:
                checked[TR_id] = COUNTS[TR_type]
            checked[TR_id][1] -= 1
            checked[TR_id][2] -= 1
            if np.sum(checked[BL_id]) == -4:
                checked[BL_id] = COUNTS[BL_type]
            checked[BL_id][0] -= 1
            checked[BL_id][3] -= 1
            if np.sum(checked[BR_id]) == -4:
                checked[BR_id] = COUNTS[BR_type]
            checked[BR_id][0] -= 1
            checked[BR_id][2] -= 1

    # assert np.sum(checked) == 0 checked>0 checked<0   --looped

    # set TR, BR, BL
    whole_locations = locations[whole_nodes,:]

    def intersection(list1, list2):
        list3 = [v for v in list1 if v in list2]
        return list3

    #### TABLE DETECTION
    if tab_loc is None:
        x1 = min(whole_locations[:,0])
        x2 = max(whole_locations[:,0])
        y1 = min(whole_locations[:,1])
        y2 = max(whole_locations[:,1])

        output_table.re_set_TL(np.array([x1,y1]))
        output_table.set_BL(np.array([x1,y2]))
        output_table.set_BR(np.array([x2,y2]))
        output_table.set_TR(np.array([x2,y1]))
    else:
        output_table.re_set_TL(np.array(tab_loc[0]))
        output_table.set_BL(np.array(tab_loc[1]))
        output_table.set_BR(np.array(tab_loc[2]))
        output_table.set_TR(np.array(tab_loc[3]))

    output_table.indexing_rows_cols()
    output_table.indexing_cells()
    output_table.cells = {k: v for k, v in sorted(output_table.cells.items())}

    # print('rows num: %d' % output_table.rowsNum, output_table.rows_id2pos)
    # print('cols num: %d' % output_table.colsNum, output_table.cols_id2pos)

    if tab_loc is None:
        remain_idx = [v for v in range(len(locations)) if v not in whole_nodes]
        nNodes = len(remain_idx)
        if len(remain_idx) > 0:
            locations = locations[remain_idx, :] #[locations[v] for v in remain_idx]
            labels = np.asarray([labels[v] for v in remain_idx])
            horz_adj = horz_adj[remain_idx,:]
            horz_adj = horz_adj[:,remain_idx]
            vert_adj = vert_adj[remain_idx,:]
            vert_adj = vert_adj[:,remain_idx]
        else:
            locations = labels = []
            horz_adj = vert_adj = np.empty([1,1])
    else:
        remain_idx = []
        nNodes = 0
        locations = labels = []
        horz_adj = vert_adj = np.empty([1,1])

    return output_table, nNodes, locations, labels, horz_adj, vert_adj


def write_xmlfile(filename, nNodes, locations, labels, horz_adj, vert_adj, DIR, test_scan=False, tab_loc=None):
    """ write html file, for evaluation
        """
    output_tables = []
    horz_adj_for_table = np.copy(horz_adj)
    vert_adj_for_table = np.copy(vert_adj)
    locations_for_table = np.copy(locations)
    labels_for_table = np.copy(labels)
    nNodes_for_table = nNodes

    while nNodes > 0:
        output_table, nNodes, locations, labels, horz_adj, vert_adj = each_table(nNodes, locations, labels, horz_adj, vert_adj, test_scan, tab_loc=tab_loc)
        if output_table != None:
            output_tables.append(output_table)

    # Write results in xml form
    xml_filename = DIR+'/str/'+filename+'.xml'
    if tab_loc is None:
        root = Element('document', filename=filename+'.jpg')
    else:
        if os.path.exists(xml_filename):
            root = ElementTree.parse(xml_filename).getroot()
        else:
            root = Element('document', filename=filename+'.jpg')
    for output_table in output_tables:
        table_obj = SubElement(root, 'table')
        positions = [output_table.TL_pos, output_table.BL_pos, output_table.BR_pos, output_table.TR_pos]
#        positions = [locations[output_table.TL_ind], locations[output_table.BL_ind], locations[output_table.BR_ind], locations[output_table.TR_ind]]
        positions = "%d,%d %d,%d %d,%d %d,%d" % (positions[0][0], positions[0][1], positions[1][0], positions[1][1], positions[2][0], positions[2][1], positions[3][0], positions[3][1])
        SubElement(table_obj, 'Coords', points=positions)
        for cell_name in output_table.cells.keys():
            cellname = cell_name.split('#')
            start_row = str(int(cellname[1]))
            start_col = str(int(cellname[2]))
            end_row = str(int(cellname[3]))
            end_col = str(int(cellname[4]))

            positions = output_table.cells[cell_name]
            positions = "%d,%d %d,%d %d,%d %d,%d" % (positions[0][0], positions[0][1], positions[1][0], positions[1][1], positions[2][0], positions[2][1], positions[3][0], positions[3][1])

            cell_obj = SubElement(table_obj, 'cell')
            attrib_dict = OrderedDict()
            attrib_dict['start-row'] = str(start_row)
            attrib_dict['start-col']=str(start_col)
            attrib_dict['end-row']=str(end_row)
            attrib_dict['end-col']=str(end_col)
            cell_obj.attrib = attrib_dict
            SubElement(cell_obj, 'Coords', points=positions)

            # dump(cell_obj)
    indent(root)
    # dump(root)

    et = ElementTree.ElementTree(root)
    et.write(xml_filename, encoding='UTF-8', xml_declaration=True)


def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
         if level and (not elem.tail or not elem.tail.strip()):
             elem.tail = i
