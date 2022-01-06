import sys, os
sys.path.append(os.path.join(os.getcwd(), "inference"))
from table_inference import *
import gurobipy as gp
from gurobipy import GRB
from auxil_ftns import *

from evaluation import *

sys.path.append(os.path.join(os.getcwd(), "code_commons"))
from config import scale_lscore

def save_result(niter, vman, filename, DIR, scale, img_save=True, xml_save=True, test_scan=False, tab_loc=None, img_write=None):
    result_dict = vman.get_inference_results()

    """
    #final_extra_labels, final_extra_location = \
    #            junction_squeeze( final_extra_labels, final_extra_location)
    """
    num_extra_candidates = vman.num_extra_nodes

    # print results
    indices = [v for v in range(len(result_dict["candidate_labels"]))]
#    print([v for v in zip(indices, result_dict["candidate_labels"])] )
#    print([v for v in zip(indices, result_dict["candidate_location"])] )
#    print( result_dict["extra_labels"] )
#    print( result_dict["extra_location"] )

    labels = result_dict["labels"]
    locations = result_dict["junction_location"]
    if tab_loc is not None and len(locations)>0:
        x1 = int(min(tab_loc[:,0])*scale/2)
        y1 = int(min(tab_loc[:,1])*scale/2)
        locations[:,0] += x1
        locations[:,1] += y1
    horz_adj = result_dict["horz_adj"]
    vert_adj = result_dict["vert_adj"]
    names = result_dict["names"]
    nNodes = locations.shape[0]

    if img_write is None:
        image = draw_table( nNodes, locations, labels, horz_adj, vert_adj, vman.input_image, names)
    else:
        image = draw_table( nNodes, locations, labels, horz_adj, vert_adj, img_write, names)

#    filename_ = filename.split('/')[-1].rstrip('.npy')
#    FOLDER = './test' #/'+filename_
#    if not os.path.exists(FOLDER):
#        os.makedirs(FOLDER)
    filename_ = filename+'_iter'+str(niter)+'_ext'+str(num_extra_candidates)
    if img_save:
        cv2.imwrite(DIR+'/table_%s.jpg' % filename_, image)
        print("save result in %s" % (DIR+'/table_%s.jpg' % filename_))

#    print(locations)
    for i in range(len(locations)):
        loc = locations[i]
        locations[i] = [int(round(loc[0]/scale*2)), int(round(loc[1]/scale*2))]
#    print(locations)

    if xml_save:
        write_xmlfile(filename, nNodes, locations, labels, horz_adj, vert_adj, DIR, test_scan=test_scan, tab_loc=tab_loc)

    return result_dict["extra_labels"], image

def prime_optimization(vman, gpm, niter_th=3, save_inter_result=False, DIR='', filename='', scale=1.):
    num_real_candidates = vman.num_candidates
    num_extra_candidates = vman.num_extra_nodes

    # Set objectives
    obj = 0
    for i in range(num_real_candidates + num_extra_candidates):
        obj =  obj + vman.get_node_score( i )

    for i in range(num_real_candidates + num_extra_candidates):
#        obj = obj + 2 * (0.5 * vman.get_link_score( i )) # recognition
#        obj = obj + (0.5 * vman.get_link_score( i )) # detection
        obj = obj + scale_lscore * (0.5 * vman.get_link_score( i ))


    vman.add_single_link_consts()
    gpm.setObjective(obj, GRB.MAXIMIZE)


    # optimize
    # re-optimize when line-intersections are detected
    bIntersections = True
    niter = 0
    while niter<=niter_th and bIntersections == True:
        gpm.update()
        gpm.optimize()

        active_edges = vman.get_active_edges( )
        intersections = intersection_checks( active_edges )

        if len(intersections) == 0:
            bIntersections = False
        else:
            for elements in intersections:
                vman.add_linkpair_constrs( elements[0], elements[1] )
        print(obj.getValue())

        if not os.path.exists(os.path.join(DIR, 'inter')):
            os.makedirs(os.path.join(DIR, 'inter'))
        save_result(niter, vman, filename, os.path.join(DIR, 'inter'), scale, save_inter_result, False) # temp, for ablation
        niter += 1
    extra_labels, _ = save_result(niter, vman, filename, os.path.join(DIR, 'inter'), scale, save_inter_result, False)

    print('Obj: %g, iter: %d' % (obj.getValue(), niter))

    return extra_labels

def extra_pos_optimization(vman, gpm):
    num_real_candidates = vman.num_candidates
    # Set objectives
    pos_obj = 0.
    for v in gpm.getVars():
        if v.x == 1:
            rst = vman.decoding( v.varName )
            if rst[0] == 'edge':
                if rst[1] >= num_real_candidates or rst[2] >= num_real_candidates:
                    _, _, pos1 = vman.get_node( rst[1] )
                    _, _, pos2 = vman.get_node( rst[2] )
                    if rst[3] == 'E' or rst[3] == 'W':
                        pos_obj = pos_obj + ( pos1[1] - pos2[1]) * ( pos1[1] - pos2[1])
                    else:
                        pos_obj = pos_obj + ( pos1[0] - pos2[0]) * ( pos1[0] - pos2[0])
    vman.set_constrs_except_extra_pos()

    # Optimize
    gpm.setObjective(pos_obj, GRB.MINIMIZE)
    gpm.optimize()

    return pos_obj


