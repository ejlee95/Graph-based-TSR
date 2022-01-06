import numpy as np
import cv2
import sys
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from matplotlib import pyplot as plt
from copy import deepcopy

sys.path.append(os.path.join(os.getcwd(), "code_commons"))
#sys.path.append(os.path.join(os.getcwd(), '../code_commons'))
from config import th_conf, rad_conf, mu_d

SHAPES = ['ro', 'gv', 'bs', 'r>', 'yP', 'b<', 'mD', 'y^', 'cp']

def draw_table( nNodes, locations, labels, horz_adj, vert_adj, img, names=None):
    H,W,_ = img.shape
    H = H//2
    W = W//2
    canvas1 = cv2.resize(img, (2*W, 2*H))
    image = np.zeros( shape=[2*H+20,2*W+50,3], dtype = np.uint8 )
    canvas = np.zeros_like(image)
    canvas[5:5+2*H, 20:20+2*W, :] = canvas1
    if names is None:
        names = ['' for x in range(nNodes)]

    for i in range(nNodes):
        for j in range(nNodes):
            if horz_adj[i,j] == 1:
                color = (191,223,169)
#                color = (np.random.uniform(100,255),np.random.uniform(100,255),np.random.uniform(100,255))
                cv2.line( image, (int(2*locations[i,0]+20), int(2*locations[i,1]+5)), \
                        (int(2*locations[j,0]+20), int(2*locations[j,1]+5)), color, thickness=2)
            if vert_adj[i,j] == 1:
                color = (160,215,250)
#                color = (np.random.uniform(100,255),np.random.uniform(100,255),np.random.uniform(100,255))
                cv2.line( image, (int(2*locations[i,0]+20), int(2*locations[i,1]+5)), \
                        (int(2*locations[j,0]+20), int(2*locations[j,1]+5)), color, thickness=2 )


    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range( nNodes ):
        if labels[i] < 0:
            color = (255,0,0)
        else:
            color = (0,255,255)
        lab = int(round(labels[i]))
        pt = (int(2*locations[i,0]+20), int(2*locations[i,1]+5))
        cv2.circle( image, pt, 5, color, thickness=-1 ) #9
#        pos_str = str(i) + "," + str(np.abs(lab)) + "," + str(names[i])
        pos_str = str(np.abs(lab))
        cv2.putText(image, pos_str,  pt, font, .4, (0,0,255), 2, cv2.LINE_AA) #.6

    result = cv2.addWeighted(canvas, 0.2, image, 0.8, 20)

    if result.shape[0]>1400 or result.shape[1]>1400:
        scale = 1400./max(result.shape[0], result.shape[1])
        result = cv2.resize(result, (int(result.shape[1]*scale), int(result.shape[0]*scale)), interpolation=cv2.INTER_CUBIC)

    return result #image

def get_candidates(heatmap, idx, thres=0.3, radius=10):
    """ get junctions candidates from heatmap
        heatmap: BxHxWx11 (0:9 = junction heatmaps, 10:12 = x,y-affinity)
        idx: reference heatmap index, -1=0:9
        thres: initial threshold value
        radius: ? """
    # sum(heatmap)- comparison standard(?)
    if idx<0:
#        cur_feature_map = np.max(heatmap[:,:,:9], axis=-1) # junction heatmap
        cur_feature_map = np.sum(heatmap[:,:,:9], axis=-1) # junction heatmap
    else:
        cur_feature_map = heatmap[:,:,idx]

    # initial candidates - thresholding
    pos = np.where(cur_feature_map > thres)
    max_indicate = np.ones_like(cur_feature_map)

    R = radius//2

    # second thresholding with smaller thres
    while pos[0].shape[0] == 0 and thres > 0.05:
        thres = thres*0.8
        pos = np.where(cur_feature_map > thres)

    # sorted(in decreasing order) candidates
    icandidates = []
    for i in range(pos[0].shape[0]):
        y,x = pos[0][i], pos[1][i]
        z = cur_feature_map[y,x]
        icandidates.append([z,y,x])
    icandidates.sort(reverse=True)  # decreasing order

    # candidates - remove nearby candidates ; square-shape
    candidates = []
    for cand in icandidates:
        z,y,x = cand[0], cand[1], cand[2]
        if max_indicate[y,x] > 0:
            candidates.append([z,y,x])
            max_indicate[y-R:y+R+1, x-R:x+R+1] = 0
    candidates.sort(reverse=True)
    candidates = np.array(candidates)
    length = candidates.shape[0]

    # candidates, removing duplicates with radius
    for i in range(length):
        for j in range(i+1, length):
            dist = np.abs(candidates[i,1] - candidates[j,1]) + np.abs(candidates[i,2] - candidates[j,2])
            if dist < radius: # diamond-shape
                candidates[j,:] = 0

    # remove redundant elements(0)
    new_candidate = []
    for i in range(length):
        if candidates[i,0] > thres:
            new_candidate.append(candidates[i,:])
    new_candidate = np.array(new_candidate)

    return new_candidate

def nms(heatmap, image=None, filename=None, save_result_image=False):
    new_cand = get_candidates(heatmap, -1, th_conf, rad_conf)

    if len(new_cand) == 0:
        return None, None
    pos = new_cand[:,1:3]
    pos = np.asarray(pos)
    pos = pos.astype(np.int32)

    heatmap_cand = heatmap[[j for j in pos[:,0]], [k for k in pos[:,1]], :]  # N,9
    types = np.argmax(heatmap, axis=-1)

    if save_result_image:
        img_show = image.copy()
        if img_show.shape[0]>1400 or img_show.shape[1]>1400:
            scale = 1400./max(img_show.shape[0], img_show.shape[1])
            img_show = cv2.resize(img_show, (int(img_show.shape[1]*scale), int(img_show.shape[0]*scale)), interpolation=cv2.INTER_CUBIC)

        rate = min(img_show.shape[0]/heatmap.shape[0], img_show.shape[1]/heatmap.shape[1])
        dpi=200
        fig_size = (img_show.shape[1]/dpi, img_show.shape[0]/dpi)
        fig = plt.figure(dpi=dpi, figsize=fig_size)
        img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
        plt.imshow(img_show, interpolation='lanczos')
        for j in range(len(new_cand)):
            y,x = int(new_cand[j][1]), int(new_cand[j][2])
            shape = SHAPES[types[y,x]]
            plt.plot(int(rate*x), int(rate*y), shape, markersize=6) #13) #30
        plt.subplots_adjust(left=0.0001, right=0.9999, top=0.9999, bottom=0.0001)
        plt.xticks([]); plt.yticks([])
#        plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
#        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        plt.clf()

    # pos = [x,y]
    temp = np.copy(pos)
    pos[:,0] = temp[:,1]
    pos[:,1] = temp[:,0]

    # # to evaluate junction detection
    # scale = image.shape[0]/heatmap.shape[0]
    # np.save(filename.replace('.jpg', '.npy').replace('.png', '.npy'), pos*scale)

    return heatmap_cand, pos

def get_adjacency_matrix(candidate_location, test_scan=False):
    """ return X/Y-direction conneting candidates
        Ix: (Ix)ij = 1 : the (j-point) is horizontally connected to the (i-point), in range of -2~2
                     0 : the (j-point) and the (i-point) are not in range of -2~2
        Iy: (Iy)ij = 1 : the (j-point) is vertically connected to the (i-point), in range of 88~92
                     0 : the (j-point) and the (i-point) are not in range of 88~92 """
    y = [p[1] for p in candidate_location]
    x = [p[0] for p in candidate_location]
    N = len(x)

    # just right, left, above, under positions
    X = np.asarray([x-xx for xx in x])
    Y = np.asarray([y-yy for yy in y])
#    Right = (X>5).astype('int')
#    Left = (X<(-5)).astype('int')
#    Above = (Y<5).astype('int')
#    Under = (Y>(-5)).astype('int')
    Right = (X>mu_d).astype('int') #x is right from xx
    Left = (X<(-mu_d)).astype('int')
    Above = (Y<(-mu_d)).astype('int')
    Under = (Y>mu_d).astype('int')


    # in -2~2, 88~92 range
#    th = np.tan(4*np.pi/180)
#
#    eps = 1e-15
#    X_degree = np.asarray([(y-yy)/(x-xx+eps) for xx,yy in zip(x,y)])
#    Y_degree = np.asarray([(x-xx)/(y-yy+eps) for xx,yy in zip(x,y)])
#    Ind_X = (X_degree <= th) * (X_degree >= -th)
#    Ind_Y = (Y_degree <= th) * (Y_degree >= -th)

#    epsilon = 10 #margin #20
#    margin_adj = 6 #margin #20 #ctdar19
    X_dif = np.asarray([y-yy for yy in y])
    Y_dif = np.asarray([x-xx for xx in x])
    if test_scan:
        mu_s = 10
    else:
        mu_s = 6
    Ind_X = abs(X_dif) <= mu_s
    Ind_Y = abs(Y_dif) <= mu_s

    Right *= Ind_X
    Left *= Ind_X
    Above *= Ind_Y
    Under *= Ind_Y

    horz_adjacency_matrix = (Right+Left).astype(np.int32)
    vert_adjacency_matrix = (Above+Under).astype(np.int32)

    return horz_adjacency_matrix, vert_adjacency_matrix

def get_adjacency_score_matrix(candidate_location, adjacency_matrix, adjacency_heatmap, direction):
    N = len(candidate_location)
    adjacency_heatmap_for_cal = adjacency_heatmap.copy()
    adjacency_score = np.zeros_like(adjacency_matrix, np.float32)

    for i in range(N):
        for j in range(i,N):
            if adjacency_matrix[i][j]==1:
                px, py = candidate_location[i].astype(np.int32)
                qx, qy = candidate_location[j].astype(np.int32)
                adjacency_heatmap_for_cal[py][px] = -1
                adjacency_heatmap_for_cal[qy][qx] = -1

    for i in range(N):
        for j in range(i,N):
            if adjacency_matrix[i][j]==1:
                px, py = candidate_location[i].astype(np.int32)
                qx, qy = candidate_location[j].astype(np.int32)
                if direction == "horz":
                    ratio = (qy-py)/(qx-px)
                    step = (qx>=px).astype(np.int32)
                    step = int((step-0.5)*2)
                    values = [adjacency_heatmap_for_cal[int(py+k*ratio*step)][px+k*step] for k in range(1,abs(qx-px))]
                    score = sum(np.asarray(values))/len(values)
                    adjacency_score[i][j] = adjacency_score[j][i] = score
                else:
                    ratio = (qx-px)/(qy-py)
                    step = (qy>=py).astype(np.int32)
                    step = int((step-0.5)*2)
                    values = [adjacency_heatmap_for_cal[py+k*step][int(px+k*ratio*step)] for k in range(1, abs(qy-py))]
                    score = sum(np.asarray(values))/len(values)
                    adjacency_score[i][j] = adjacency_score[j][i] = score

    return adjacency_score

def get_test_vector_from_path(heatmap, path, DIR='', img_type='.jpg', test_scan=False):
    """ return data in defined format from path(heatmap numpy file) """
#    heatmap = np.load(path+'.npy')
    num_labels = heatmap.shape[2]-2
    image = cv2.imread(path+img_type)
    image_plot = image.copy()
    image = cv2.resize(image, (heatmap.shape[1], heatmap.shape[0]))
    filename = path.split('/')[-1]

    data_table, candidate_location = nms(heatmap[:,:,0:-2], image=image_plot, filename=DIR+'/nms_'+filename+'.jpg', save_result_image=True)
    if data_table is None and candidate_location is None:
        return None, None, None, None, None, None
    candidate_location = candidate_location.astype(np.float32)
    num_real_candidates = candidate_location.shape[0]

    horz_adjacency_matrix, vert_adjacency_matrix = get_adjacency_matrix(candidate_location, test_scan=test_scan)
    horz_adjacency_score = get_adjacency_score_matrix(candidate_location, horz_adjacency_matrix, heatmap[:,:,-2], "horz")
    vert_adjacency_score = get_adjacency_score_matrix(candidate_location, vert_adjacency_matrix, heatmap[:,:,-1], "vert")

    return data_table, candidate_location, horz_adjacency_matrix, horz_adjacency_score, \
        vert_adjacency_matrix, vert_adjacency_score

def get_test_vector():
    num_labels = 9
    num_real_candidates = 10

    num_candidates = num_real_candidates

    data_table = np.zeros( shape = [num_real_candidates, num_labels], dtype = np.float32 )
    candidate_location = np.zeros( shape = [num_real_candidates, 2], dtype = np.float32 )

    horz_adjacency_matrix = np.zeros( shape = [num_real_candidates,num_real_candidates], dtype = np.int32 )
    horz_adjacency_score = np.ones_like( horz_adjacency_matrix )

    vert_adjacency_matrix = np.zeros( shape = [num_real_candidates,num_real_candidates], dtype = np.int32 )
    vert_adjacency_score = np.ones_like( vert_adjacency_matrix )

    # test vector
    data_table[1:,:] = np.eye(9)
    data_table = data_table + 0.1 *  np.random.normal( size = data_table.shape )
    data_table = np.clip ( data_table, 0, 1 )
    #data_table[0,:] = -1


    candidate_location[0] = [0,0]
    candidate_location[1] = [100,100]
    candidate_location[2] = [200,100]
    candidate_location[3] = [300,100]
    candidate_location[4] = [100,200]
    candidate_location[5] = [200,200]
    candidate_location[6] = [300,200]
    candidate_location[7] = [100,300]
    candidate_location[8] = [200,300]
    candidate_location[9] = [300,300]

    candidate_location = candidate_location + np.array( [200,200], dtype = np.float32)

    horz_adjacency_matrix = np.zeros_like( horz_adjacency_matrix )
    horz_adjacency_score = np.ones_like( horz_adjacency_score )

    vert_adjacency_matrix = np.zeros_like( vert_adjacency_matrix )
    vert_adjacency_score = np.ones_like( vert_adjacency_score )

    horz_adjacency_matrix[0,1] = horz_adjacency_matrix[1,0] = 1
    horz_adjacency_matrix[0,2] = horz_adjacency_matrix[2,0] = 1
    horz_adjacency_matrix[0,3] = horz_adjacency_matrix[3,0] = 1
    horz_adjacency_matrix[1,2] = horz_adjacency_matrix[2,1] = 1
    horz_adjacency_matrix[1,3] = horz_adjacency_matrix[3,1] = 1
    horz_adjacency_matrix[2,3] = horz_adjacency_matrix[3,2] = 1

    #horz_adjacency_matrix[4,5] = horz_adjacency_matrix[5,4] = 1
    horz_adjacency_matrix[4,6] = horz_adjacency_matrix[6,4] = 1
    #horz_adjacency_matrix[5,6] = horz_adjacency_matrix[6,5] = 1

    horz_adjacency_matrix[7,8] = horz_adjacency_matrix[8,7] = 1
    horz_adjacency_matrix[7,9] = horz_adjacency_matrix[9,7] = 1
    horz_adjacency_matrix[8,9] = horz_adjacency_matrix[9,8] = 1

    assert( np.allclose( horz_adjacency_matrix, horz_adjacency_matrix.T ) )

    vert_adjacency_matrix[0,1] = vert_adjacency_matrix[1,0] = 1
    vert_adjacency_matrix[0,4] = vert_adjacency_matrix[4,0] = 1
    vert_adjacency_matrix[0,7] = vert_adjacency_matrix[7,0] = 1
    vert_adjacency_matrix[1,4] = vert_adjacency_matrix[4,1] = 1
    vert_adjacency_matrix[1,7] = vert_adjacency_matrix[7,1] = 1
    vert_adjacency_matrix[4,7] = vert_adjacency_matrix[7,4] = 1


 #   vert_adjacency_matrix[2,5] = vert_adjacency_matrix[5,2] = 1
    vert_adjacency_matrix[2,8] = vert_adjacency_matrix[8,2] = 1
 #   vert_adjacency_matrix[5,8] = vert_adjacency_matrix[8,5] = 1

    vert_adjacency_matrix[3,6] = vert_adjacency_matrix[6,3] = 1
    vert_adjacency_matrix[3,9] = vert_adjacency_matrix[9,3] = 1
    vert_adjacency_matrix[6,9] = vert_adjacency_matrix[9,6] = 1


    assert( np.allclose( vert_adjacency_matrix, vert_adjacency_matrix.T ) )
    return data_table, candidate_location, horz_adjacency_matrix, horz_adjacency_score, \
        vert_adjacency_matrix, vert_adjacency_score


def on_segment(p, q, r):
    if r[0] <= max(p[0], q[0]) and r[0] >= min(p[0], q[0]) and r[1] <= max(p[1], q[1]) and r[1] >= min(p[1], q[1]):
        return True
    return False

def orientation(p, q, r):
    val = ((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1]))
    if val == 0 :
        return 0
    return 1 if val > 0 else -1

def intersects(seg1, seg2):

    p1, q1 = seg1
    p2, q2 = seg2

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment(p1, q1, p2) :
        return True
    if o2 == 0 and on_segment(p1, q1, q2) :
        return True
    if o3 == 0 and on_segment(p2, q2, p1) :
        return True
    if o4 == 0 and on_segment(p2, q2, q1) :
        return True

    return False


def intersection_checks( edges ):
    intesections = []
    length = len( edges )
    for i in range(length):
        for j in range(i+1,length):
            p1 = edges[i][2]
            p2 = edges[i][3]
            q1 = edges[j][2]
            q2 = edges[j][3]


            rst = intersects( [p1,p2], [q1,q2] )
            if rst is True:
                if edges[i][0] != edges[j][0] and edges[i][0] != edges[j][1]: # to prevent regarding edges which share the node(s) as intersections
                    if edges[i][1] != edges[j][0] and edges[i][1] != edges[j][1]:
                        intesections.append( [[edges[i][0], edges[i][1]], [edges[j][0], edges[j][1]] ])
    return intesections
