import tensorflow as tf
import numpy as np
import cv2

import gurobipy as gp
from gurobipy import GRB

import sys, os
import time
import input_stream
import json
from datetime import datetime as datetime
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, ElementTree

sys.path.append( os.path.join(os.getcwd(), "code_commons") )
from global_constants import *
from config import CH

sys.path.append(os.path.join(os.getcwd(), "inference"))
from table_inference import *
from auxil_ftns import get_test_vector_from_path
from testcode import *

def run_test( sess, rtnet, FLAGS, mode = 'webcam',  frame_size = None, srcname=None,
                                    wait_time = 1, save_video = False, out_video_file_name = None, fps = 24.0,  \
                                    ckpt_basename = None, save_result_image=True, save_heatmap=False,\
                                    num_extra_candidates = 3, niter_th=3, save_intermediate_optimization=False,
                                    test_scan=False, with_detection=False):

    if save_video is True:
        assert ckpt_basename is not None

    font = cv2.FONT_HERSHEY_SIMPLEX

    if mode == 'webcam':
        frame = input_stream.Frame('webcam')
        if frame_size is not None:
            frame.set_size( frame_size )
        else:
            frame.set_size( (640,480) )
    elif mode == 'folder':
        assert srcname is not None
        frame = input_stream.Frame('folder', srcname)
        if frame_size is not None:
            frame.set_size( frame_size )
    elif mode == 'video':
        assert srcname is not None
        frame = input_stream.Frame( 'video', srcname )
        if frame_size is not None:
            frame.set_size( frame_size )


    if FLAGS.test_name=='':
        now = datetime.now()
        dir_name = "test/"+str(now.year)+"-"+str(now.month)+"-"+str(now.day)+"-"+str(now.hour)+"-"+str(now.minute)
    else:
        dir_name = "test/"+FLAGS.test_name
    if save_result_image:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        if not os.path.exists(dir_name+'/str'): # structure recognition
            os.makedirs(dir_name+'/str')
#        if not os.path.exists(dir_name+'/reg'): # detection
#            os.makedirs(dir_name+'/reg')

    counter = 0
    time_notes = []

    try:
        while True:
    #        save_result_image = False
            #print( f'image counter = {counter}')
            counter = counter + 1

            tic = time.time()
            image, filename  = frame.get_frame()
            print(filename)

            if image is None:
                break

            isJPG = filename.endswith('.jpg')
            ext = '.jpg' if isJPG else '.png'
            jsonfilename = filename.replace( ext, ".json" )

            scale = 1
            # character_height = 106 #CH #12 #STANDARD_CHARACTER_HEIGHT
            # scale = np.float(STANDARD_CHARACTER_HEIGHT) / character_height
#            scale = 1246.50838 / max(image.shape[0], image.shape[1])
            STR = 2150. #general #661. #handwritten #
            character_height = np.float(STANDARD_CHARACTER_HEIGHT) * np.float(max(image.shape[0], image.shape[1])) / STR
            scale = np.float(STR) / max(image.shape[0], image.shape[1])

            if os.path.exists(jsonfilename):
                with open(jsonfilename) as json_data:
                    d = json.load(json_data)
                    json_data.close()

                    scale = np.float(STANDARD_CHARACTER_HEIGHT) / d["character_height"]
                    character_height = d["character_height"]

                    expectshape = scale*np.array(image.shape)
                    if expectshape[0]>2500 or expectshape[1]>2500:
                        scale = 2500/max(image.shape[0], image.shape[1])
            else:
                expectshape = scale*np.array(image.shape)
                if expectshape[0]>2500 or expectshape[1]>2500:
                    scale = 2500/max(image.shape[0], image.shape[1])
                    print(expectshape)

            newshape = scale * np.array( image.shape )
            newshape = newshape.astype( np.int32 )

            image = cv2.resize(image, (newshape[1], newshape[0]), interpolation=cv2.INTER_CUBIC)
#            if newshape[0] < image.shape[0]:
#                image = cv2.resize( image, (newshape[1], newshape[0]), interpolation=cv2.INTER_AREA)
#            else:
#                image = cv2.resize(image, (newshape[1], newshape[0]), interpolation=cv2.INTER_LANCZOS4)

            print('character height: ', character_height)

            #network_input = cv2.resize( image[:,:,::-1], (IMAGE_WIDTH, IMAGE_WIDTH) )
            network_input = image[:,:,::-1]
            ny, nx, nc = network_input.shape
            src = network_input.copy().astype(np.float32)/255.0
            network_input = (src   - 0.5) * np.sqrt(2.0)
            network_input = np.array( network_input ).reshape([1,ny,nx,3])

            _output = sess.run( rtnet.output, feed_dict={rtnet.input: network_input} )
            _output = _output[0][0,...]

            savename = filename.split('/')[-1].rstrip(ext) #('.jpg')

            if save_heatmap:
                np.save(dir_name+'/heatmap_'+savename+'.npy', _output)

                canvas_save = np.zeros((image.shape[0], image.shape[1]*2, 3), np.uint8)
                affinity_save_horz = (_output[:,:,9]-np.min(_output[:,:,9]))/(np.max(_output[:,:,9]) - np.min(_output[:,:,9]))*255
                affinity_save_horz = affinity_save_horz.astype(np.uint8)
                affinity_save_horz = cv2.resize(affinity_save_horz, (image.shape[1], image.shape[0]))
                affinity_save_horz = np.tile(np.expand_dims(affinity_save_horz, -1), (1,1,3))
                affinity_save_horz = cv2.addWeighted(affinity_save_horz, 0.8, image, 0.2, 20)
                affinity_save_vert = (_output[:,:,10]-np.min(_output[:,:,10]))/(np.max(_output[:,:,10]) - np.min(_output[:,:,10]))*255
                affinity_save_vert = affinity_save_vert.astype(np.uint8)
                affinity_save_vert = cv2.resize(affinity_save_vert, (image.shape[1], image.shape[0]))
                affinity_save_vert = np.tile(np.expand_dims(affinity_save_vert, -1), (1,1,3))
                affinity_save_vert = cv2.addWeighted(affinity_save_vert, 0.8, image, 0.2, 20)
                canvas_save[:,:image.shape[1],:] = affinity_save_horz
                canvas_save[:,image.shape[1]:,:] = affinity_save_vert
                cv2.imwrite(dir_name+'/affinity_'+savename+'.jpg', canvas_save)
                colors = [(255,0,0), (0,255,0), (0,0,255), (254,1,0), (0,254,1), (1,0,254), (254,0,1), (1,254,0), (0,1,254)]
                for ind_ in range(11):
#                    canvas_save = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
                    canvas_save = (_output[:,:,ind_]-np.min(_output[:,:,ind_]))/(np.max(_output[:,:,ind_]) - np.min(_output[:,:,ind_]))
                    canvas_save = cv2.resize(canvas_save, (image.shape[1], image.shape[0]))
                    if ind_ < 9:
                        canvas_r = np.expand_dims(canvas_save.copy()*colors[ind_][0], -1)
                        if colors[ind_][0] != 0:
                            canvas_r = (canvas_r-np.min(canvas_r))/(np.max(canvas_r) - np.min(canvas_r))
                        canvas_g = np.expand_dims(canvas_save.copy()*colors[ind_][1], -1)
                        if colors[ind_][1] != 0:
                            canvas_g = (canvas_g-np.min(canvas_g))/(np.max(canvas_g) - np.min(canvas_g))
                        canvas_b = np.expand_dims(canvas_save.copy()*colors[ind_][2], -1)
                        if colors[ind_][2] != 0:
                            canvas_b = (canvas_b-np.min(canvas_b))/(np.max(canvas_b) - np.min(canvas_b))
                        canvas_save = np.concatenate((canvas_b, canvas_g, canvas_r), -1)
                    else:
                        canvas_save = np.tile(np.expand_dims(canvas_save, -1), (1,1,3))
                    canvas_save = canvas_save*255
                    canvas_save = canvas_save.astype(np.uint8)

                    canvas_save = cv2.addWeighted(canvas_save, 0.8, image, 0.2, 20)
                    cv2.imwrite(dir_name+'/'+savename+'heatmap{:d}'.format(ind_)+'.jpg', canvas_save)

            file_notes = ['filename:', filename, '::']
            file_notes += ['num of extra nodes:', str(num_extra_candidates), '::']

            if with_detection:
                # detection label: in xml form
                xmlfilename = filename.replace(ext, '.xml')
                tree_table = ET.parse(xmlfilename)
                root_table = tree_table.getroot()
                table_loc = []
                for child_table in root_table:
                    for child in child_table:
                        if child.tag == "Coords":
                            tab_loc = child.attrib["points"].split(' ')
                            tab_loc = [x.split(',') for x in tab_loc]
                            tab_loc = np.array(tab_loc)
                            tab_loc = np.reshape(tab_loc, (-1))
                            tab_loc = list(map(int, tab_loc))
                            tab_loc = np.reshape(tab_loc, (-1,2))
                            table_loc.append(tab_loc)
                            break

                _outputs = []
                for tab_loc in table_loc:
                    x1, y1, x2, y2 = min(tab_loc[:,0]), min(tab_loc[:,1]), max(tab_loc[:,0]), max(tab_loc[:,1])
                    x1 = int(x1*scale/2); y1 = int(y1*scale/2); x2 = int(x2*scale/2); y2 = int(y2*scale/2)
                    _output_tab = _output[y1:y2, x1:x2, :]
                    _outputs.append(_output_tab)
            else:
                _outputs = [_output]

            img_write = None
            for ind_o, _o in enumerate(_outputs):
                # Optimize the type and links of found junctions
                data_table, candidate_location, horz_adjacency_matrix, horz_adjacency_score, \
                    vert_adjacency_matrix, vert_adjacency_score = get_test_vector_from_path(_o, filename.rstrip(ext), dir_name, img_type=ext, test_scan=test_scan)

                if data_table is None:
                    print('No table in %s' % filename)
                    cv2.imwrite(dir_name+'/table_%s%s' % (filename.split('/')[-1].rstrip(ext), ext), np.zeros([_o.shape[0], _o.shape[1], 3], np.uint8))
                    continue
                gpm = gp.Model("qcp")
                gpm.setParam('OutputFlag', False)

                if len(candidate_location) < 4:
                    continue
                vman = variable_manager(gpm, data_table, candidate_location, \
                        horz_adjacency_matrix, horz_adjacency_score, vert_adjacency_matrix, vert_adjacency_score, \
                        num_extra_nodes=num_extra_candidates, H=_o.shape[0], W=_o.shape[1], img=image, test_scan=test_scan)

                # Optimization
                extra_lab = prime_optimization(vman, gpm, niter_th=niter_th, save_inter_result=save_intermediate_optimization, DIR=dir_name, filename=savename, scale=scale)
                print('primary optimization succeed')

                # check whether there are any valid extra nodes
                check=0
                temp = [e for e in extra_lab if e==vman.num_labels]

                if len(temp) != len(extra_lab): # valid extra nodes found!
                    pos_obj = extra_pos_optimization(vman, gpm)
                    print('extra nodes optimization processed')

                    if gpm.status == 2: # Problem solved!
                        print('... succeed')
                    else:
                        print('invalid optimization result... reset.. having 0 extra node')
                        gpm = gp.Model("qcp")
                        gpm.setParam('OutputFlag', False)

                        vman = variable_manager(gpm, data_table, candidate_location, \
                                horz_adjacency_matrix, horz_adjacency_score, vert_adjacency_matrix, vert_adjacency_score, \
                                num_extra_nodes=0, H=_o.shape[0], W=_o.shape[1], img=image, test_scan=test_scan)

                        _ = prime_optimization(vman, gpm, niter_th, save_intermediate_optimization, dir_name, savename, scale=scale)

                if with_detection:
                    _, img_write = save_result(niter_th, vman, savename, dir_name, scale, img_save=True, xml_save=True, test_scan=test_scan, tab_loc=table_loc[ind_o], img_write=img_write)
                else:
                    _, img_write = save_result(niter_th, vman, savename, dir_name, scale, img_save=True, xml_save=True, test_scan=test_scan, img_write=img_write)

            elapsed = time.time() - tic
            print( "output elapsed (fps)=",  elapsed, 1.0/elapsed )
            file_notes += ['processing time: ', '{:.4f}'.format(elapsed), '(sec)']
            time_notes.append(file_notes)

    except KeyboardInterrupt:
        print('Keyboard Interrupt')
    finally:
        f=open(dir_name+'/time.txt', 'w')
        f.write(FLAGS.experiment_name+'\n')
        for note in time_notes:
            for word in note:
                f.write(word)
            f.write('\n')
        f.close()



if __name__ == "__main__":
    #W111_to_159 = landmark_conversion.get_111_to_159_conversion_matrix()
    #print( W111_to_159 )
    points = np.array( [ [10,1], [1,10],  [10,10] ] )
    print( convex_hull_area( points) )
