import os, json,sys
import numpy as np
import cv2
import tqdm
import math
import time
import skimage.transform
import random

sys.path.append( os.path.join(os.getcwd(), "code_commons") )
from global_constants import *
from auxiliary_ftns import *


def iterateTrainData(datadirs ):
    import pickle
    dataset = []

    #   List up all files and metadata
    for datadir in datadirs:
        sample_set_filename = os.path.join( datadir, "datafile_information")

        if os.path.isfile( sample_set_filename) is True:
            with open (sample_set_filename, 'rb') as fp:
                cur_dataset = pickle.load(fp)
            print( f"fileinformation (having {len(cur_dataset)} images) of '{datadir}'' exists in '{sample_set_filename}', using this file without searches" )

        else:
            cur_dataset = []

            for dirpath, _, filenames in os.walk(datadir):
                for filename in  tqdm.tqdm(filenames):
                    isJPG = filename.endswith('.jpg')
                    isPNG = filename.endswith('.png')
                    if isJPG or isPNG:

                        if isJPG:
                            jsonfilename = filename.replace(".jpg",".json" )
                        if isPNG:
                            jsonfilename = filename.replace(".png",".json" )
                        jsonfilename = os.path.join(dirpath, jsonfilename )

                        with open(jsonfilename) as json_data:
                            d = json.load(json_data)
                            json_data.close()

                            d["junction_infor"] = np.array( d["junction_infor"]  )
                            d["junction_location"] = np.array( d["junction_infor"][:,1:3])
                            d["junction_type"] = np.array( d["junction_infor"][:,3], dtype = np.int32 )
                            d["junction_radius"] = np.array( d["junction_infor"][:,4] )
                            d["x_connection"] = np.array( d["x_connection"], dtype = np.int32 )
                            d["y_connection"] = np.array( d["y_connection"], dtype = np.int32 )

                            num_junctions = d["junction_location"].shape[0]

                            d["x_connection_matrix"] = np.zeros( shape=[num_junctions,num_junctions], dtype = np.int32 )
                            d["y_connection_matrix"] = np.zeros( shape=[num_junctions,num_junctions], dtype = np.int32 )

                            for a in d["x_connection"]:
                                d["x_connection_matrix"][ a[0], a[1] ] = d["x_connection_matrix"][ a[1], a[0] ] = 1

                            for a in d["y_connection"]:
                                d["y_connection_matrix"][ a[0], a[1] ] = d["y_connection_matrix"][ a[1], a[0] ] = 1

                        if isJPG:
                            jpgPath = os.path.join(dirpath, os.path.splitext(filename)[0] + '.jpg')
                        if isPNG:
                            jpgPath = os.path.join(dirpath, os.path.splitext(filename)[0] + '.png')


                        if os.path.isfile(jpgPath):
                            img = cv2.imread( jpgPath )

                            d["width"] = img.shape[1]
                            d["height"] = img.shape[0]
                            cur_dataset.append((
                                os.path.join(dirpath, filename ),
                                d
                            ))

            with open(sample_set_filename, 'wb') as fp:
                pickle.dump(cur_dataset, fp)

        dataset += cur_dataset

    return dataset


def valid_x_line( left_label, right_label ):
    assert left_label[0] ==  right_label[2]
    if left_label[0] == 1 and right_label[2] == 1:
        return True
    else:
        return False

def valid_y_line( up_label, down_label ):
    assert up_label[1] ==  down_label[3]

    if up_label[1] == 1 and down_label[3] == 1:
        return True
    else:
        return False

def draw_table( d, img, T ):
    tmpx = img[:,:,0].copy()
    tmpy = img[:,:,1].copy()

    for a in d["x_connection"]:
        idx1, idx2 = a
        color = (1)
        thickness = 2
        x1, y1 = d["junction_location"][ idx1 ]
        x2, y2 = d["junction_location"][ idx2 ]

        xpos1 = np.int( T[0,0] * x1 + T[0,1] * y1 + T[0,2])
        ypos1 = np.int( T[1,0] * x1 + T[1,1] * y1 + T[1,2])

        xpos2 = np.int( T[0,0] * x2 + T[0,1] * y2 + T[0,2])
        ypos2 = np.int( T[1,0] * x2 + T[1,1] * y2 + T[1,2])


        cv2.line( tmpx, (xpos1,ypos1), (xpos2,ypos2), color, thickness=thickness )

    for a in d["y_connection"]:
        idx1, idx2 = a
        color = (1)
        thickness = 2
        """
        x1, y1 = (d["junction_location"][ idx1 ]).astype(np.int32)
        x2, y2 = (d["junction_location"][ idx2 ]).astype(np.int32)
        cv2.line( tmpy, (x1,y1), (x2,y2), color, thickness=thickness )
        """
        x1, y1 = d["junction_location"][ idx1 ]
        x2, y2 = d["junction_location"][ idx2 ]

        xpos1 = np.int( T[0,0] * x1 + T[0,1] * y1 + T[0,2])
        ypos1 = np.int( T[1,0] * x1 + T[1,1] * y1 + T[1,2])

        xpos2 = np.int( T[0,0] * x2 + T[0,1] * y2 + T[0,2])
        ypos2 = np.int( T[1,0] * x2 + T[1,1] * y2 + T[1,2])

        cv2.line( tmpy, (xpos1,ypos1), (xpos2,ypos2), color, thickness=thickness )


    img[:,:,0] = tmpx
    img[:,:,1] = tmpy


# generate data consisting of images with landmarks
class TrainDataGenerator:
    def __init__(
            self,
            datadir, GEOMETRIC_DISTORTION = True, PHOTOMETRIC_DISTORTION = False ):

        self.dataset = iterateTrainData(datadir)
        self.channels = 3
        self.PHOTOMETRIC_DISTORTION = PHOTOMETRIC_DISTORTION
        self.GEOMETRIC_DISTORTION = GEOMETRIC_DISTORTION


    def geometric_distortion_validation(self, img, character_height, i=0, j=0):
        """ i:x-axis, j:y-axis """
        DST_WIDTH = IMAGE_WIDTH
        DST_HEIGHT = IMAGE_HEIGHT

        scale = 1.0 / np.float(STANDARD_CHARACTER_HEIGHT) * character_height

        img_h, img_w, _ = img.shape

        STEP_WIDTH = np.int(scale*DST_WIDTH)
        STEP_HEIGHT = np.int(scale*DST_HEIGHT)

        src_pts = np.array([[STEP_WIDTH*i, STEP_HEIGHT*j], [STEP_WIDTH*(i+1)-1, STEP_HEIGHT*j], [STEP_WIDTH*(i+1)-1, STEP_HEIGHT*(j+1)-1], [STEP_WIDTH*i, STEP_HEIGHT*(j+1)-1]], dtype=np.float32)
        dst_pts = np.array([[0,0], [DST_WIDTH-1,0], [DST_WIDTH-1,DST_HEIGHT-1], [0,DST_HEIGHT-1]], dtype=np.float32)

        tform = skimage.transform.estimate_transform('similarity', src_pts, dst_pts)

        return tform.params

    def geometric_distortion(self, img, character_height  ):
        DST_WIDTH = IMAGE_WIDTH
        DST_HEIGHT = IMAGE_HEIGHT


        theta = np.random.normal() * np.pi / 40.

        random_seed = random.uniform(0,1)
        if random_seed > 0.5:
            scale = 1.0 / np.float(STANDARD_CHARACTER_HEIGHT)  * character_height  * np.exp(np.random.normal()/10)
        else:
            if img.shape[0] > 2000 or img.shape[1] > 2000:
                scale = 2000/max(img.shape[0], img.shape[1])
            else:
                scale = 1.
        scalex = scale # * (1 + 0.2*np.random.uniform(-1.5,0.5) )
        scaley = scale

        W = scalex * DST_WIDTH // 2
        H = scaley * DST_HEIGHT // 2

        W = np.int( W )
        H = np.int( H )

        minx = np.min( [W,img.shape[1]-W] )
        maxx = np.max( [W,img.shape[1]-W, minx+1] )

        miny = np.min( [H,img.shape[0]-H] )
        maxy = np.max( [H,img.shape[0]-H, miny+1] )


        cx = np.random.randint( minx, maxx )
        cy = np.random.randint( miny, maxy )

        src_pts = np.zeros( shape=[4,2], dtype = np.float32)

        Wp = W * np.cos(theta) - H * np.sin(theta)
        Hp = W * np.sin(theta) + H * np.cos(theta)

        src_pts[0,0] = - W
        src_pts[0,1] = - H
        src_pts[1,0] = + W
        src_pts[1,1] = - H
        src_pts[2,0] = W
        src_pts[2,1] = H
        src_pts[3,0] = -W
        src_pts[3,1] = H

        src_pts = src_pts @ np.array( [[ np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype = np.float32 )
        src_pts = src_pts + np.array( [cx,cy], dtype=np.float32 )

        # src_pts = np.random.normal( size=[4,2])
        dst_pts = np.array( [ [0,0], [DST_WIDTH-1,0], [DST_WIDTH-1,DST_HEIGHT-1], [0,DST_HEIGHT-1] ], dtype = np.float32 )
        #src_pts = 2 * dst_pts

        tform = skimage.transform.estimate_transform( 'similarity', src_pts, dst_pts )

        return tform.params


    def generate_data( self, imgfilename, annotations, i=0, j=0):
        validity = True
        dataset = self.dataset
        image_size = IMAGE_WIDTH

        width = annotations["width"]
        height = annotations["height"]

        img = cv2.imread(imgfilename)
#        img_ = np.ones([img.shape[0]+IMAGE_HEIGHT, img.shape[1]+IMAGE_WIDTH, img.shape[2]], dtype=np.uint8) * 255
#        img_[0:img.shape[0], 0:img.shape[1], :] = img
#        T = self.geometric_distortion_validation(img, annotations["character_height"], i, j)
        T = self.geometric_distortion( img, annotations["character_height"] )

        canvas = cv2.warpAffine(img, T[:2], (IMAGE_WIDTH, IMAGE_HEIGHT), flags=cv2.INTER_CUBIC)
        if np.sum(canvas<200) == 0:
            validity=False

        s = float(HEATMAP_WIDTH) / IMAGE_WIDTH
        T = np.array( [[s,0,0],[0,s,0],[0,0,1]], dtype = np.float32 ) @ T

        heatmap = np.zeros( shape = (HEATMAP_HEIGHT, HEATMAP_WIDTH, HEATMAP_CHANNEL), dtype = np.float32 )


#        sigma = 3 * annotations["junction_radius"][0]
        sigma = 3 # borderless -> set one borderline!
        x = np.arange(0,HEATMAP_WIDTH,1)
        y = np.arange(0,HEATMAP_HEIGHT,1)
        xv, yv = np.meshgrid(x,y)


        """
        scalex = float(HEATMAP_WIDTH) / img.shape[1]
        scaley = float(HEATMAP_HEIGHT) / img.shape[0]



        planes = []
        for xdx in range(num_x_lines):
            xpos = (x_positions[xdx] * scalex)

            z = ( x - xpos ) ** 2
            planes.append(z)

        planes = np.array( planes )
        planes = np.min( planes, axis=0)
        projection[:,0] = np.exp( - planes / 2.0 / sigma / sigma )


        planes = []
        for ydx in range(num_y_lines):
            ypos = (y_positions[ydx] * scaley)

            z = ( y - ypos ) ** 2
            planes.append(z)

        planes = np.array( planes )
        planes = np.min( planes, axis=0)
        projection[:,1] = np.exp( - planes / 2.0 / sigma / sigma )
        """



        for label_idx in range( HEATMAP_CHANNEL ):
            planes = []
            for jtype, jloc in zip( annotations["junction_type"], annotations["junction_location"]):
                if jtype == label_idx:
                    x, y = jloc
                    xpos = T[0,0] * x + T[0,1] * y + T[0,2]
                    ypos = T[1,0] * x + T[1,1] * y + T[1,2]

                    if 0 <= xpos and xpos < HEATMAP_WIDTH and 0 <= ypos and ypos < HEATMAP_HEIGHT:
                        z = ( xv - xpos ) ** 2 + ( yv - ypos ) ** 2
                        planes.append(z)

            if len(planes) > 0:
                planes = np.array( planes )
                planes = np.min( planes, axis=0)
                heatmap[:,:,label_idx] = np.exp( - planes / 2.0 / sigma / sigma )


        affinity_map = np.zeros( shape = (HEATMAP_HEIGHT, HEATMAP_WIDTH, 3), dtype = np.float32 )
        draw_table( annotations, affinity_map, T )
        affinity_map = cv2.GaussianBlur( affinity_map, (11,11), sigma )
        affinity_map = affinity_map / (np.max( affinity_map )+0.000001)

#        if validity:
#            cv2.imwrite('./temp/%s_%d_%d.jpg' % (os.path.basename(imgfilename).rstrip('.jpg'), i, j), canvas)

        canvas = (canvas.astype('float32') / 255.0 - 0.5) * np.sqrt(2.0)

        dict = {}
        dict["image"] = canvas[:,:,::-1]
        dict["affinity"] = affinity_map[:,:,:2]
        dict["heatmap"] = heatmap
        #dict["projection"] = projection

        ############### data check ###############
        for idx, name in enumerate(DATA_FIELD_NAMES):
            assert dict[name].dtype == DATA_FIELD_TYPES[idx]
            assert dict[name].shape == DATA_FIELD_SHAPES[idx]

        return dict , validity #True

    def __call__(self):
        dataset = self.dataset

        """"""
        s = np.uint32 ( os.getpid() * ( np.uint64( time.time() ) % 1000 ) )
        np.random.seed(s)
        print('seed:', s)


        try:
            while True:
                idx = np.random.randint(len(dataset))
                jpgPath, annotations = dataset[idx]

                train_dict, valid_data = self.generate_data( jpgPath, annotations )

                if valid_data is False:
                    continue

                yield train_dict
        except EOFError:
            return


