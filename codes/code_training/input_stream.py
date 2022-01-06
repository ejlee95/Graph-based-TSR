import numpy as np
import cv2
import glob
#import cv2.cv as cv


class Frame(object):
    def __init__(self, mode, srcname=None, repeat=False):
        self.mode = mode
        self.target_size = None
        self.repeat = repeat
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

        if mode == 'webcam':
            self.cap = cv2.VideoCapture(0)
            if int(major_ver) < 3 :
                fps = self.cap.get(cv2.cv.CV_CAP_PROP_FPS)
                print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
                raise Exception("Check OpenCV versions")
            else:
                default_frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
                default_frame_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                default_frame_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
 #               print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(default_frame_rate))
                print( default_frame_width, default_frame_height, default_frame_rate)

        elif mode == 'folder':
            self.image_counter = 0
            self.filelist = []
            for one_srcname in srcname:
                self.filelist += glob.glob( one_srcname )
            self.filelist = sorted( set(self.filelist) )
            self.nFiles = len(self.filelist)
            print( "The number of file is: {}".format(self.nFiles))

        elif mode == 'video':
            self.cap = cv2.VideoCapture(srcname)
            default_frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
            default_frame_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            default_frame_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#               print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(default_frame_rate))
            print( default_frame_width, default_frame_height, default_frame_rate)

#   print(f"width={default_frame_width}, height={default_frame_height}, framerate={default_frame_rate}")
    def get_frame(self):
        filename = None

        if self.mode == 'webcam':
            ret, frame = self.cap.read()
            filename = 'from webcam'
        elif self.mode == 'video':
            if self.cap.isOpened() is True:
                ret, frame = self.cap.read()
                filename = 'from video'
            else:
                return None, None
        elif self.mode == 'folder':
            if self.image_counter >= self.nFiles:
                if self.repeat is False:
                    return None, None
                else:
                    self.image_counter = 0
            frame = cv2.imread( self.filelist[self.image_counter], cv2.IMREAD_COLOR )
            filename = self.filelist[self.image_counter]
            self.image_counter = self.image_counter + 1


        if self.target_size is not None and frame is not None:
            frame = cv2.resize(frame, self.target_size, cv2.INTER_LINEAR)


        return frame, filename

    def get_frame_from_folder(self, max_dimension = 1024 ):
        filename = None

        assert self.mode == 'folder'
        if self.image_counter >= self.nFiles:
            if self.repeat is False:
                return None, None, None
            else:
                self.image_counter = 0

        frame = cv2.imread( self.filelist[self.image_counter], cv2.IMREAD_COLOR )
        filename = self.filelist[self.image_counter]
        self.image_counter = self.image_counter + 1

        scale_factor = np.min( [ max_dimension/frame.shape[1], max_dimension/frame.shape[0]] )
        if scale_factor > 1:
            scale_factor = 1

        new_width = int(frame.shape[1] * scale_factor  )
        new_height = int(frame.shape[0] * scale_factor   )


        if scale_factor < 1 and frame is not None:
            frame = cv2.resize(frame, (new_width,new_height), cv2.INTER_LINEAR)

        print( frame.shape )

        return frame, filename, scale_factor

    def set_size(self, target_size):

        if self.mode == 'webcam':
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_size[0] )
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_size[1] )

            frame_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            frame_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.target_size = target_size

            if (frame_width, frame_height) != target_size:
                print('image size will be changed with bilinear_interpolation')
        else:
            self.target_size = target_size
            print('image size will be changed with bilinear_interpolation')

    def close(self):
        if self.mode == 'webcam' or self.mode == 'video':
            self.cap.release()


def video_recording():
    frame = Frame('webcam' )
    frame.set_size( (640,480))

    counter = 0
    while True:
        image, _ = frame.get_frame()
        filename = f"../sequences/diagonal_occlusions/{counter:05}.jpg"
        cv2.imwrite( filename, image  )
        cv2.imshow( "wnd", image )
        counter = counter + 1
        key = cv2.waitKey(1)
        if key == ord('q'):
            break


def video_saving():
    frame = Frame('video', '../sequences/model-with-blonde-hair-cover-eyes-with-both-hands-Y8UGKNE.mov' )
    frame.set_size( (1280,720))

    j = 0
    while True:
        img, _ = frame.get_frame()
        #img = img[160:800,5:445,:]

        if img is None:
            break

        file_name = "../v5_data/eyes_with_hands/eyes_with_hands{0:05d}.jpg".format(j)

        #font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(img, file_name,  (20,20), font, .8, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow("wnd", img)

        print( file_name )
        if  j % 15 == 0:
            cv2.imwrite( file_name, img)
        j = j + 1
        cv2.waitKey(1)



if __name__ == '__main__':
    #video_recording()
    video_saving()
    """
    frame = Frame('video', '../sequences/beardathome.mp4' )
    frame.set_size( (1280,720))

    j = 0
    while True:
        img, _ = frame.get_frame()
        #img = img[160:800,5:445,:]

        if img is None:
            break

        file_name = "../v5_data/new_images_2/beard{0:05d}.jpg".format(j)

        #font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(img, file_name,  (20,20), font, .8, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow("wnd", img)

        print( file_name )
        if j > 4795 and j % 15 == 0:
            cv2.imwrite( file_name, img)
        j = j + 1
        cv2.waitKey(1)
    """

    """
    frame = Frame('video', '../sequences/emilia2.mp4' )
    frame.set_size( (1280,720))

    j = 0
    while True:
        img, _ = frame.get_frame()
        #img = img[160:800,5:445,:]

        if img is None:
            break

        file_name = "../v5_data/new_images/emilia{0:05d}.jpg".format(j)

        #font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(img, file_name,  (20,20), font, .8, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow("wnd", img)

        print( file_name )
        if j > 2000 and j % 15 == 0:
            cv2.imwrite( file_name, img)
        j = j + 1
        cv2.waitKey(1)
    """

