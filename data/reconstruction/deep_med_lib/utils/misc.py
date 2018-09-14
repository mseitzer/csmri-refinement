""" This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.

For example ``tile_raster_images`` helps in generating a easy to grasp
image from a set of samples or weights.
"""
from __future__ import print_function

import os
import numpy as np
import subprocess

__author__ = 'js3611'

def check_and_mkdir(dire):
    if not os.path.isdir(dire):
        os.makedirs(dire)


def get_project_root(machine=None):
    '''
    machine is either 'home' 'school' 'hipeds' or None (automatically decided)
    '''
    home = os.path.isdir('/Users/joschlemper')
    school = os.path.isdir('/vol/bitbucket/js3611/')
    if home or machine == 'home':
        project_root = '/Users/joschlemper/PyCharmProjects/mres_project/'
    elif school or machine == 'school':
        project_root = '/vol/bitbucket/js3611/caffe/examples/SegCSCNN'
    else:
        project_root = '/home/js3611/projects/mres_project/'

    return project_root


def mat2py(x):
    '''
    In Python we usually expect data to be [nt, nx, ny],
    whereas in matlab this is [nx, ny, nt]. This function makes such transition
    '''
    if x.ndim != 3:
        return ValueError

    return np.transpose(x, (2, 0, 1))


def get_headline(string, n=80):
    if len(string) % 2 == 1:
        string += ' '
    pd = (n-len(string)-1)/2
    return '\n' + '-'*n + '\n' + '-'*pd + ' ' + string + ' ' \
        + '-'*pd + '\n' + '-'*n


def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def get_hostname():
    """
    Gets host name of the machine. For example, if you run on shell,
    it will say: shell01, on school machine, it will say: doc-id
    """
    cmd = subprocess.Popen(['hostname'], shell=False,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)

    return cmd.stdout.readlines()[0].split('.')[0]

######################## LOAD/SAVE IMAGE METHODS ########################


def save_image(filename, img):
    scipy.misc.imsave(filename, img.reshape(img.shape[-2:]))


def load_image(filename, divisor=255):
    img = scipy.misc.imread(filename) / np.float32(divisor)
    return img.transpose(0, 1).reshape(img.shape[0], img.shape[1])


def load_seg(filename):
    img = scipy.misc.imread(filename) / np.float32(50)
    return img.transpose(0, 1).reshape(img.shape[0], img.shape[1])


def load_frames(filenames, dtype):
    # load all the available frames
    frame_seq = [load_image(fName) for fName in filenames if fName is not None]

    # Build the sequence
    frame_seq = np.array(frame_seq, dtype=dtype)
    frame_seq = np.moveaxis(frame_seq, 0, -1)

    return frame_seq


def resize_image(img, scale):
    img_shape = img.shape
    ns = (img_shape[0] // scale[0], img_shape[1] // scale[1])  # new image shape
    return resize(img, ns, order=1, mode='constant', preserve_range=True)


def pad_image(img, shape, padval=0):
    if shape[0] == img.shape[0] and shape[1] == img.shape[1] and shape[2] == img.shape[2]:
        return img

    padx = int(np.ceil((shape[0] - img.shape[0]) / 2.0))
    pady = int(np.ceil((shape[1] - img.shape[1]) / 2.0))
    padt = int((shape[2] - img.shape[2]))

    padx = (padx, shape[0] - img.shape[0] - padx) if (padx > 0) else (0, 0)
    pady = (pady, shape[1] - img.shape[1] - pady) if (pady > 0) else (0, 0)
    padt = (int(0), padt) if (padt > 0) else (0, 0)

    return np.pad(img, pad_width=(padx, pady, padt), mode='constant', constant_values=padval)


def crop_image(img, shape):
    if shape[0] == img.shape[0] and shape[1] == img.shape[1] and shape[2] == img.shape[2]:
        return img

    cropx = int(np.ceil((img.shape[0] - shape[0]) / 2.0))
    cropy = int(np.ceil((img.shape[1] - shape[1]) / 2.0))
    cropt = int((img.shape[2] - shape[2]))

    cropx = (cropx, img.shape[0] - shape[0] - cropx) if (cropx > 0) else (0, 0)
    cropy = (cropy, img.shape[1] - shape[1] - cropy) if (cropy > 0) else (0, 0)
    cropt = (int(0), cropt) if (cropt > 0) else (0, 0)

    return img[cropx[0]:img.shape[0] - cropx[1], cropy[0]:img.shape[1] - cropy[1], cropt[0]:img.shape[2] - cropt[1]]


def save_image_old(filename, img):
    scipy.misc.imsave(filename, img.reshape(c.height, c.width))


def load_image_old(filename):
    img = scipy.misc.imread(filename) / np.float32(255)
    return img.transpose(0, 1).reshape(1, c.height, c.width)


def load_seg_old(filename):
    img = scipy.misc.imread(filename) / np.float32(50)
    return img.transpose(0, 1).reshape(1, c.height, c.width)


# load all images/labels from the data_dir and randomly split them into train/val (if seed!=0)
def load_data(fold=1, num_folds=10, seed=0, datadir='train', only_names=True, autoencoder=False, subjects=None,
              image_ext='.tif'):
    mask_suffix = '_mask'

    # list of subjects
    if subjects is None:
        subjects = os.listdir(datadir)
        sort_nicely(subjects)

    if seed is not 0:
        np.random.seed(seed)
        np.random.shuffle(subjects)

    num_subjects = {}
    # validation subjects
    if num_folds <= 1:
        num_subjects[0] = 0
    else:
        num_subjects[0] = math.ceil(1 / num_folds * len(subjects))
    # train subjects
    num_subjects[1] = len(subjects) - num_subjects[0]

    sind = num_subjects[0] * (fold - 1)
    lind = sind + num_subjects[0]
    if lind > len(subjects):
        sub = lind - len(subjects)
        sind -= sub
        lind -= sub
    subjects = np.hstack([subjects[sind:lind], subjects[0:sind], subjects[lind:]]).tolist()

    Xs = {}
    ys = {}
    valsubjs = []
    trainsubjs = []
    for d in range(2):
        d_num_subjects = num_subjects[d]
        if d_num_subjects == 0:
            Xs[d] = None
            ys[d] = None
            continue;
        mask_names = [];
        for i in range(d_num_subjects):
            s = subjects.pop(0)
            if d == 0:
                valsubjs.append(s)
            else:
                trainsubjs.append(s)
            if autoencoder:
                new_mask_names = glob.glob(datadir + '/' + s + '/*' + image_ext)
                new_mask_names = [curimg for curimg in new_mask_names if mask_suffix not in curimg]
                mask_names = mask_names + new_mask_names
            else:
                mask_names = mask_names + glob.glob(datadir + '/' + s + '/*' + mask_suffix + image_ext)
        num_images = len(mask_names)
        sort_nicely(mask_names)
        Xs[d] = {}
        ys[d] = {}
        ind = 0
        for mask_name in mask_names:
            Xs[d][ind] = mask_name.replace(mask_suffix, "")
            ys[d][ind] = mask_name
            ind = ind + 1
    valstr = 'validation set:'
    trainstr = 'training set:'
    for s in range(len(valsubjs)):
        valstr += ' ' + valsubjs[s]
    for s in range(len(trainsubjs)):
        trainstr += ' ' + trainsubjs[s]
    print(valstr)
    print(trainstr)

    return Xs[1], ys[1], Xs[0], ys[0]


######################## USEFUL METHODS ########################

def uniq(input):
    output = []
    for x in input:
        if x not in output:
            output.append(x)
    return output


def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    l.sort(key=alphanum_key)


######################## LOAD/SAVE RESULTS METHODS ########################

def get_fold_dir(version, fold=1, num_folds=10, seed=1234):
    suffix = "/fold{}_{}_seed{}".format(fold, num_folds, seed)
    return os.path.join(params_dir, '{}{}'.format(version, suffix))


# load config file
def load_config(version):
    # load default config
    global c
    import config_default as c

    # merge default and model config
    model_config = params_dir + "/" + str(version) + "/config.py"
    if os.path.exists(model_config):
        import importlib
        import sys
        sys.path.append(os.path.dirname(model_config))
        mname = os.path.splitext(os.path.basename(model_config))[0]
        c2 = importlib.import_module(mname)
        c.__dict__.update(c2.__dict__)
    else:
        import warnings
        warnings.warn("using default parameters")

    # load subjects if there is a list
    c.subjects = None
    subj_config = params_dir + "/" + str(version) + "/subjects.csv"
    if os.path.exists(subj_config):
        with open(subj_config, 'r') as f:
            content = f.readlines()
        c.subjects = [x.strip() for x in content]

        # params for augmentation
    c.aug_params = {
        'use': c.augment,
        'non_elastic': c.non_elastic,
        'non_elastic_type': c.non_elastic_type,
        'translation': c.shift,
        'translation_std': c.shift_std,
        'rotation': c.rotation,
        'rotation_std': c.rotation_std,
        'shear': c.shear,
        'shear_std': c.shear_std,
        'zoom': c.scale,
        'zoom_std': c.scale_std,
        'do_flip': c.flip,
        'allow_stretch': c.stretch,
        'elastic': c.elastic,
        'elastic_warps_dir': c.elastic_warps_dir,
        'alpha': c.alpha,
        'sigma': c.sigma,
        'autoencoder': c.autoencoder
    }
    return c


def save_pickle(obj, filename):
    d = os.path.dirname(filename)
    if not os.path.exists(d):
        os.makedirs(d)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object


def load_results(folddir):
    mve = None
    train_error = {}
    val_error = {}
    val_accuracy = {}
    fn = folddir + '/results.pickle'
    # print("load_results "+fn)
    if os.path.isfile(fn):
        [mve, train_error, val_error, val_accuracy] = load_pickle(fn)
    return [mve, train_error, val_error, val_accuracy]


def save_results(mve, train_error, val_error, val_accuracy, folddir):
    fn = folddir + '/results.pickle'
    # print("save_results "+fn)
    save_pickle([mve, train_error, val_error, val_accuracy], fn)


def resume(model, folddir):
    epoch = 0
    batch = 0
    fn = folddir + '/checkpoint.pickle'
    if os.path.isfile(fn):
        [epoch, batch] = load_params(model, fn)
    [mve, train_error, val_error, val_accuracy] = load_results(folddir)
    return [epoch, batch, mve, train_error, val_error, val_accuracy]


def load_best_params(model, folddir):
    return load_params(model, folddir + '/params_best.pickle')


def load_params(model, fn):
    [param_vals, epoch, batch] = load_pickle(fn)
    if model is not None:
        lasagne.layers.set_all_param_values(model, param_vals)
    return [epoch, batch]


def save_params(model, epoch, batch, folddir, type='current'):
    if type == 'epoch':
        fn = folddir + '/params_e{}.pickle'.format(epoch)
    elif type == 'best':
        fn = folddir + '/params_best.pickle'.format(epoch)
    else:
        fn = folddir + '/checkpoint.pickle'
    param_vals = lasagne.layers.get_all_param_values(model)
    save_pickle([param_vals, epoch, batch], fn)


def completed(folddir):
    fn = folddir + '/completed'
    # print("completed "+fn)
    if os.path.isfile(fn):
        with open(fn, 'r') as rf:
            print("best score: " + rf.readline() + '\n')
        return True
    return False


def finish(mve, folddir):
    fn = folddir + '/completed'
    # print("finish "+fn)
    with open(fn, 'w') as wf:
        wf.write(str(mve))


######################## ADDED BY OO2113 / RETRIEVE FILENAMES / Number of Classes ####################

def star(f):
    return lambda args: f(*args)


def groupintegers(i):
    for a, b in itertools.groupby(enumerate(i), star(lambda x, y: (y - x))):
        b = list(b)
        yield b[0][1], b[-1][1]


def findfiles(directory, extension):
    imagefullnames = []
    imagenames = []
    for folder, subfolder, files in os.walk(directory):
        for file in sorted(files):
            if file.endswith(extension):
                imagefullnames.append(os.path.join(folder, file))
                imagenames.append(file.split(extension)[0])
    return imagefullnames, imagenames


def find_empty_dirs(root_dir='.'):
    for dirpath, dirs, files in os.walk(root_dir):
        if not dirs and not files:
            yield dirpath


def recursive_delete_if_empty(path):
    """Recursively delete empty directories; return True
    if everything was deleted."""

    if not os.path.isdir(path):
        # If you also want to delete some files like desktop.ini, check
        # for that here, and return True if you delete them.
        return False

    # Note that the list comprehension here is necessary, a
    # generator expression would shortcut and we don't want that!
    if all([recursive_delete_if_empty(os.path.join(path, filename))
            for filename in os.listdir(path)]):
        # Either there was nothing here or it was all deleted
        os.rmdir(path)
        return True
    else:
        return False


def GetNumberOfClasses(imgdir):
    globalClassNames = sorted(next(os.walk(imgdir))[1])
    nGlbClass = len(globalClassNames)
    return nGlbClass


def readcsvfile(csvname, offset=int(0), delimiter=","):
    with open(csvname, "r") as f:
        reader = csv.reader(f, delimiter=delimiter)
        lines = []
        for idx, line in enumerate(reader):
            if idx <= offset:  # (Optionally) skip headers
                continue
            lines.append(line)
        return lines


def read_xlsx(excelfilename, sheet_number, offset=int(0)):
    workbook = xlrd.open_workbook(excelfilename)
    worksheet = workbook.sheet_by_index(sheet_number)

    # Change this depending on how many header rows are present
    # Set to 0 if you want to include the header data.
    rows = []
    for i, row in enumerate(range(worksheet.nrows)):
        if i <= offset:  # (Optionally) skip headers
            continue
        r = []
        for j, col in enumerate(range(worksheet.ncols)):
            r.append(worksheet.cell_value(i, j))
        rows.append(r)

    return rows


def mkdirfunc(directory):
    if not (os.path.exists(directory)):
        os.makedirs(directory)


def mydeletefile(filename):
    try:
        os.remove(filename)
    except OSError:
        pass


def writetxt2img(img, text, loc=(0.1, 0.9), color=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_size = img.shape
    loc = (int(img_size[1] * loc[0]), int(img_size[0] * loc[1]))
    cv2.putText(img, text, loc, font, 1, color, thickness=2, lineType=cv2.LINE_AA)


def getClassLookupTable(imgdir):
    classLookupTable = dict()
    globalClassNames = sorted(next(os.walk(imgdir))[1])
    for classId, classname in enumerate(globalClassNames):
        classLookupTable[classname] = classId
    return classLookupTable


def getFrameNamesLookupTable(listoffilenames, nframes):
    frameNamesLookupTable = dict()
    frameSubjectLookupTable = dict()

    for filename in listoffilenames:
        subjectname = filename.split('/')[-2]
        ints = list(map(int, re.findall(r'\d+', os.path.basename(filename.split('/')[-1]))));
        assert len(ints) == 1
        framenumber = ints[0]
        frameSubjectLookupTable[(subjectname, framenumber)] = filename

    for filename in listoffilenames:
        subjectname = filename.split('/')[-2]
        ints = list(map(int, re.findall(r'\d+', os.path.basename(filename.split('/')[-1]))));
        assert len(ints) == 1
        framenumber = ints[0]
        framenames = []
        for frameId in range(nframes):
            framename = frameSubjectLookupTable.get((subjectname, framenumber - frameId))
            framenames.append(framename)
        frameNamesLookupTable[filename] = framenames

    return frameNamesLookupTable


def returnIntAfterWord(input_string, inputword):
    match = re.search('{word}(\d+)'.format(word=inputword), input_string)
    if match:
        return int(match.group(1))
    else:
        return None

from scipy.ndimage import zoom
def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # width and height of the zoomed image
    zh = int(np.round(zoom_factor * np.float32(h)))
    zw = int(np.round(zoom_factor * np.float32(w)))

    # for multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor, zoom_factor)  + (1,) * (img.ndim - 2)

    # zooming out
    if zoom_factor < 1:
        # bounding box of the clip region within the output array
        top = (h - zh) // 2
        left = (w - zw) // 2
        # zero-padding
        out = np.zeros_like(img)
        out[top:top + zh, left:left + zw] = zoom(img, zoom_tuple, **kwargs)

    # zooming in
    elif zoom_factor > 1:
        # bounding box of the clip region within the input array
        top = (zh - h) // 2
        left = (zw - w) // 2
        out = zoom(img[top:top + zh, left:left + zw], zoom_tuple, **kwargs)
        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top + h, trim_left:trim_left + w]

    # identity output
    else:
        out = img.copy()

    return out

######################## CLASSES ADDED BY OO2113 ########################

# Convert a stack of numpy arrays to a video:
class VideoWriter(object):
    def __init__(self, filenames, featuremap_names=None):
        self.filenames = sorted(filenames)
        self.nImages = len(self.filenames)
        self.usefms = True if featuremap_names else False
        self.featuremap_names = sorted(featuremap_names) if self.usefms else []
        if self.usefms: assert self.nImages == len(self.featuremap_names)

    def write_video(self, outputvideoname):
        from scipy.misc import imread, toimage

        # Grab the stats from image1 to use for the resultant video
        if self.usefms:
            image1 = VideoWriter.merge_images(self.filenames[0], self.featuremap_names[0])
        else:
            image1 = Image.open(self.filenames[0])
        height, width = np.array(image1).shape

        # Create the OpenCV VideoWriter
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        video = cv2.VideoWriter(filename=outputvideoname, fourcc=fourcc, fps=20, frameSize=(width, height))

        # Load all the frames
        for framename, fmname in zip(self.filenames, self.featuremap_names):

            if self.usefms:
                frame = VideoWriter.merge_images(framename, fmname)
            else:
                frame = Image.open(framename)

            # Conversion from PIL to OpenCV from: http://blog.extramaster.net/2015/07/python-converting-from-pil-to-opencv-2.html
            video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_GRAY2RGB))

        # Release the video for it to be committed to a file
        video.release()

    @staticmethod
    def merge_images(file1, file2):
        """Merge two images into one, displayed side by side
        :param file1: path to first image file
        :param file2: path to second image file
        :return: the merged Image object
        """
        image1 = toimage(imread(file1), high=255, low=0)
        image2 = toimage(imread(file2), high=255, low=0)

        (width1, height1) = image1.size
        (width2, height2) = image2.size

        result_width = width1 + width2
        result_height = max(height1, height2)

        result = Image.new('L', (result_width, result_height))
        result.paste(im=image1, box=(0, 0))
        result.paste(im=image2, box=(width1, 0))
        return result
