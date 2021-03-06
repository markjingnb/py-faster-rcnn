import os
import time
import cPickle
import datetime
import logging
import flask
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import pandas as pd
from PIL import Image
import cStringIO as StringIO
import urllib
import exifutil
#from tools.jnbdemo import zfjnb
from tools.jnbdemo import init_cnn
from tools.jnbdemo import do_run
import cv2

#caffe_root = '../../../caffe-fast-rcnn/'
import sys  
#sys.path.insert(0, caffe_root + 'python')  
#sys.path.insert(0, caffe_root + 'example/web_demo/lib')  
import caffe  
reload(sys)
from utils.timer import Timer
faster_net=init_cnn()
#import caffe
sys.setdefaultencoding('utf-8')

REPO_DIRNAME = '/home/wechat/jnbgit/py-faster-rcnn/caffe-fast-rcnn'#os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../..')
UPLOAD_FOLDER = '/tmp/caffe_demos_uploads'
#UPLOAD_FOLDER = '/dev/shm'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])
#FORMAT_FILE = '/dev/shm/fmfile.png'
FORMAT_FILE = '/tmp/caffe_demos_uploads/fmfile.jpg'

# Obtain the flask app object
app = flask.Flask(__name__)


@app.route('/')
def index():
    return flask.render_template('index2.html', has_result=False)


@app.route('/classify_url', methods=['GET'])
def classify_url():
    imageurl = flask.request.args.get('imageurl', '')
    try:
        string_buffer = StringIO.StringIO(
            urllib.urlopen(imageurl).read())
        image = caffe.io.load_image(string_buffer)

    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        return flask.render_template(
            'index2.html', has_result=True,
            result=(False, 'Cannot open image from URL.')
        )

    logging.info('Image: %s', imageurl)
    #result = app.clf.classify_image(image)
    
    #im_file2 = ('/home/ubuntu/jnb/py-faster-rcnn/data/demo/dog.jpg')
    image = Image.fromarray((255 * image).astype('uint8'))
    #cv2.imwrite('tmp.jpg',image)
    image.save(FORMAT_FILE,'BMP')
    #zfjnb(FORMAT_FILE)
    do_run(faster_net,FORMAT_FILE)
    #logging.info('Saving to %s.', filename)
    image = exifutil.open_oriented_im('tmp.png')

    return flask.render_template(
        #'index.html', has_result=True, result=result, imagesrc=imageurl)
        'index2.html', has_result=True, result=False, imagesrc=embed_image_html(image))

@app.route('/classify_upload', methods=['POST'])
def classify_upload():
    timer = Timer()
    timer.tic()
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']

        timer.toc()
        print ('get src file time {:.3f}s ').format(timer.total_time)
        

        timer = Timer()
        timer.tic()
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        imagefile.save(filename)
        logging.info('Saving to %s.', filename)
        image = exifutil.open_oriented_im(filename)
        #print('Saving to %s.\n\n\n\n\n\n\n\n ', filename)

    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'index2.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )

    #result = app.clf.classify_image(image)
    #im_file2 = ('/home/ubuntu/jnb/py-faster-rcnn/data/demo/dog.jpg')

    image = Image.fromarray((255 * image).astype('uint8'))

    #cv2.imwrite('tmp.jpg',image)
    image.save(FORMAT_FILE,'BMP')

    timer.toc()
    print ('orginal alg {:.3f}s ').format(timer.total_time)


    timer = Timer()
    timer.tic()
    #zfjnb(FORMAT_FILE)
    do_run(faster_net,FORMAT_FILE)
    timer.toc()
    print ('jnb alg {:.3f}s ').format(timer.total_time)
    #logging.info('Saving to %s.', filename)
    image = exifutil.open_oriented_im(FORMAT_FILE)
    #zfjnb(filename)
    #image = exifutil.open_oriented_im(filename)

    timer = Timer()
    timer.tic()
    result = flask.render_template(
        'index2.html', has_result=True, result=False,
        imagesrc=embed_image_html(image)
    )

    timer.toc()
    print ('post file {:.3f}s ').format(timer.total_time)
    return result


def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    image_pil = Image.fromarray((255 * image).astype('uint8'))
    #image_pil = image_pil.resize((256, 256))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    #print string_buf
    #logging.info('Classijjjjjjnjnjnjnj: %s', string_buf)

    #im_file = os.path.join(string_buf, '.png')
    #im = cv2.imread(im_file)
    #zfjnb(string_buf);
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    #data=zfjnb(data);
    #print 'data:image/png;base64,' + data
    #print '\n\n\n\n'
    #print '\n\n\n\n'
    return 'data:image/png;base64,' + data


def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
    )

class ImagenetClassifier(object):
    default_args = {
        'model_def_file': (
            '{}/models/bvlc_reference_caffenet/deploy.prototxt'.format(REPO_DIRNAME)),
        'pretrained_model_file': (
            '{}/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'.format(REPO_DIRNAME)),
        'mean_file': (
            '{}/python/caffe/imagenet/ilsvrc_2012_mean.npy'.format(REPO_DIRNAME)),
        'class_labels_file': (
            '{}/data/ilsvrc12/synset_words.txt'.format(REPO_DIRNAME)),
        'bet_file': (
            '{}/data/ilsvrc12/imagenet.bet.pickle'.format(REPO_DIRNAME)),
    }
    for key, val in default_args.iteritems():
        if not os.path.exists(val):
            raise Exception(
                "File for {} is missing. Should be at: {}".format(key, val))
    default_args['image_dim'] = 256
    default_args['raw_scale'] = 255.

    def __init__(self, model_def_file, pretrained_model_file, mean_file,
                 raw_scale, class_labels_file, bet_file, image_dim, gpu_mode):
        logging.info('Loading net and associated files...')
        if gpu_mode:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        self.net = caffe.Classifier(
            model_def_file, pretrained_model_file,
            image_dims=(image_dim, image_dim), raw_scale=raw_scale,
            mean=np.load(mean_file).mean(1).mean(1), channel_swap=(2, 1, 0)
        )

        with open(class_labels_file) as f:
            labels_df = pd.DataFrame([
                {
                    'synset_id': l.strip().split(' ')[0],
                    'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
                }
                for l in f.readlines()
            ])
        self.labels = labels_df.sort('synset_id')['name'].values

        self.bet = cPickle.load(open(bet_file))
        # A bias to prefer children nodes in single-chain paths
        # I am setting the value to 0.1 as a quick, simple model.
        # We could use better psychological models here...
        self.bet['infogain'] -= np.array(self.bet['preferences']) * 0.1

    def classify_image(self, image):
        try:
            starttime = time.time()
            scores = self.net.predict([image], oversample=True).flatten()
            endtime = time.time()

            indices = (-scores).argsort()[:5]
            predictions = self.labels[indices]

            # In addition to the prediction text, we will also produce
            # the length for the progress bar visualization.
            meta = [
                (p, '%.5f' % scores[i])
                for i, p in zip(indices, predictions)
            ]
            logging.info('result: %s', str(meta))

            # Compute expected information gain
            expected_infogain = np.dot(
                self.bet['probmat'], scores[self.bet['idmapping']])
            expected_infogain *= self.bet['infogain']

            # sort the scores
            infogain_sort = expected_infogain.argsort()[::-1]
            bet_result = [(self.bet['words'][v], '%.5f' % expected_infogain[v])
                          for v in infogain_sort[:5]]
            logging.info('bet result: %s', str(bet_result))

            return (True, meta, bet_result, '%.3f' % (endtime - starttime))

        except Exception as err:
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one?')


def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)
    parser.add_option(
        '-g', '--gpu',
        help="use gpu mode",
        action='store_true', default=False)

    opts, args = parser.parse_args()
    #ImagenetClassifier.default_args.update({'gpu_mode': opts.gpu})

    # Initialize classifier + warm start by forward for allocation
    #app.clf = ImagenetClassifier(**ImagenetClassifier.default_args)
    #app.clf.net.forward()

    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    start_from_terminal(app)
