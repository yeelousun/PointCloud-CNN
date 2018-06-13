import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import random 
import scipy.misc

from collections import Counter
from sklearn.neighbors import KDTree 
#from mpl_toolkits.mplot3d import Axes3D
from input.Inputdatas import * 
from model.model import * 
from model.disply import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = 'log/72/model.ckpt-210'
SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(BASE_DIR, 'input/shape_names.txt'))] 

visu = 'store_true'
DUMP_DIR = 'dump'
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
#LOG_FOUT.write(str(FLAGS)+'\n')
NUM_CLASSES = 40
NUM_POINT  = 2048
BATCH_SIZE = 32
numkernel  = 64
pgnump     = 64

MAX_EPOCH  = 500

MOMENTUM = 0.9
OPTIMIZER = 'adam' #'adam or momentum'
DECAY_STEP = 200000
BASE_LEARNING_RATE= 0.001
DECAY_RATE = 0.7

#load train file
TRAIN_FILES = getDataFiles( \
    os.path.join(BASE_DIR, 'input/train_files.txt'))

TEST_FILES = getDataFiles( \
    os.path.join(BASE_DIR, 'input/test_files.txt'))

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate(num_votes):
    is_training = False
     
    with tf.device('/cpu:0'):
        pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, numkernel, pgnump)    
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred, end_points = get_model(pointclouds_pl, is_training_pl)
        loss = get_loss(pred, labels_pl, end_points)
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    sess = tf.Session(config=tf.ConfigProto(device_count={'GPU':0}))

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': loss}

    eval_one_epoch(sess, ops, num_votes)

   
def eval_one_epoch(sess, ops, num_votes=1, topk=1):
    error_cnt = 0
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    dis=np.zeros([40,40])
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    fout = open(os.path.join(DUMP_DIR, 'pred_label.txt'), 'w')
    for fn in range(len(TEST_FILES)):
        log_string('----'+str(fn)+'----')
        current_data, current_label =loadDataFile(TEST_FILES[fn])
        current_data = current_data[:,0:NUM_POINT,:]
        current_label = np.squeeze(current_label)
        print(current_data.shape)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        print(file_size)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            cur_batch_size = end_idx - start_idx
            
            test_data  = current_data[start_idx:end_idx, :, :]
            test_label = current_label[start_idx:end_idx]
            
            test_data_pg=np.zeros([BATCH_SIZE,numkernel,pgnump,3])
            for point_idx in range (BATCH_SIZE):
                rerowcol=test_data[point_idx]
                KernelList=random.sample(range(rerowcol.shape[0]),numkernel)
                #rerowcol_downsample is downsample point
                rerowcol_downsample=rerowcol[KernelList,:]
                kdt = KDTree(rerowcol, leaf_size=30, metric='euclidean')
                pgi=kdt.query(rerowcol_downsample, k=pgnump, return_distance=False)
                pgr=np.zeros([numkernel,pgnump,3])
                for n in range(pgi.shape[0]):
                  pgr[n]=rerowcol[pgi[n,:],:]
                test_data_pg[point_idx]=pgr
            print('test batch',batch_idx)

            # Aggregating BEG
            batch_loss_sum = 0 # sum of losses for the batch
            batch_pred_sum = np.zeros((cur_batch_size, NUM_CLASSES)) # score for classes
            batch_pred_classes = np.zeros((cur_batch_size, NUM_CLASSES)) # 0/1 for classes
            for vote_idx in range(num_votes):
                
                feed_dict = {ops['pointclouds_pl']: test_data_pg,
                             ops['labels_pl']: test_label,
                             ops['is_training_pl']: is_training}
                loss_val, pred_val = sess.run([ops['loss'], ops['pred']],
                                          feed_dict=feed_dict)
                batch_pred_sum += pred_val
                batch_pred_val = np.argmax(pred_val, 1)
                for el_idx in range(cur_batch_size):
                    batch_pred_classes[el_idx, batch_pred_val[el_idx]] += 1
                batch_loss_sum += (loss_val * cur_batch_size / float(num_votes))
            # pred_val_topk = np.argsort(batch_pred_sum, axis=-1)[:,-1*np.array(range(topk))-1]
            # pred_val = np.argmax(batch_pred_classes, 1)
            pred_val = np.argmax(batch_pred_sum, 1)
            # Aggregating END
            
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            # correct = np.sum(pred_val_topk[:,0:topk] == label_val)
            total_correct += correct
            total_seen += cur_batch_size
            loss_sum += batch_loss_sum

            for i in range(start_idx, end_idx):
                l = current_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx] == l)
                fout.write('%d, %d\n' % (pred_val[i-start_idx], l))
                if pred_val[i-start_idx] == l:
                    dis[l,l] += 1
                if pred_val[i-start_idx] != l and visu: # ERROR CASE, DUMP!
                    dis[l,pred_val[i-start_idx]] += 1
                    img_filename = '%d_label_%s_pred_%s.jpg' % (error_cnt, SHAPE_NAMES[l],
                                                           SHAPE_NAMES[pred_val[i-start_idx]])
                    img_filename = os.path.join(DUMP_DIR, img_filename)
                    output_img = point_cloud_three_views(np.squeeze(current_data[i, :, :]))
                    scipy.misc.imsave(img_filename, output_img)
                    error_cnt += 1
                
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    
    class_accuracies = np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float)
    for i, name in enumerate(SHAPE_NAMES):
        log_string('%10s:\t%0.3f' % (name, class_accuracies[i]))
        dis[i] = np.array(dis[i])/np.array(total_seen_class[i])
        #log_string('%10s:\t%0.3f' % (name, dis[i]))
    print (dis)
    plt.imshow(dis, interpolation='nearest',cmap=plt.cm.Blues) # Plot the confusion matrix as an image.
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(NUM_CLASSES)
    tick_marks = np.arange(NUM_CLASSES)
    plt.xticks(tick_marks, range(NUM_CLASSES))
    plt.yticks(tick_marks, range(NUM_CLASSES))
    plt.xlabel('Predicted')
    plt.ylabel('True')    
    plt.show()

if __name__ == "__main__":
    evaluate(num_votes=1)
    LOG_FOUT.close()
