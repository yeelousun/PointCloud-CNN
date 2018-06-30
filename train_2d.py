import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import random 


from collections import Counter
from sklearn.neighbors import KDTree 
from mpl_toolkits.mplot3d import Axes3D
from input.Inputdatas_2dcnn import * 
from model.model_2d import * 
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))



LOG_DIR = 'log'
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
#LOG_FOUT.write(str(FLAGS)+'\n')
NUM_CLASSES = 40
NUM_POINT  = 2048
BATCH_SIZE = 32
numkernel  = 512
pgnump     = 64

MAX_EPOCH  = 500

MOMENTUM = 0.9
OPTIMIZER = 'momentum' #'adam or momentum'
DECAY_STEP = 200000
BASE_LEARNING_RATE= 0.05
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

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def variable_summaries(var, name="layer"):
    with tf.variable_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def train():
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            pointclouds_pl, pointclouds_pl_m4, pointclouds_kernel, pointclouds_all ,labels_pl = placeholder_inputs(BATCH_SIZE, numkernel, pgnump)    
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            
            # Get model and loss 
            pred = get_model(pointclouds_pl, pointclouds_pl_m4, pointclouds_kernel, pointclouds_all, is_training_pl)
            loss = get_loss(pred, labels_pl)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            '''
            var_list_w = [var for var in tf.trainable_variables() if "w" in var.name]
            #var_list_b = [var for var in tf.trainable_variables() if "b" in var.name]

            gradient_w = optimizer.compute_gradients(loss, var_list=var_list_w)
            #gradient_b = optimizer.compute_gradients(loss, var_list=var_list_b)
   
            for idx, itr_g in enumerate(gradient_w):
                variable_summaries(itr_g[0], "layer%d-w"%idx)
            #for idx, itr_g in enumerate(gradient_b):
            #    variable_summaries(itr_g[0], "layer%d-b"%idx)
            '''
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
        
        # Create a session
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # config.allow_soft_placement = True
        # config.log_device_placement = False
        # sess = tf.Session(config=config)
        sess = tf.Session(config=tf.ConfigProto(device_count={'GPU':0})) 

        # Add summary writers
        #merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'pointclouds_pl_m4': pointclouds_pl_m4,
               'pointclouds_kernel':pointclouds_kernel,
               'pointclouds_all':pointclouds_all,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}
        
        max_acc=0

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            allaccuracy = train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)
            
            # Save the variables to disk.
            if epoch % 1 == 0:
                log_string('epoch: '+str(epoch)+', val_acc: '+str(allaccuracy)+'\n')
                if allaccuracy > max_acc:
                    max_acc=allaccuracy
                    save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), global_step=epoch)
                    log_string("Model saved in file: %s" % save_path)

def train_one_epoch(sess, ops, train_writer):
    is_training = True
    #shuffle train file
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)
    allaccuracy = 0

    for fn in range(len(TRAIN_FILES)):
        log_string('train file NUM---' + str(fn) + '-----')
        current_data, current_label = loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
        current_data = current_data[:,0:NUM_POINT,:]
        current_data, current_label, _ = shuffle_data(current_data, np.squeeze(current_label))            
        current_label = np.squeeze(current_label)
        file_size = current_data.shape[0]

        num_batches = file_size // BATCH_SIZE
        log_string('train file size---' + str(file_size) + '-----')
        
        total_correct = 0
        total_seen = 0
        loss_sum = 0

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            
            train_data_r  = rotate_point_cloud(current_data[start_idx:end_idx, :, :])
            train_data_rt  = rotate_point_cloud(train_data_r)
            train_label = current_label[start_idx:end_idx]
            train_data, train_data_m4, idx_data, train_data_all = point_group_first(train_data_rt,train_data_rt,numkernel,pgnump)
            print('train batch',batch_idx)
            # #disply point cloud
            # point = train_data[0][:][:][:]
            # point1 = idx_data[0][:][:]
            # print(point.shape)
            # print(train_label[0])

            # fig1=plt.figure(dpi=120)  
            # ax1=fig1.add_subplot(111,projection='3d')  
            # plt.title('point cloud1')
            # for i in range(numkernel):  
            #     col = random.sample(range(1, 100), 3)
            #     ax1.scatter(point1[i,0],point1[i,1],point1[i,2],color=[float(col[0])/100.0, float(col[1])/100.0, float(col[2])/100.0],marker='.',s=10,linewidth=1,alpha=1,cmap='spectral')  
            # ax1.axis('scaled') 
            # ax1.set_xlabel('X Label')  
            # ax1.set_ylabel('Y Label')  
            # ax1.set_zlabel('Z Label') 

            # fig=plt.figure(dpi=120)  
            # ax=fig.add_subplot(111,projection='3d')  
            # plt.title('point cloud')
            # for i in range(numkernel):  
            #     col = random.sample(range(1, 100), 3)
            #     ax.scatter(point[i,:,0],point[i,:,1],point[i,:,2],color=[float(col[0])/100.0, float(col[1])/100.0, float(col[2])/100.0],marker='.',s=10,linewidth=1,alpha=1,cmap='spectral')  
            # ax.axis('scaled') 
            # ax.set_xlabel('X Label')  
            # ax.set_ylabel('Y Label')  
            # ax.set_zlabel('Z Label')  
            # plt.show()  
            
            #log_string('batch num---' + str(batch_idx) + '-----')
            feed_dict = {ops['pointclouds_pl']: train_data,
                         ops['pointclouds_pl_m4']: train_data_m4,
                         ops['pointclouds_kernel']: idx_data,
                         ops['pointclouds_all']: train_data_all,
                         ops['labels_pl']: train_label,
                         ops['is_training_pl']: is_training,}
            summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
            train_writer.add_summary(summary, step)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += loss_val
        
        log_string('mean loss: %f' % (loss_sum / float(num_batches)))
        log_string('accuracy: %f' % (total_correct / float(total_seen)))
        allaccuracy += (total_correct / float(total_seen))

    return (allaccuracy / float(len(TRAIN_FILES)))
            #print(train_data.shape)
            #print(train_label.shape)
            
            # #disply point cloud
            # point = train_data[0][:][:][:]
            # print(point.shape)
            # print(train_label[0])    
            # fig=plt.figure(dpi=120)  
            # ax=fig.add_subplot(111,projection='3d')  
            # plt.title('point cloud')
            # for i in range(128):  
            #     col = random.sample(range(1, 100), 3)
            #     ax.scatter(point[i,:,0],point[i,:,1],point[i,:,2],color=[float(col[0])/100.0, float(col[1])/100.0, float(col[2])/100.0],marker='.',s=10,linewidth=1,alpha=1,cmap='spectral')  
            # ax.axis('scaled') 
            # ax.set_xlabel('X Label')  
            # ax.set_ylabel('Y Label')  
            # ax.set_zlabel('Z Label')  
            # plt.show()  

# def eval_one_epoch(sess, ops, test_writer):
#     """ ops: dict mapping from string to tf ops """
#     is_training = False #test
#     total_correct = 0
#     total_seen = 0
#     loss_sum = 0
#     total_seen_class = [0 for _ in range(NUM_CLASSES)]
#     total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
#     for fn in range(len(TEST_FILES)):
#         log_string('train file NUM---' + str(fn) + '-----')
#         current_data, current_label = loadDataFile(TEST_FILES[fn])
#         current_data = current_data[:,0:NUM_POINT,:]
#         current_label = np.squeeze(current_label)
        
#         file_size = current_data.shape[0]
#         num_batches = file_size // BATCH_SIZE
#         log_string('test file size---' + str(file_size) + '-----')

#         for batch_idx in range(num_batches):
#             start_idx = batch_idx * BATCH_SIZE
#             end_idx = (batch_idx+1) * BATCH_SIZE
#             test_data_p  = current_data[start_idx:end_idx, :, :]
#             test_data=np.zeros([BATCH_SIZE,numkernel,pgnump,3])

#             for point_idx in range (BATCH_SIZE):
#                 rerowcol=test_data_p[point_idx]
#                 KernelList=random.sample(range(rerowcol.shape[0]),numkernel)
#                 #rerowcol_downsample is downsample point
#                 rerowcol_downsample=rerowcol[KernelList,:]
#                 kdt = KDTree(rerowcol, leaf_size=30, metric='euclidean')
#                 pgi=kdt.query(rerowcol_downsample, k=pgnump, return_distance=False)
#                 pgr=np.zeros([numkernel,pgnump,3])
#                 for n in range(pgi.shape[0]):
#                   pgr[n]=rerowcol[pgi[n,:],:]
#                 test_data[point_idx]=pgr

#             feed_dict = {ops['pointclouds_pl']: test_data,
#                          ops['labels_pl']: current_label[start_idx:end_idx],
#                          ops['is_training_pl']: is_training}
#             summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
#                 ops['loss'], ops['pred']], feed_dict=feed_dict)
            
#             #test_writer.add_summary(summary, step)
#             pred_val = np.argmax(pred_val, 1)
#             correct = np.sum(pred_val == current_label[start_idx:end_idx])
#             total_correct += correct
#             total_seen += BATCH_SIZE
#             loss_sum += (loss_val*BATCH_SIZE)
#             for i in range(start_idx, end_idx):
#                 l = current_label[i]
#                 total_seen_class[l] += 1
#                 total_correct_class[l] += (pred_val[i-start_idx] == l)
            
#     log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
#     log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
#     log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))

def eval_one_epoch(sess, ops, test_writer):
    is_training = False
    #shuffle train file
    test_file_idxs = np.arange(0, len(TEST_FILES))
    #np.random.shuffle(train_file_idxs)
    
    for fn in range(len(TEST_FILES)):
        log_string('test file NUM---' + str(fn) + '-----')
        current_data, current_label = loadDataFile(TEST_FILES[test_file_idxs[fn]])
        current_data = current_data[:,0:NUM_POINT,:]
        current_data, current_label, _ = shuffle_data(current_data, np.squeeze(current_label))            
        current_label = np.squeeze(current_label)
        file_size = current_data.shape[0]

        num_batches = file_size // BATCH_SIZE
        log_string('test file size---' + str(file_size) + '-----')
        
        total_correct = 0
        total_seen = 0
        loss_sum = 0

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            
            test_data  = current_data[start_idx:end_idx, :, :]
            test_label = current_label[start_idx:end_idx]
            test_data_pg, test_data_pg_m4, idx_data,test_data_all= point_group_first(test_data,test_data,numkernel,pgnump)
            
            print('test batch',batch_idx)
            #log_string('batch num---' + str(batch_idx) + '-----')
            feed_dict = {ops['pointclouds_pl']: test_data_pg,
                         ops['pointclouds_pl_m4']: test_data_pg_m4,
                         ops['pointclouds_kernel']: idx_data,
                         ops['pointclouds_all']: test_data_all,
                         ops['labels_pl']: test_label,
                         ops['is_training_pl']: is_training,}
            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
               ops['loss'], ops['pred']], feed_dict=feed_dict)
            test_writer.add_summary(summary, step)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += loss_val
        
        log_string('test mean loss: %f' % (loss_sum / float(num_batches)))
        log_string('test accuracy: %f' % (total_correct / float(total_seen)))



if __name__ == "__main__":
    train()
    LOG_FOUT.close()
