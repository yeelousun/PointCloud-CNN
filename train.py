import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import random 


from collections import Counter
from sklearn.neighbors import KDTree 
from mpl_toolkits.mplot3d import Axes3D
from input.Inputdatas import * 
from model.model import * 
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))



LOG_DIR = 'log'
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
#LOG_FOUT.write(str(FLAGS)+'\n')
NUM_CLASSES = 40
NUM_POINT  = 2048
BATCH_SIZE = 32
numkernel  = 256
pgnump     = 32

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
            #make feature1 w and b
            outf = 64
            PGV = 2
            midHid = 10
            weightsf1_1 = _variable_with_weight_decay('weights1_1', [outf, PGV, midHid], stddev=1.0 / PGV, activation="selu", wd=0.0004)
            biasesf1_1 = tf.get_variable(name='biases1_1', shape=[outf,midHid], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
            weightsf1_2 = _variable_with_weight_decay('weights1_2', [outf, midHid, midHid], stddev=1.0 / midHid, activation="selu", wd=0.0004)
            biasesf1_2 = tf.get_variable(name='biases1_2', shape=[outf,midHid], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
            weightsf1_3 = _variable_with_weight_decay('weights1_3', [outf, midHid, 1], stddev=1.0 / midHid, activation="selu", wd=0.0)
            biasesf1_3 = tf.get_variable(name='biases1_3', shape=[outf,1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
            #make feature2 w and b
            outf = 64
            PGV = 63
            midHid = 128
            weightsf2_1 = _variable_with_weight_decay('weights2_1', [outf, PGV, midHid], stddev=1.0 / PGV, activation="selu", wd=0.0004)
            biasesf2_1 = tf.get_variable(name='biases2_1', shape=[outf,midHid], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
            weightsf2_2 = _variable_with_weight_decay('weights2_2', [outf, midHid, midHid], stddev=1.0 / midHid, activation="selu", wd=0.0004)
            biasesf2_2 = tf.get_variable(name='biases2_2', shape=[outf,midHid], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
            weightsf2_3 = _variable_with_weight_decay('weights2_3', [outf, midHid, 1], stddev=1.0 / midHid, activation="selu", wd=0.0)
            biasesf2_3 = tf.get_variable(name='biases2_3', shape=[outf,1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
            #make feature3 w and b
            outf = 64
            PGV = 63
            midHid = 128
            weightsf3_1 = _variable_with_weight_decay('weights3_1', [outf, PGV, midHid], stddev=1.0 / PGV, activation="selu", wd=0.0004)
            biasesf3_1 = tf.get_variable(name='biases3_1', shape=[outf,midHid], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
            weightsf3_2 = _variable_with_weight_decay('weights3_2', [outf, midHid, midHid], stddev=1.0 / midHid, activation="selu", wd=0.0004)
            biasesf3_2 = tf.get_variable(name='biases3_2', shape=[outf,midHid], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
            weightsf3_3 = _variable_with_weight_decay('weights3_3', [outf, midHid, 1], stddev=1.0 / midHid, activation="selu", wd=0.0)
            biasesf3_3 = tf.get_variable(name='biases3_3', shape=[outf,1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
            #make featureall w and b
            outf = 128
            PGV = 2
            midHid = 10
            weightsfa_1 = _variable_with_weight_decay('weightsa_1', [outf, PGV, midHid], stddev=1.0 / PGV, activation="selu", wd=0.0004)
            biasesfa_1 = tf.get_variable(name='biasesa_1', shape=[outf,midHid], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
            weightsfa_2 = _variable_with_weight_decay('weightsa_2', [outf, midHid, midHid], stddev=1.0 / midHid, activation="selu", wd=0.0004)
            biasesfa_2 = tf.get_variable(name='biasesa_2', shape=[outf,midHid], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
            weightsfa_3 = _variable_with_weight_decay('weightsa_3', [outf, midHid, 1], stddev=1.0 / midHid, activation="selu", wd=0.0)
            biasesfa_3 = tf.get_variable(name='biasesa_3', shape=[outf,1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)


            pointclouds_pl, pointclouds_kernel, pointclouds_all ,labels_pl = placeholder_inputs(BATCH_SIZE, numkernel, pgnump)    
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            
            # Get model and loss
            print ('make loss1')
            pointf1_x = tf.placeholder(tf.float32, [None, 2]) 
            pointf1_y = tf.placeholder(tf.float32, [None, 1]) 
            num1  = get_feature_num (pointf1_x,pointf1_y,weightsf1_1,weightsf1_2,weightsf1_3,biasesf1_1,biasesf1_2,biasesf1_3)
            loss1 = get_feature_loss(pointf1_x,pointf1_y,weightsf1_1,weightsf1_2,weightsf1_3,biasesf1_1,biasesf1_2,biasesf1_3,num1)
            print ('make loss2')
            pointf2_x = tf.placeholder(tf.float32, [None, 63]) 
            pointf2_y = tf.placeholder(tf.float32, [None, 1]) 
            num2  = get_feature_num (pointf2_x,pointf2_y,weightsf2_1,weightsf2_2,weightsf2_3,biasesf2_1,biasesf2_2,biasesf2_3)
            loss2 = get_feature_loss(pointf2_x,pointf2_y,weightsf2_1,weightsf2_2,weightsf2_3,biasesf2_1,biasesf2_2,biasesf2_3,num2)
            print ('make loss3')
            pointf3_x = tf.placeholder(tf.float32, [None, 63]) 
            pointf3_y = tf.placeholder(tf.float32, [None, 1]) 
            num3  = get_feature_num (pointf3_x,pointf3_y,weightsf3_1,weightsf3_2,weightsf3_3,biasesf3_1,biasesf3_2,biasesf3_3)
            loss3 = get_feature_loss(pointf3_x,pointf3_y,weightsf3_1,weightsf3_2,weightsf3_3,biasesf3_1,biasesf3_2,biasesf3_3,num3)
            print ('make lossa')
            pointa_x = tf.placeholder(tf.float32, [None, 2]) 
            pointa_y = tf.placeholder(tf.float32, [None, 1]) 
            numa  = get_feature_num (pointa_x,pointa_y,weightsfa_1,weightsfa_2,weightsfa_3,biasesfa_1,biasesfa_2,biasesfa_3)
            lossa = get_feature_loss(pointa_x,pointa_y,weightsfa_1,weightsfa_2,weightsfa_3,biasesfa_1,biasesfa_2,biasesfa_3,numa)

            print ('make model')
            pred,predf1,predf2  = get_model(pointclouds_pl, pointclouds_kernel, pointclouds_all, 
                                            weightsf1_1,weightsf1_2,weightsf1_2,biasesf1_1,biasesf1_2,biasesf1_3,
                                            weightsf2_1,weightsf2_2,weightsf2_2,biasesf2_1,biasesf2_2,biasesf2_3,
                                            weightsf3_1,weightsf3_2,weightsf3_2,biasesf3_1,biasesf3_2,biasesf3_3,
                                            weightsfa_1,weightsfa_2,weightsfa_3,biasesfa_1,biasesfa_2,biasesfa_3,
                                            is_training_pl)
            print ('make loss')
            loss = get_loss(pred, labels_pl)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)
            print ('make operator')
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            print('train_op')
            optimizerf1 = tf.train.MomentumOptimizer(0.05,0.9)
            optimizerf2 = tf.train.MomentumOptimizer(0.05,0.9)
            optimizerf3 = tf.train.MomentumOptimizer(0.05,0.9)
            optimizerfa = tf.train.MomentumOptimizer(0.05,0.9)
            
            print('train_f')
            trainf1 = optimizerf1.minimize(loss1)
            trainf2 = optimizerf2.minimize(loss2)
            trainf3 = optimizerf3.minimize(loss3)
            trainfa = optimizerfa.minimize(lossa)
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
        print ('make init')
        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'pointclouds_kernel':pointclouds_kernel,
               'pointclouds_all':pointclouds_all,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'predf1': predf1,
               'predf2': predf2,
               'loss': loss,
               'loss1': loss1,
               'loss2': loss2,
               'loss3': loss3,
               'lossa': lossa,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'pointf1_x':pointf1_x,
               'pointf2_x':pointf2_x,
               'pointf3_x':pointf3_x,
               'pointa_x':pointa_x,
               'pointf1_y':pointf1_y,
               'pointf2_y':pointf2_y,
               'pointf3_y':pointf3_y,
               'pointa_y':pointa_y,
               'trainf1':trainf1,
               'trainf2':trainf2,
               'trainf3':trainf3,
               'trainfa':trainfa}
        
        max_acc=0

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            allaccuracy = train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)
            
            # Save the variables to disk.
            if epoch % 10 == 0:
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
            train_data, idx_data, train_data_all = point_group_first(train_data_rt,train_data_rt,numkernel,pgnump)
            print('train batch',batch_idx)
            
            #feature all
            for featurea_batch_idx in range (BATCH_SIZE):
                featurea_train = train_data_all[featurea_batch_idx,:,0:2]
                featurea_label = train_data_all[featurea_batch_idx,:,2:3]
                feed_dict = {ops['pointa_x']: featurea_train,
                            ops['pointa_y']: featurea_label,}
                count = 0
                while count< 100: 
                    featurea_loss,_ = sess.run([ops['lossa'], ops['trainfa']], feed_dict=feed_dict)
                    count += 1
                    print('feature1_loss', count, featurea_loss)

            #feature 1
            for feature1_batch_idx in range (BATCH_SIZE):
                for feature1_kernel_idx in range (numkernel):
                    feature1_train = train_data[feature1_batch_idx,feature1_kernel_idx,:,0:2]
                    feature1_label = train_data[feature1_batch_idx,feature1_kernel_idx,:,2:3]
                    feed_dict = {ops['pointf1_x']: feature1_train,
                                 ops['pointf1_y']: feature1_label,}
                    count = 0
                    while count< 100: 
                        feature1_loss,_ = sess.run([ops['loss1'], ops['trainf1']], feed_dict=feed_dict)
                        count += 1
                        print('feature1_loss', count, feature1_loss)
            
            feed_dict = {ops['pointclouds_pl']: train_data,
                         ops['pointclouds_kernel']: idx_data,
                         ops['pointclouds_all']: train_data_all,
                         ops['labels_pl']: train_label,
                         ops['is_training_pl']: is_training,}

            feature1_res = sess.run([ops['predf1']], feed_dict = feed_dict)


            #feature 2
            for feature2_batch_idx in range (BATCH_SIZE):
                for feature2_kernel_idx in range (32):
                    feature2_train = feature1_res[feature2_batch_idx,feature2_kernel_idx,:,0:63]
                    feature2_label = feature1_res[feature2_batch_idx,feature2_kernel_idx,:,63:64]
                    feed_dict = {ops['pointf2_x']: feature2_train,
                                 ops['pointf2_y']: feature2_label,}
                    count = 0
                    while count< 100: 
                        feature2_loss,_ = sess.run([ops['loss2'], ops['trainf2']], feed_dict=feed_dict)
                        count += 1
                        print('feature2_loss', count, feature2_loss)
            
            feed_dict = {ops['pointclouds_pl']: train_data,
                         ops['pointclouds_kernel']: idx_data,
                         ops['pointclouds_all']: train_data_all,
                         ops['labels_pl']: train_label,
                         ops['is_training_pl']: is_training,}

            feature2_res = sess.run([ops['predf2']], feed_dict = feed_dict)

            #feature 3
            for feature3_batch_idx in range (BATCH_SIZE):
                for feature3_kernel_idx in range (1):
                    feature3_train = feature2_res[feature3_batch_idx,feature3_kernel_idx,:,0:63]
                    feature3_label = feature2_res[feature3_batch_idx,feature3_kernel_idx,:,63:64]
                    feed_dict = {ops['pointf3_x']: feature3_train,
                                 ops['pointf3_y']: feature3_label,}
                    count = 0
                    while count< 100: 
                        feature3_loss,_ = sess.run([ops['loss3'], ops['trainf3']], feed_dict=feed_dict)
                        count += 1
                        print('feature3_loss', count, feature3_loss)
           

            ##train feature1 
            
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
            test_data_pg, idx_data,test_data_all= point_group_first(test_data,test_data,numkernel,pgnump)
            
            print('test batch',batch_idx)
            #log_string('batch num---' + str(batch_idx) + '-----')
            feed_dict = {ops['pointclouds_pl']: test_data_pg,
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

def _variable_with_weight_decay(name, shape, activation, stddev, wd=None):    
    # Determine number of input features from shape
    f_in = np.prod(shape[:-1]) if len(shape) == 4 else shape[0]
    
    # Calculate sdev for initialization according to activation function
    if activation == selu:
        sdev = sqrt(1 / f_in)
    elif activation == tf.nn.relu:
        sdev = sqrt(2 / f_in)
    elif activation == tf.nn.elu:
        sdev = sqrt(1.5505188080679277 / f_in)
    else:
        sdev = stddev
    
    var = tf.get_variable(name=name, shape=shape,
                          initializer=tf.truncated_normal_initializer(stddev=sdev, dtype=tf.float32))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

if __name__ == "__main__":
    train()
    LOG_FOUT.close()
