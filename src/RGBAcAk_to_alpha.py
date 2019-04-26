#%%
import tensorflow as tf
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mp
import copy
import random


#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#%%
# 进度条
import sys, time

class ProgressBar:
    def __init__(self, count = 0, total = 0, width = 500):
        self.count = count
        self.total = total
        self.width = width
    def move(self):
        self.count += 1
    def log(self, s):
        sys.stdout.write(' ' * (self.width + 9) + '\r')
        sys.stdout.flush()
#        print(s)
        progress = self.width * self.count / self.total
        sys.stdout.write('{0:3}/{1:3}: '.format(self.count, self.total))
        sys.stdout.write('#' * int(progress) + '-' * int(self.width - progress) + '\r')
        if progress == self.width:
            sys.stdout.write('\n')
        sys.stdout.flush()

#bar = ProgressBar(total = 10)
#for i in range(10):
#    bar.move()
#    bar.log('We have arrived at: ' + str(i + 1))
#    time.sleep(1)

#%%       
# 参数们
patch_batch_num=3000
class_select_num = 300#300#30#300
crop_num = 99999#99999#1000#9999

test_stride_size=1

channel_num=5
read_data = 0
train=0
test=1-train
train_num=600
img_height=800#513#800
img_width=600#288#600
batch_num=1
patch_size=27
patch_half_size = np.int32((patch_size - 1)/2)
each_class_num = 4
dataset_dir="../dataset/"
train_dir=os.path.join(dataset_dir,"train/")
test_dir=os.path.join(dataset_dir,"test/")
model_dir="../dataset/model/model_RGBAcAk/"
meta_file=os.path.join(model_dir,'my-model-600.meta')
TFrecords_name="data_train_RGBAcAk.tfrecords"
saveName="_RGBAcAk"

#%%
# 读数据
def read_data_from_file(data_dir,fname):
    rgb_dir = os.path.join(data_dir,"rgb/")
    alpha_c_dir = os.path.join(data_dir,"alpha_c/")
    alpha_k_dir = os.path.join(data_dir,"alpha_k/")
    alpha_dir = os.path.join(data_dir,"alpha/")

    
    fpath_rgb = os.path.join(rgb_dir, fname)  
    fpath_alpha_c = os.path.join(alpha_c_dir, fname)  
    fpath_alpha_k = os.path.join(alpha_k_dir, fname)  
    fpath_alpha = os.path.join(alpha_dir, fname)


    image_rgb = Image.open(fpath_rgb)   
    image_alpha_c = Image.open(fpath_alpha_c)   
    image_alpha_k = Image.open(fpath_alpha_k)   
    image_alpha = Image.open(fpath_alpha)  

    

    
       
    data_rgb = np.array(image_rgb)
    data_alpha_c = np.array(image_alpha_c)/255.0
    data_alpha_k = np.array(image_alpha_k)/255.0
    data_alpha = np.array(image_alpha)/255.0
#        data_rgb2 = np.sum(data_rgb*data_rgb,axis=2)
#        data_rgb2[data_rgb2==0]=1
#        data_rgb2 = 1 / data_rgb2
#        data_rgb3 = np.concatenate([data_rgb2,data_rgb2,data_rgb2],axis=0).reshape(data_rgb.shape)
#        data = data_rgb * data_rgb3 
    data = data_rgb/255.0
    # R G B Ac Ak
    data = np.concatenate([data[:,:,0].reshape(data_rgb.shape[0],data_rgb.shape[1],1), 
                           data[:,:,1].reshape(data_rgb.shape[0],data_rgb.shape[1],1), 
                           data[:,:,2].reshape(data_rgb.shape[0],data_rgb.shape[1],1), 
                           data_alpha_c.reshape(data_rgb.shape[0],data_rgb.shape[1],1), 
                           data_alpha_k.reshape(data_rgb.shape[0],data_rgb.shape[1],1)],
                        axis=2)
        



    label =  np.array(data_alpha).reshape(data.shape[0],data.shape[1],1)
    return data,label
    
# 读取本地数据存成TFRecord文件
def read_data_to_tfrecords(data_dir):
    rgb_dir = os.path.join(train_dir,"rgb/")

    writer= tf.python_io.TFRecordWriter(os.path.join(data_dir, TFrecords_name)) #要生成的文件
    i=0
    for fname in os.listdir(rgb_dir): 
        i+=1
        if i%10==0:
            print(fname)
#            break

        data,label=read_data_from_file(data_dir,fname)


#        plt.figure(figsize=(20, 100))
#        plt.subplot(1,4,1)
#        plt.imshow(data[:,:,0:3])
#        plt.subplot(1,4,2)
#        plt.imshow(np.concatenate([data[:,:,3].reshape(data.shape[0],data.shape[1],1),
#                                   data[:,:,3].reshape(data.shape[0],data.shape[1],1),
#                                   data[:,:,3].reshape(data.shape[0],data.shape[1],1)],
#            axis=2))
#        plt.subplot(1,4,3)
#        plt.imshow(np.concatenate([data[:,:,4].reshape(data.shape[0],data.shape[1],1),
#                                   data[:,:,4].reshape(data.shape[0],data.shape[1],1),
#                                   data[:,:,4].reshape(data.shape[0],data.shape[1],1)],
#        axis=2))
#        plt.subplot(1,4,4)
#        plt.imshow(np.concatenate([label,label,label],axis=2))
#        plt.show()



        data_raw=data.tobytes()
        label_raw=label.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw])),
            'data_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_raw])),
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[data.shape[0]])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[data.shape[1]]))
        })) #example对象对label和image数据进行封装
        writer.write(example.SerializeToString())  #序列化为字符串
    writer.close()

#%%
def define_model(x_image):
    # 一，函数声明部分  
    def weight_variable(shape):  
        # 正态分布，标准差为0.1，默认最大为1，最小为-1，均值为0  
            initial = tf.truncated_normal(shape, stddev=0.1)  
            return tf.Variable(initial,name="w")  
    def bias_variable(shape):  
        # 创建一个结构为shape矩阵也可以说是数组shape声明其行列，初始化所有值为0.1  
            initial = tf.constant(0.1, shape=shape)  
            return tf.Variable(initial,name="b")  
    def conv2d(x, W):    
        # 卷积遍历各方向步数为1，SAME：边缘外自动补0，遍历相乘  
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')    
    def max_pool_2x2(x):    
        # 池化卷积结果（conv2d）池化层采用kernel大小为2*2，步数也为2，周围补0，取最大值。数据量缩小了4倍  
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')    
    
    with tf.variable_scope("conv1"):
        weights = weight_variable([9, 9, channel_num, 64])
        biases = bias_variable([64])
        conv1 = tf.nn.relu(conv2d(x_image, weights) + biases)
    
    with tf.variable_scope("conv2"):
        weights = weight_variable([1, 1, 64, 64])
        biases = bias_variable([64])
        conv2 = tf.nn.relu(conv2d(conv1, weights) + biases)
    
    with tf.variable_scope("conv3"):
        weights = weight_variable([1, 1, 64, 64])
        biases = bias_variable([64])
        conv3 = tf.nn.relu(conv2d(conv2, weights) + biases)
    
    with tf.variable_scope("conv4"):
        weights = weight_variable([1, 1, 64, 64])
        biases = bias_variable([64])
        conv4 = tf.nn.relu(conv2d(conv3, weights) + biases)
    
    with tf.variable_scope("conv5"):
        weights = weight_variable([1, 1, 64, 64])
        biases = bias_variable([64])
        conv5 = tf.nn.relu(conv2d(conv4, weights) + biases)
    
    with tf.variable_scope("conv6"):
        weights = weight_variable([5, 5, 64, 1])
        biases = bias_variable([1])
        conv6 = tf.nn.relu(conv2d(conv5, weights) + biases)
    
    return conv6

#%%
# 读取数据  
if read_data:
    read_data_to_tfrecords(train_dir)

#%%    
xs = tf.placeholder(tf.float32, [None, patch_size, patch_size, channel_num],name="xs")   
ys = tf.placeholder(tf.float32, [None, patch_size, patch_size, 1],name="ys")
        
x_image = xs
y_image = ys

initial_learning_rate = 0.001
y_conv = define_model(x_image)
  
cross_entropy = tf.reduce_mean(tf.squared_difference(y_image,y_conv),name="loss") # 定义交叉熵为loss函数    
train_step = tf.train.AdamOptimizer(learning_rate=initial_learning_rate, beta1=0.9, beta2=0.999,
                                    epsilon=1e-08).minimize(cross_entropy) # 调用优化器优化，其实就是通过喂数据争取cross_entropy最小化    

saver = tf.train.Saver(max_to_keep=4)
tf.add_to_collection("predict",y_conv)

#tf.reset_default_graph()
with tf.Session() as sess:
    if train:
        feature = {'label_raw': tf.FixedLenFeature([], tf.string),
                   'data_raw': tf.FixedLenFeature([], tf.string),
                   'height': tf.FixedLenFeature([], tf.int64),
                   'width': tf.FixedLenFeature([], tf.int64)}
        # define a queue base on input filenames
        filename_queue = tf.train.string_input_producer([os.path.join(train_dir, TFrecords_name)], num_epochs=1)
        # define a tfrecords file reader
        reader = tf.TFRecordReader()
        # read in serialized example data
        _, serialized_example = reader.read(filename_queue)
        # decode example by feature
        features = tf.parse_single_example(serialized_example, features=feature)
        
        img = tf.decode_raw(features['data_raw'], tf.float64)
        label = tf.decode_raw(features['label_raw'], tf.float64)
        height = tf.cast(features['height'], tf.int64)
        width = tf.cast(features['width'], tf.int64)
    #    print("height")
    #    print(height.eval())
        # restore image to [height, width, channel]
        img = tf.reshape(img, [img_height, img_width, channel_num])  
        img = tf.cast(img, tf.float64) #在流中抛出img张量
        label = tf.reshape(label, [img_height, img_width,1])  
        label = tf.cast(label, tf.float64) #在流中抛出label张量
        
        # create bathch
        images, labels = tf.train.shuffle_batch(
                [img, label], batch_size=batch_num, capacity=30, 
                num_threads=1, min_after_dequeue=10) 
        # capacity是队列的最大容量，num_threads是dequeue后最小的队列大小，num_threads是进行队列操作的线程数。
    
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        for batch_index in range(np.int32(train_num/batch_num)):
            batch_images, batch_labels = sess.run([images, labels])
            
            # random crop image to patch
            seeds=[]
            for i in range(each_class_num*each_class_num):
                seeds.append([])
            for batch_crop_index in range(batch_num):
                for crop_index in range(crop_num):
                    seed_h = np.random.randint(patch_half_size, img_height-patch_half_size-1)
                    seed_w = np.random.randint(patch_half_size, img_width-patch_half_size-1)
                    label_patch=batch_labels[batch_crop_index,seed_h-patch_half_size:seed_h+patch_half_size+1,seed_w-patch_half_size:seed_w+patch_half_size+1,:]
                    label_patch_gradient=np.gradient(label_patch.reshape(patch_size,patch_size))
                    n1=np.sum(label_patch==0)
                    n2=np.sum((label_patch_gradient[0]==0)*(label_patch_gradient[1]==0))
                    if n1/(patch_size*patch_size) <1/4:
                        class1=0
                    elif n1/(patch_size*patch_size) <2/4:
                        class1=1
                    elif n1/(patch_size*patch_size) <3/4:
                        class1=2
                    else:
                        class1=3
                    if n2/(patch_size*patch_size) <1/4:
                        class2=0
                    elif n2/(patch_size*patch_size) <2/4:
                        class2=1
                    elif n2/(patch_size*patch_size) <3/4:
                        class2=2
                    else:
                        class2=3
                    class_num = class1*each_class_num+class2
                    seeds[class_num].append([seed_h,seed_w])
                    
            input_images=[]
            label_images=[]
            for i in range(len(seeds)):
                if len(seeds[i])==0:
                    continue
                else:
                    for j in range(min(len(seeds[i]),class_select_num)):
                        seed_h=seeds[i][j][0]
                        seed_w=seeds[i][j][1]
                        input_patch=batch_images[batch_crop_index,seed_h-patch_half_size:seed_h+patch_half_size+1,seed_w-patch_half_size:seed_w+patch_half_size+1,:]
                        label_patch=batch_labels[batch_crop_index,seed_h-patch_half_size:seed_h+patch_half_size+1,seed_w-patch_half_size:seed_w+patch_half_size+1,:]
                        input_images.append(input_patch)
                        label_images.append(label_patch)
            
            sele_index=np.array(range(len(input_images)))
            random.shuffle(sele_index)
            for patch_batch_index in range(max(np.int32(len(input_images)/patch_batch_num),1)):
            
                input_patch_images=[]
                label_patch_images=[]
                for sele_cur in range(patch_batch_index*patch_batch_num,(patch_batch_index+1)*patch_batch_num):
                    input_patch_images.append(input_images[sele_index[sele_cur%len(sele_index)]])
                    label_patch_images.append(label_images[sele_index[sele_cur%len(sele_index)]])
                input_patch_images=np.array(input_patch_images)
                label_patch_images=np.array(label_patch_images)           
            
                train_feed_dict={xs:input_patch_images,ys:label_patch_images}
                print(input_patch_images.shape)
                print(label_patch_images.shape)
                if batch_index == 0 or (batch_index+1) % 10 == 0:  
                    print("batch_index: ", batch_index+1)
                    loss=cross_entropy.eval(feed_dict=train_feed_dict)
                    print("patch_batch_index: ",(patch_batch_index+1) ,": loss: ",loss)
                    saver.save(sess,os.path.join(model_dir,"my-model"), global_step=(batch_index+1))
                    for i in range(input_patch_images.shape[0]):
                        if i%100==0:
                            res_ = y_conv.eval(feed_dict=train_feed_dict)[i,:,:,:]
                            alpha_ = label_patch_images[i,:,:].reshape(patch_size,patch_size,1)
                            rgb_ = input_patch_images[i,:,:,0:3].reshape(patch_size,patch_size,3)
                            alpha_c_ = input_patch_images[i,:,:,3].reshape(patch_size,patch_size,1)
                            alpha_k_ = input_patch_images[i,:,:,4].reshape(patch_size,patch_size,1)
                            
                            plt.figure(figsize=(20, 100))
                            plt.subplot(1,5,1)
                            plt.imshow(rgb_)
                            plt.subplot(1,5,2)
                            plt.imshow(np.concatenate([alpha_c_,alpha_c_,alpha_c_],axis=2))
                            plt.subplot(1,5,3)
                            plt.imshow(np.concatenate([alpha_k_,alpha_k_,alpha_k_],axis=2))
                            plt.subplot(1,5,4)
                            plt.imshow(np.concatenate([res_,res_,res_],axis=2))
                            plt.subplot(1,5,5)
                            plt.imshow(np.concatenate([alpha_,alpha_,alpha_],axis=2))
                            plt.show()
            train_step.run(feed_dict=train_feed_dict) 
        coord.request_stop()
        coord.join(threads)
        sess.close()
        
    if test:
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        
        # load the meta graph and weights
        saver = tf.train.import_meta_graph(meta_file)
        graph = tf.get_default_graph()
        allNames=[n.name for n in graph.as_graph_def().node]
        saver.restore(sess,tf.train.latest_checkpoint(model_dir))
        

        #prepare the test data
        rgb_dir=os.path.join(test_dir,"rgb/")
        result_dir=os.path.join(test_dir,"result_RGBAcAk_"+str(test_stride_size)+"/")
        i=0
        for fname in os.listdir(rgb_dir): 
            i+=1
            if i%10==0:
                print(fname)

            data,label=read_data_from_file(test_dir,fname)
            
            data = data.reshape(1,img_height,img_width,channel_num)
            label = label.reshape(1,img_height,img_width,1)
            
            res_=np.zeros(((img_height,img_width,1)))
            sum_=np.zeros(((img_height,img_width,1)))
            
            hw_index=[]
            test_data_patch=[]
            test_label_patch=[]
            h=patch_half_size
            while h < img_height-patch_half_size:
#                print("h: ", h)
                w=patch_half_size
                while w < img_width-patch_half_size:
                    test_data_patch_1=data[:,h-patch_half_size:h+patch_half_size+1,w-patch_half_size:w+patch_half_size+1,:].reshape(patch_size,patch_size,data.shape[3])
                    test_label_patch_1=label[:,h-patch_half_size:h+patch_half_size+1,w-patch_half_size:w+patch_half_size+1,:].reshape(patch_size,patch_size,label.shape[3])
                    
                    hw_index.append([h,w])
                    test_data_patch.append(test_data_patch_1)
                    test_label_patch.append(test_label_patch_1)
                    
                    if(len(hw_index)==patch_batch_num):
                        test_data_patch=np.array(test_data_patch)
                        test_label_patch=np.array(test_label_patch)
                        
                        test_feed_dict={"xs:0":test_data_patch,"ys:0":test_label_patch}
                    
    #                    y = graph.get_tensor_by_name("ys:0")
    #                    yy = sess.run(y, test_feed_dict)
                        
                        pred_y = tf.get_collection("predict")
                        pred = sess.run(pred_y, test_feed_dict)[0]
                        
            #            loss = graph.get_operation_by_name("loss")
#                        loss = graph.get_tensor_by_name("loss:0")
#                        loss = sess.run(loss,test_feed_dict)
                        
                        for res_index in range(pred.shape[0]):
                            h_cur=hw_index[res_index][0]
                            w_cur=hw_index[res_index][1]
                            res_[h_cur-patch_half_size:h_cur+patch_half_size+1,w_cur-patch_half_size:w_cur+patch_half_size+1,:]+=pred[res_index,:,:,:].reshape(patch_size,patch_size,1)
                            sum_[h_cur-patch_half_size:h_cur+patch_half_size+1,w_cur-patch_half_size:w_cur+patch_half_size+1,:]+=1
                        hw_index=[]
                        test_data_patch=[]
                        test_label_patch=[]
                    
                    if w+test_stride_size>img_width-patch_half_size-1 and w+patch_half_size!=img_width-1:
                        w=img_width-patch_half_size-1-test_stride_size
#                    w+=patch_size
                    w+=test_stride_size
                
                if h+test_stride_size>img_height-patch_half_size-1 and h+patch_half_size!=img_height-1:
                    h=img_height-patch_half_size-1-test_stride_size
#                h+=patch_size
                    
#                if h%100==0:
#                    print(h)
                h+=test_stride_size
               
            if len(hw_index):
                cur_num=len(hw_index)
                for t in range(cur_num,patch_batch_num):
                    hw_index.append(hw_index[t%cur_num])
                    test_data_patch.append(test_data_patch[t%cur_num])
                    test_label_patch.append(test_label_patch[t%cur_num])
                test_data_patch=np.array(test_data_patch)
                test_label_patch=np.array(test_label_patch)
                
                test_feed_dict={"xs:0":test_data_patch,"ys:0":test_label_patch}
            
#                    y = graph.get_tensor_by_name("ys:0")
#                    yy = sess.run(y, test_feed_dict)
                
                pred_y = tf.get_collection("predict")
                pred = sess.run(pred_y, test_feed_dict)[0]
                
    #            loss = graph.get_operation_by_name("loss")
#                        loss = graph.get_tensor_by_name("loss:0")
#                        loss = sess.run(loss,test_feed_dict)
                
                for res_index in range(pred.shape[0]):
                    h_cur=hw_index[res_index][0]
                    w_cur=hw_index[res_index][1]
                    res_[h_cur-patch_half_size:h_cur+patch_half_size+1,w_cur-patch_half_size:w_cur+patch_half_size+1,:]+=pred[res_index,:,:,:].reshape(patch_size,patch_size,1)
                    sum_[h_cur-patch_half_size:h_cur+patch_half_size+1,w_cur-patch_half_size:w_cur+patch_half_size+1,:]+=1
            hw_index=[]
            test_data_patch=[]
            test_label_patch=[]
                           
            res_=res_/sum_
            result_=copy.deepcopy(res_)
            alpha_ = label.reshape(img_height,img_width,1)
            rgb_ = data[:,:,:,0:3].reshape(img_height,img_width,3)
            alpha_c_ = data[:,:,:,3].reshape(img_height,img_width,1)
            alpha_k_ = data[:,:,:,4].reshape(img_height,img_width,1)
            
            simaliar_=np.abs(alpha_c_-alpha_k_)<0.01
            result_[simaliar_]=alpha_c_[simaliar_]
            
            print(i, ": ")
#            print("loss: ",loss)
            print("loss_c: ", np.sum(np.abs(alpha_c_ - result_)))
            print("loss_k: ", np.sum(np.abs(alpha_k_ - result_)))
            print("loss_gt: ", np.sum(np.abs(alpha_ - result_)))
            
            rgb_img = rgb_
            alpha_img = np.concatenate([alpha_,alpha_,alpha_],axis=2)
            alpha_c_img = np.concatenate([alpha_c_,alpha_c_,alpha_c_],axis=2)
            alpha_k_img = np.concatenate([alpha_k_,alpha_k_,alpha_k_],axis=2)
            res_img = np.concatenate([res_,res_,res_],axis=2)
            result_img = np.concatenate([result_,result_,result_],axis=2)
            
            # save results
            mp.imsave(result_dir+fname[0:5]+'_rgb.png',rgb_img)
            mp.imsave(result_dir+fname[0:5]+'_alpha.png',alpha_img)
            mp.imsave(result_dir+fname[0:5]+'_alpha_c.png',alpha_c_img)
            mp.imsave(result_dir+fname[0:5]+'_alpha_k.png',alpha_k_img)
            mp.imsave(result_dir+fname[0:5]+saveName+'_res.png',res_img)
            mp.imsave(result_dir+fname[0:5]+saveName+'_result.png',result_img)
            
            # show results
            plt.figure(figsize=(20, 100))
            plt.subplot(1,6,1)
            plt.imshow(rgb_)
            plt.subplot(1,6,2)
            plt.imshow(alpha_c_img)
            plt.subplot(1,6,3)
            plt.imshow(alpha_k_img)
            plt.subplot(1,6,4)
            plt.imshow(res_img)
            plt.subplot(1,6,5)
            plt.imshow(result_img)
            plt.subplot(1,6,6)
            plt.imshow(alpha_img)
            plt.show()
