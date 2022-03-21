#
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#from __future__ import absolute_import, division, print_function, unicode_literals
import my_mnist  
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class Dropout(layers.Layer):
    def __init__(self,in_shape,dropout_rate=0.3):
        super(Dropout,self).__init__()
        self.in_shape = in_shape
        self.out_shape = self.in_shape
        self.dropout_rate = dropout_rate
    @tf.function
    def call(self,x,training=True):
        if training == True:
            y = tf.nn.dropout(x,self.dropout_rate)
            return y
        else:
            y = x
            return y
class LeakyReLU(layers.Layer):
    def __init__(self,in_shape):
        super(LeakyReLU,self).__init__()
        self.in_shape = in_shape
        self.out_shape = self.in_shape
    @tf.function
    def call(self,x,training=True):
        if training == True:
            y = tf.nn.leaky_relu(x)
            return y
        else:
            y = tf.nn.leaky_relu(x)
            return y
class Dense(layers.Layer):
    def __init__(self, input_dim, units,use_bias=True):
        super(Dense,self).__init__()
        initializer = tf.initializers.glorot_uniform()
        # initializer = tf.initializers.glorot_normal()
        self.w = tf.Variable(initializer((input_dim,units)),dtype=tf.float32,trainable=True)
        self.b = tf.Variable(tf.zeros((1,units)),dtype=tf.float32,trainable=use_bias)#鑺傜偣鐨勫亸缃篃鏄鍚戦噺 鎵嶅彲浠ユ甯歌绠 鍗冲鍫嗗彔鐨刡atch 閮芥槸鍔犺浇鍗曚釜batch鍐
        self.out_dim = units
    @tf.function
    def call(self,x,training=True):
        if training == True:
            y = tf.matmul(x,self.w)+self.b
            return y
        else:
            y = tf.matmul(x,self.w)+self.b
            return y 
class BatchNormalization(layers.Layer):
    def __init__(self,in_shape):
        super(BatchNormalization,self).__init__()
        """
        in_shape 涓嶈冭檻batch缁村害
        """
        self.beta = tf.Variable(tf.zeros((1)),trainable=True)
        self.gamma = tf.Variable(tf.ones((1)),trainable=True)
        self.global_u = tf.Variable(tf.zeros(in_shape),dtype=tf.float32,trainable=False)#鍙備笌娴嬭瘯涓嶅弬涓庤缁
        self.global_sigma2=tf.Variable(tf.ones(in_shape),dtype=tf.float32,trainable=False)#鍙備笌娴嬭瘯涓嶅弬涓庤缁
        self.in_shape = in_shape
        self.out_shape = self.in_shape

    @tf.function
    def call(self,x,training=True):
        """
        x = tf.constant([[1., 1.], [2., 2.]]) [y,x]
        tf.reduce_mean(x)  # 1.5
        tf.reduce_mean(x, 0)  # [1.5, 1.5]  鍥哄畾姣忎釜x 瀵箉缁存墍鏈夊厓绱犳眰鍧囧
        tf.reduce_mean(x, 1)  # [1.,  2.]  鍥哄畾姣忎釜y 瀵箈涓烘墍鏈夊厓绱犳眰鍧囧
        tf.reduce_mean(x,axis) axis=0 鍙傛暟 琛ㄧず 瀵逛簬鍥哄畾鐨勯潪0缁  鍗充竴涓‘瀹氱殑1 2 3 銆傘傘傜淮搴 瀵瑰簲鐨勬墍鏈0缁村厓绱犳眰鍧囧 鐒跺悗鍒犳帀璇ョ淮搴 璇ョ淮搴︾淮鏁版槸1 鍒犳帀鏁翠綋闄嶇淮浣嗘槸涓嶅奖鍝嶅厓绱犱釜鏁 濡傛灉鍒犻櫎鍚庡嚭鐜板垪鍚戦噺 鍒欒浆涓鸿鍚戦噺 
        """
        #imput 鏄痻 shape=[batch_size,x,x,x]
        if training== True:
            u = tf.reduce_mean(x,axis=0)
            sigma2 = tf.reduce_mean(tf.math.square(x-u),axis=0)

            self.global_u.assign(self.global_u*0.99+0.01*u)
            self.global_sigma2.assign(self.global_sigma2*0.99+0.01*sigma2)

            x_hat = (x-u)/tf.math.sqrt(sigma2+0.001) #璁＄畻鍑哄潎鍊煎悗褰掍竴鍖 纭疄搴斿綋淇濊瘉杈撳叆缁村害涓嶅彉
            y = self.gamma*x_hat+self.beta
            return y
        else:#training== False
            x_hat = (x-self.global_u)/tf.math.sqrt(self.global_sigma2+0.001) #璁＄畻鍑哄潎鍊煎悗褰掍竴鍖 纭疄搴斿綋淇濊瘉杈撳叆缁村害涓嶅彉
            y = self.gamma*x_hat+self.beta
            return y

class Conv2DTranspose(layers.Layer):
    def __init__(self,in_shape,out_depth,kernel_size,strides=[1,1],pandding_way="SAME",use_bias=False):
        super(Conv2DTranspose,self).__init__()
        if len(strides)<=2:
            self.strides = [1]+strides+[1] #婊¤冻[1,h,w,1]鐨勫舰寮  浣嗘槸鎸夌収瀹樼綉鐨勬剰鎬 搴旇鏄0xx0 鎵嶅 涓嶇煡閬撳叿浣撴庝箞鏍 浣嗘槸鐭ラ亾鐨勬槸 杩欓噷姝ｅ父鏄墠鍚庤ˉ1 涓嶆敼鍙樺弬鏁伴噺鐨勫崌缁
        else:
            pass
        self.pandding_way = pandding_way
        self.out_depth = out_depth
        initializer = tf.initializers.glorot_uniform()
        #鍙嶅嵎绉殑鍏ㄤ妇璇佷互[height, width, output_channels, in_channels]鏂瑰紡瀹氫箟 渚濇嵁浜唗f.nn.conv2d_transpose鐨勮姹
        w_shape = kernel_size+[self.out_depth]+[in_shape[-1]]
        self.w = tf.Variable(initializer(w_shape),dtype=tf.float32,trainable=True)
        P = [None,None]
        if self.pandding_way =="SAME":
            P[0]= kernel_size[0]//2
            P[1]= kernel_size[1]//2
            out_shape= [self.strides[1]*in_shape[0],self.strides[2]*in_shape[1]]
        elif self.pandding_way == "VALID":
            P = [0,0]
            out_shape= [self.strides[1]*in_shape[0]+max(kernel_size[0]- self.strides[1],0),self.strides[2]*in_shape[1]+max(kernel_size[1]-self.strides[2],0)]
            # 璇ュ叕寮忕収鎶勪簡婧愮爜 鍘熺悊涔熸槸渚濇嵁涓嬭堪鎺ㄥ 瀵瑰悜涓嬪彇鏁翠骇鐢熷鍊兼椂鍙栦簡涓涓悎鐞嗗畾鍒 stride=2鏃朵笉鍙伩鍏嶅弽鍗风Н鍑虹幇缁村害姝т箟
            # 鍙鍙嶅嵎绉殑瀹氫箟涓嶅簲璇ヤ粠鍗风Н鍙嶈繃鏉ョ湅 搴斿綋鏈変釜鏇村ソ鐨勫畾涔夋嫆缁濇涔
        else:
            raise ValueError
        self.b_shape = out_shape+[self.out_depth]
        self.b = tf.Variable(tf.zeros(self.b_shape),dtype=tf.float32,trainable=use_bias)
        self.in_shape = in_shape
        self.out_shape = self.b_shape
        """
        闈炴鏂瑰舰鍗风Н鏍 鐗瑰緛鍥 涔熸槸閫傜敤鐨 涓轰簡鏂逛究 鍋囧畾鏄鏂瑰舰 鍙槓杩版í鍚戞柟鍚
        鍗风Н [B,H1,W1,C1] 鍒 [B,H2=?,W2=?,C2]鐨勫彉鍖 缁欏畾鍗风Н鏍哥殑澶у皬Wk 
        涓轰簡纭畾鍗风Н鏍稿弬鏁颁釜鏁 闇瑕佺煡閬 C2 Hk Wk C1 鍙傛暟涓暟=C2*Hk*Wk*C1
        涓轰簡纭畾鍋忕疆鍙傛暟涓暟 闇瑕佺煡閬 W2 鍙傛暟涓暟=H2*W2*C2 鏈夊叕寮忓涓
        W2 = [(W1-Wk+2*P)/S]+1 鍗风Н鐨勫叕寮忔槸濮嬬粓鎴愮珛鐨 []琛ㄧず鍚戜笅鍙栨暣

        鍙嶅嵎绉 鍦ㄨ〃闈笂 鏄皢涓婅堪杩囩▼鍊掕繃鏉 浣跨敤鐩稿悓澶у皬鐨勫嵎绉牳(鍗风Н鐭╅樀鐩稿悓 浣嗘槸鍏剁粍鏁板拰灞傛暟涓嶅悓 渚濇嵁鍓嶅悗閫氶亾鏁扮殑鍙樺寲鑰屾潵)鍜屾闀縮 瀹炵幇浠嶽B,H2,W2,C2] 鍒 [B,H1',W1',C1']鐨勫彉鎹 鍘熷垯鏄娇鐢ㄧ浉鍚屽ぇ灏忕殑鍗风Н鏍(鍙傛暟鍐呭蹇呯劧鏄敼鍙樼殑,鍗风Н鏍哥殑缁勬暟 鍜 鍗风Н鐭╅樀涓暟蹇呯劧鏄浉鍙嶇殑 浠栦滑瀵瑰簲鍓嶅悗涓嶅悓鐨勯氶亾)鍜屾闀縮
        鍘熷垯鏄娇鐢ㄧ浉鍚屽ぇ灏忕殑鍗风Н鏍(鍙傛暟鍐呭蹇呯劧鏄敼鍙樼殑)鍜屾闀縎锛侊紒锛 鎵浠ユ墠鍙弽鍗风Н
        鍘熷垯鏄娇鐢ㄧ浉鍚屽ぇ灏忕殑鍗风Н鏍(鍙傛暟鍐呭蹇呯劧鏄敼鍙樼殑)鍜屾闀縎锛侊紒锛 鎵浠ラ渶瑕佺粰瀹氬嵎绉牳
        鍘熷垯鏄娇鐢ㄧ浉鍚屽ぇ灏忕殑鍗风Н鏍(鍙傛暟鍐呭蹇呯劧鏄敼鍙樼殑)鍜屾闀縎锛侊紒锛 鎵浠ラ噰鐢ㄧ浉鍚屽ぇ灏忕殑鍙﹀鐨勫嵎绉牳 鍜屾闀縎 鏄彲浠ヤ粠[H1',W1']鍙樻崲鍒
        瀹為檯涓  鏄[B,H2,W2,C2]渚濇嵁缁欏畾鐨 鍙嶅嵎绉牳  鍙嶅嵎绉痯adding 杩涜姝ｅ父鐨勫嵎绉搷浣 鍙槸padding鐨勬柟寮忔湁鎵鍙樺寲 鏈夊彲鑳芥媺鏁ｅ師鏁版嵁 涓棿琛ラ浂(渚濇嵁padding鏂瑰紡) 瀹炵幇鍒癧B,H1',W1',C1']鐨勫彉鎹 涓鑸儏鍐甸兘鏄皢鍗曚釜閫氶亾鐨勫浘鍍忔斁澶т簡

        鍗冲鏋滄垜浠煡閬撲簡W2 Wk padding strides 鍙互鎺ㄧ畻鍑篧1 鍙鎵惧埌閫傚悎涓婅堪鐨刉1鍗冲彲 
        闂鍦ㄤ簬 瀵逛簬鍚戜笅鍙栨暣[] W1鐨勫彇鍊煎彲鑳芥槸涓嶅敮涓鐨 鎵浠ュ瓨鍦ㄥ瑙 
        
        閭 鑻ユ槸鎴戜滑鍐嶇粰瀹氫簡鍘熻緭鍏ョ淮搴1(鍗虫湡鏈涘弽鍗风Н杈撳嚭缁村害) 鍙涓嶄簣涓婇潰鐨勫瑙ｅ啿绐 鍗虫槸鍙鐨 

        鎸夌収tf.keras.layers.Conv2DTranspose 鐨勮緭鍏ヨ姹 Conv2DTranspose铏界劧鏄庨潰涓婁笉闇瑕佺煡閬撹緭鍏ョ淮搴 浣嗘槸鏄嚜鍔ㄥ叧鑱斾笂涓灞傜殑杈撳嚭浜  绗竴灞傛椂闇瑕佹寚瀹 鏁呰繕鏄渶瑕佺煡閬(H2,W2) C2
        宸茬煡 (H2,W2) C2 C1' [Hk,Wk] S=[S_h,S_w] 鍜宲adding鏂瑰紡 
        渚濇嵁    padding="SAME" P_w=Wk//2,P_h=Hk//2   
                padding="VALID" P_w=P_h=0
        绾︽潫涓
                W2 = [(W1'-Wk+2*P_w)/S_w]+1
                H2 = [(H1'-Hk+2*P_h)/S_h]+1
        鍗风Н鏍稿弬鏁颁釜鏁颁负 C1'*Hk*Wk*C2
        鍋忕疆鍙傛暟涓暟涓   H1'*W1'*C1'

        tensorflow 鏈夋剰鎬濈殑鏄 tf.nn.conv2d_transpose瑕佹眰鐨勮緭鍏ヤ腑 鏈塱nput 鍗风Н鏍 output_shape,strides,padding='SAME' 鍗
        鍙互鐭ラ亾 (H2,W2) C2  Hk Wk     H1' W1' C1' S=[S_h,S_w] padding  閭ｅ彧闇瑕侀獙璇佺瓑寮忔槸鍚︽垚绔嬪氨濂戒簡

        澶ц儐鐚滄兂 tf.keras.layers.Conv2DTranspose瀵逛笂杩扮殑绾︽潫閲囧彇浜嗘瀬绔舰寮 鐩存帴缁欏畾浜嗗洜涓哄彇鍊艰屽鍊肩殑H1' W1'
        鑰宼f.nn.conv2d_transpose鍒欒姹傜敤鎴风粰瀹欻1' W1' 闃叉鏈夋涔

        鎴戝熀浜巘f.nn.conv2d_transpose鏋勫缓鑷繁鐨刲ayers.Conv2DTranspose鏃惰繕鏄渶瑕佽绠楀苟鏁插畾W1' H1'鐨 涓嶇劧灏辨棤娉曚娇鐢ㄥ亸缃簡 涔熸棤娉曞畬鎴愬墠鍚戣繃绋
        鎸夌収tf.keras.layers.Conv2DTranspose婧愮爜涓 瀵逛簬"SAME"鐨凱adding鏂瑰紡 鍙嶅嵎绉洿鎺ラ噰鐢╓1'=W2*S_w 鐨勬柟寮忔眰鍊 鎴戣繖閲屼篃灏变笉鍐嶅幓鍋氬鍊肩殑璁＄畻浜  
        """
    @tf.function
    def call(self,x):
        convtranspose_out = tf.nn.conv2d_transpose(input=x,filters=self.w,output_shape=[x.shape[0]]+self.b_shape,strides=self.strides, padding=self.pandding_way)
        l_out = convtranspose_out+self.b
        return l_out
class Conv2D(layers.Layer):
    def __init__(self,input_shape,out_depth,filter_size,strides,use_bias=True,pandding_way="SAME"):
        super(Conv2D,self).__init__()
        if len(strides)<=2:
            self.strides = [1]+strides+[1] #婊¤冻[1,h,w,1]鐨勫舰寮
        else:
            pass
        self.pandding_way = pandding_way
        """
        闈炴鏂瑰舰鍗风Н鏍 鐗瑰緛鍥 涔熸槸閫傜敤鐨 涓轰簡鏂逛究 鍋囧畾鏄鏂瑰舰 鍙槓杩版í鍚戞柟鍚
        input_shape 鏄浘鐗囧ぇ灏忓拰閫氶亾鏁
        kernel_size 鏄嵎绉牳澶у皬
        step鏄闀縮tride
        kernel_initializer='glorot_uniform', bias_initializer='zeros'

        瀵逛竴涓猙atch涓殑鐗瑰畾涓涓緭鍏
        杈撳叆鐗瑰緛鍥句负 [W1,H1,D1] 涓変釜鍙傛暟 鍗崇壒寰佸浘瀹絎1 楂楬1 鍜屾繁搴-閫氶亾D1
        
        杈撳嚭鐗瑰緛鍥句负 [W2,H2,D2] 涓変釜鍙傛暟 鍗崇壒寰佸浘瀹絎2 楂楬2 鍜屾繁搴-閫氶亾D2

        鍗风Н鏍稿拰鍋忕疆缁寸郴浜嗚緭鍏ヨ緭鍑虹壒寰佸浘鐨勫叧绯

        鍗风Н鏍告湁 鍥涗釜鍙傛暟 [D2,Wk,Hk,D1]  瀵逛簬杈撳叆鐗瑰緛鍥句腑鐨勪竴涓氶亾 瀵逛簬涓缁勫嵎绉牳涓殑涓涓
        鍗 W1 H1 D1 瀵瑰簲 Wk Hk D1

        杈撳叆鐗瑰緛鍥剧殑涓涓氶亾鍙細鏈夊敮涓鐨勪竴涓嵎绉煩闃靛拰鍏跺嵎绉  
        涓缁勫嵎绉牳涓殑涓涓嵎绉煩闃靛彧涓庡搴旂殑涓涓緭鍏ョ壒寰侀氶亾鍗风Н 

        鎵浠ヤ竴缁勫嵎绉牳涓殑涓涓嵎绉煩闃 鍜  杈撳叆鐗瑰緛涓殑涓涓氶亾 鏄竴涓瀵瑰簲鐨 涓嶄細鍜屽叾浠栫殑鏈夊叧鑱

        鍗风Н鏍哥殑缁勬暟 绫讳技浜巄atch浣嗗嵈涓嶆槸  鐢辫緭鍑虹壒寰佸浘閫氶亾鏁板喅瀹

        鍗冲灏戠粍鍗风Н鏍 灏辨湁澶氬皯涓緭鍑虹壒寰侀氶亾 

        鎵浠  瀵逛簬涓涓嵎绉搷浣 鏄皢姣忎竴缁勫嵎绉牳涓殑D1涓嵎绉煩闃礫Wk,Hk] 鍜岃緭鍏ョ壒寰佸浘鐨凞1涓氶亾[W1,H1] 涓瀵逛竴 鍒嗗埆鍗风Н 涓嶄簰鐩稿共鎵 寰楀埌D1鏁伴噺鐨勫嵎绉緭鍑篬W2,H2] 灏嗗畠浠浉鍔 
        寰楀埌涓涓猍W2,H2]鐭╅樀 鏋勬垚涓嶅姞鍋忕疆鐨 杈撳嚭鐗瑰緛鍥剧殑涓涓氶亾 鍔犱笂 涓涓猍W2,H2] 鐨勫亸缃煩闃 寰楀埌甯﹀亸缃殑杈撳嚭鐗瑰緛鍥剧殑涓涓氶亾

        瀵笵2缁勫嵎绉牳鍋氬悓鏍风殑鎿嶄綔  灏卞緱鍒颁簡 D2涓 [W2,h2]鐨勮緭鍑虹壒寰佸浘 D2灏辨槸杈撳嚭鐗瑰緛鍥鹃氶亾鏁

        W1 H1 D1 D2 閮芥槸鎸囧畾鐨 涓嬮潰闇瑕佽绠梂2 H2鐨勫叧绯 瀵逛簬padding 鐞嗚В鎴愬厛鏀瑰彉W1 H1 鍐嶈繘琛屼笂杩扮殑鎿嶄綔鍗冲彲
        杈呬箣 padding鐨勫ぇ灏廝  鍗风Н姝ラ暱S 鎸囧畾"SAME"鏃 padding 鍦ㄨ緭鍏ョ壒寰佸浘鐨勬瘡涓氶亾鐨勫懆鍥磋ˉ澶氬眰0鍊 淇濊瘉濡傛灉鏄闀夸负1 鍗风Н鍓嶅悗鐗瑰緛鍥剧殑澶у皬涓嶅彉 閫氶亾鍙兘鏀瑰彉 P=Wk/2
                                    鎸囧畾"VALID"鏃 涓峱adding P=0
        鍗风Н鍓嶅悗 鐗瑰緛鍥剧殑瀹界殑鍙樺寲瑙勫緥濡備笅 
        W2 = [(W1-Wk+2*P)/S]+1 杩欎釜鍏紡鐨勭悊瑙ｆ槸  鏈宸︿笂瑙掍竴瀹氭槸鍗风Н绗竴娆¤繍绠 鐒跺悗鍓╀綑閮ㄥ垎鍙互璁╁嵎绉牳绉诲姩鍑犳鍛 鐒跺悗鐩稿姞鍗冲彲 [.]鍚戜笅鍙栨暣 鍥犱负褰撲笉鑳芥暣闄ゆ椂 閴翠簬绗竴涓綅缃殑瀛樺湪 灏变笉澶熻窛绂绘斁涓嬩笅涓涓嵎绉牳浜
        W2 = {(W1-Wk+2*P+1)/S} 鍜屼笂闈㈢殑鍏紡绛変环 鐞嗚В涓 璁＄畻鍑哄嵎绉牳鐨勬渶涓婅鎴栬呮渶宸﹀垪鍙互鍑虹幇鐨勪綅缃釜鏁 鍦ㄦ闀縮鐨勫尯闂翠笂鍒嗛厤 鏁撮櫎鍒欏垰濂 涓嶆暣闄ゅ垯闇瑕佽繘涓娉曡ˉ鍏 鍥犱负纭疄鍙互姣廠涓尯闂存斁涓嬩竴涓 涓旂瓑闂磋窛
        //鍙屾枩鏉犳墠鏄暣闄ゅ彇鏁村悜涓

        宸茬煡 (H1,W1) C1  C2 [Hk,Wk] S=[S_h,S_w] 鍜宲adding鏂瑰紡 
        渚濇嵁    padding="SAME" P_w=Wk//2,P_h=Hk//2   
                padding="VALID" P_w=P_h=0
        绾︽潫涓 
                W2 = [(W1-Wk+2*P_w)/S_w]+1
                H2 = [(H1-Hk+2*P_h)/S_h]+1
        鍗风Н鏍稿弬鏁颁釜鏁颁负 C2*Hk*Wk*C1
        鍋忕疆鍙傛暟涓暟涓   H2*W2*C2
        """
        P = [None,None]
        if self.pandding_way == "SAME":
            P[0] = filter_size[0]//2
            P[1] = filter_size[1]//2
        elif self.pandding_way=="VALID":
            P = [0,0]
        else:
            raise ValueError
        out_shape = [(input_shape[i]+2*P[i]-filter_size[i])//self.strides[i+1]+1 for i in range(2)]

        initializer = tf.initializers.glorot_uniform()
        w1_shape = filter_size+[input_shape[-1]]+[out_depth]
        self.w = tf.Variable(initializer(w1_shape),dtype=tf.float32,trainable=True)

        b_shape = out_shape+[out_depth]
        self.b = tf.Variable(tf.zeros(b_shape),dtype=tf.float32,trainable=use_bias)
        self.in_shape = input_shape
        self.out_shape = b_shape
    @tf.function
    def call(self,x):
        conv_out = tf.nn.conv2d(input=x,filters=self.w,strides=self.strides,padding=self.pandding_way,data_format='NHWC',dilations=None,name=None) #dilations鏄┖娲炲嵎绉殑涓涓郴鏁 鐩稿綋浜庡鍗风Н鏍稿仛涓婇噰鏍峰悓鏃堕儴鍒嗙疆闆  杩欓噷涓嶈繘琛岀┖娲炲嵎绉
        l_out = conv_out+self.b
        return l_out

if __name__ == "__main__":
    (train_images,train_labels),(_, _) = my_mnist.load_data(get_new=False,
                                                        normalization=False,
                                                        one_hot=True,
                                                        detype=np.float32)
    train_images = (train_images.astype('float32')-127.5)/127.5
    train_labels = (train_labels.astype('float32')-0.5)/0.5                                                    
    train_images = train_images.reshape(train_images.shape[0], 28, 28,1)
    print(train_labels[0])
    plt.imshow(train_images[0, :, :,0], cmap='gray')
    plt.show()

    x = tf.random.normal(shape=(64,784))
    a = Dense(28*28,128)
    print(a(x))
    print(len(a.trainable_variables))
    y = train_images[0:1, :, :,0:1]
    C1 = Conv2D([28,28,1],2,[5,5],strides=[2,2],use_bias=True,pandding_way="SAME")
    print(len(C1.trainable_variables)) #鍗风Н鎿嶄綔鐨勮緭鍏ュ繀椤绘弧瓒砃(B) H W C 
    fielt_out = C1(y)
    plt.imshow(fielt_out[0, :, :,0], cmap='gray')
    plt.show()
    plt.imshow(fielt_out[0, :, :,1], cmap='gray')
    plt.show()
    C2 = Conv2D([28,28,1],2,[5,5],strides=[2,2],use_bias=False,pandding_way="SAME")
    print(len(C2.trainable_variables))
    C3 = Conv2DTranspose([28,28,1],2,[5,5],strides=[1,1],pandding_way="SAME",use_bias=False)
    print(len(C3.trainable_variables))
    filter_out = C3(y)
    plt.imshow(filter_out[0, :, :,0], cmap='gray')
    plt.show()
    plt.imshow(filter_out[0, :, :,1], cmap='gray')
    plt.show()
    C4 = Conv2DTranspose([28,28,1],2,[5,5],strides=[1,1],pandding_way="SAME",use_bias=True)
    print(len(C4.trainable_variables))

    B1 = BatchNormalization(in_shape=[28,28,1])
    x = train_images[0:2, :, :,:]
    print(B1(x,training=True))
    print(B1(x,training=False))
    print(B1.trainable_variables)
    print(len(B1.trainable_variables))