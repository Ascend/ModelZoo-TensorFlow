## 1、原始模型
[pb模型](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=qGsoRKs7PXjEdQRD69cSXlYD2JkUaqMH6VY0pAxGW4wCieEAZmGnQ28vOC2bzXuozqjrasFcESwg3pPH5r7u2Fbtd9IpnjTN/Mg9iFTh5vvp1UN9Y7NqfbZiP7APWiUSn1h4OCE3RzkFifTM3FfkvzzVB8/XRCUv0OP7BaaoqGgbOF+us+PuhI5ESlXF4nIhADkK5rDNxzmJHgWXt/FT+5fpBnQ02mdB7HXGdPpCSHvvSNnZZD7tHbEE26B+uceT3shl/otoIJO22noT7m1pZP7HYk9KiM21L8murxS2VOEDkUAQlQaeubi5iY6rJRI3UzWI6P7kf3gGZRxjDHNbidFil7h4MYNNuwPSU9Gtqd46zqQMVxMlH1/B51g8BU4RM2UmhwqKOzQW1Ibayp2uDo8HEVvkJtYB8QomSTGS25BElPuCoD2++Ogvf6vnEuZW6YNWphtp1bDjgJez+khszy+7Llj3zdP7zOgismRcqoP+/+H+vUrUfpUM/onXK9cAY+DDzRdnZfdETxZ7TGZTmnjjYJsixsSNDwr+NtrwGPG+TO9BwkvaMpbf6xQLofNc//K3J2ulTxnNifB/zqHvKA==)

文件夹中有pb文件

## 2、转om模型

atc转换命令参考：

```sh
atc --model=/home/HwHiAiUser/AscendProjects/VSL/data.pb --framework=3 --output=/home/HwHiAiUser/AscendProjects/data --soc_version=Ascend310 --input_shape="conv3d_input:1,1,4,5,100" --log=info
```

## 3、编译msame推理工具
参考https://gitee.com/ascend/tools/tree/master/msame, 编译出msame推理工具


## 4、全量数据集精度测试：
根据toBin.py文件通过对原始数据的处理，可以得到om的数据集，原始数据通过以下连接可得

[原始数据](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=O40ZJJpoUK/dRYM9yqQaCQ7L4CD/lbujlI/RLmij6g5XDbpObK7iovT1W9VYZPlOh9lfDH8YU/dg1mG4DLTJa4un1SrKTr0eD7oFIIOqyb30cK/t4dT2FODusNCkP9wHngc3XgpC1UZ8AG9bISuAl+G850XM/Da8m44S2hAY4YYu9BqXBavfHjfKAiq4nmFSy0AKoBXMgvtr+xGk4Ol2gmn1a80JyE7iSlkXK2BKvgh3h/Ck+dCEQ8c1+VVzZ6soN8p1Wj4bck1XChyo5T/GNIDX/RymvGbKyu7sn2Diy62rOTNeaNzAxlPJ3GVj6HfijhtrAAmBMqOJr3LzcrJ3+w/hhUJXBuSnH028gWFdBjmHCM7tnttafw3UkjCOmYHf7AFJFQE6DiXIY+b5zSW/yHzQx6/z/iTjJ5DHHQgEt3HHNReo+7W+sYhUXrMXJ1KYPg9xKsuHhEIKwR5z5zYjENaa34VU70UJkfZ8z3aIdm+jzZlQ9QapMhFPJ2OC+CENMdpm8kjAfuzvLD3peTT9DmB/wjB8Z+oIXBrs2pL6vRg=)

提取码:
123456

*有效期至: 2022/12/15 12:05:07 GMT+08:0

om测试集总共10个数据，每一个数据一个bin，压缩后的文件夹根据以下连接可获得：

[bin数据](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=qGsoRKs7PXjEdQRD69cSXlYD2JkUaqMH6VY0pAxGW4x7tFCGRGPE3Wl2/9mO3JB15IzqCkdp1/GuHAZeXTVAxNlQ4XMJ8pOxzCJs8QjzClcxuZjjZv7G6ga6UrRoJRuL81JaDB804SQ02Qoowipx3Z6Sh8maALJj1M1Sl0xcpGhgkT7i/yZEUQPtEz/BPmDvhudPS4c9t1SEYNrk8+XfJj7XapR5mLwsBrkRYCoqOfjOi6uF7/Uj5cck08ZWsVySduI+ead+lLmlkKU1VLEJISKtvm/B10DeFqg2DGCwqki7wzdK6YF09VA46CxMidZQfP3FTmDQWnd1JO/SjI0FdFuqJKuEQdV3EKYL24ekpnHHk4rV3vPBalHvqU0GCe+1fgrTLJNA11TAOMxVW51WrjEjLBfwHAxX96vMYsSF/gpZz7ltdqecSNHFwXfLYg3bQ30sHxI5clr9SOJDlfDCkIgJyW1NpGuxG6XFUVtfDxs44wvorAJhhaSz3fiEyjeC852CzfRMi7Db/PMhbRS41uWSUM/bKfK+1DQEskOnjQE=)

提取码:
123456

*有效期至: 2022/11/25 16:27:09 GMT+08:00

### 4.1 执行推理和精度计算
使用以下命令进行推理
```
./msame --model "/home/HwHiAiUser/AscendProjects/data.om" --input "/home/HwHiAiUser/AscendProjects/data0.bin" --output "/home/HwHiAiUser/AscendProjects/VSL/om/" --outfmt TXT --loop 1
```
推理得到的结果为一个numpy数组 0 0 0 1，表示data0数据被分为第四类