# # 下载说明
# # 数据集的下载源为dropbox链接（需要外网访问权限）
# CelebA images
URL=https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=0
ZIP_FILE=./datasets/celeba.zip
mkdir -p ./datasets/
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d ./datasets/
rm $ZIP_FILE

# CelebA attribute labels
URL=https://www.dropbox.com/s/auexdy98c6g7y25/list_attr_celeba.zip?dl=0
ZIP_FILE=./datasets/list_attr_celeba.zip
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d ./datasets/
rm $ZIP_FILE
