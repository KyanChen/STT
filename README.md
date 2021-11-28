# STTNet
Building Extraction from Remote Sensing Images with Sparse Token Transformers
1. Prepare Data     
Prepare data for training, validation, and test phase. All images are with the resolution of <img src="https://latex.codecogs.com/png.image?\dpi{110}&space;512\times&space;512" title="512\times 512" />. Please refer to the directory of **Data**.
2. Get Data List    
Please refer to **Tools/GetTrainValTestCSV.py** to get the train, val, and test csv files.
3. Get Imgs Infos     
Please refer to **Tools/GetImgMeanStd.py** to get the mean value and 
standard deviation of the all image pixels in training set.
4. Modify Model Infos    
Please modify the model information if you want.
5. Run to Train    
Train the model in **Main.py**.


If you have any questions, please refer to [our paper](https://www.mdpi.com/2072-4292/13/21/4441) or contact with us by email.

```
@Article{rs13214441,
AUTHOR = {Chen, Keyan and Zou, Zhengxia and Shi, Zhenwei},
TITLE = {Building Extraction from Remote Sensing Images with Sparse Token Transformers},
JOURNAL = {Remote Sensing},
VOLUME = {13},
YEAR = {2021},
NUMBER = {21},
ARTICLE-NUMBER = {4441},
URL = {https://www.mdpi.com/2072-4292/13/21/4441},
ISSN = {2072-4292},
DOI = {10.3390/rs13214441}
}
```
