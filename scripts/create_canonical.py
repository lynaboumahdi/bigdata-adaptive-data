import pandas as pd
import numpy as np


images=np.load("data/raw/x_train.npy")
labels=np.load("data/raw/y_train.npy")


images=images.reshape(len(images),784)


columns=[f"pixel_{i}" for i in range(784)]


df=pd.DataFrame(images,columns=columns)


df["label"]=labels


df.to_csv(
"data/canonical/fashion_mnist.csv",
index=False
)


print(df.head())
print(df.shape)