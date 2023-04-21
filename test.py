import math
import re
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords

a = math.e
x = np.arange(0, 10, 0.1)
y = a ** -x
# print(a**-3)

plt.title("函数")
plt.plot(x, y)
plt.show()

a = "Handd"
print(stopwords.words('english'))
print(a.lower())