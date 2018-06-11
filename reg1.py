from matplotlib import font_manager, rc
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

font_name=font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc("font", family=font_name)

x=[3.52, 2.58, 3.31, 4.07, 4.62, 3.98, 4.29, 4.83, 3.71, 4.61, 3.90, 3.20]
y=[2.48, 2.27, 2.47, 2.77, 2.98, 3.05, 3.18, 3.46, 3.03, 3.25, 2.67, 2.53]
result=stats.linregress(x,y)
print(result)

slope, intercept, r_value, p_value, stderr=stats.linregress(x,y)
x1=np.array(x)
plt.scatter(x,y)
plt.plot(x1, slope*x1+intercept, c="red")
plt.xlabel("전기생산량")
plt.ylabel("전기사용량")
plt.show()