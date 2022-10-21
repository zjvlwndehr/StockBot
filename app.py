from time import sleep
from core.stock.api import *

d = deposit(1)
a = ai()
s = stock()
# a.update_model_samsung_force()
# a.update_model_samsung()
print(a.tomorrow_samsung())
