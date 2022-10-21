from core.stock.api import *

d = deposit(0)
a = ai()

while(1):
    querry = input('>>> ')
    if querry == 'exit' or querry == 'quit' or querry == 'q' or querry == 'e' or querry == 'exit()' or querry == 'quit()' or querry == 'q()' or querry == 'e()': 
        break
    if querry == 'show' or querry == 's':
        d.show()
    if querry == 'deposit' or querry == 'd':
        querry = int(input('howmuch> '))
        d.deposit(querry)
    if querry == 'withdraw' or querry == 'w':
        querry = int(input('howmuch> '))
        d.withdraw(querry)
    if querry == 'stock_buy' or querry == 'sb':
        querry = int(input('howmany> '))
        d.stock_buy(querry)
    if querry == 'stock_sell' or querry == 'ss':
        querry = int(input('howmany> '))
        d.stock_sell(querry)
    if querry == 'now_samsung' or querry == 'ns':
        print(f'{d.now_samsung()} KRW')
    if querry == 'tomorrow_samsung' or 'next_samsung' or querry == 'ts' or querry == 'xs':
        print(f'{a.tomorrow_samsung()} KRW')
    
    querry = ''