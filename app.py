from core.stock.api import *

a = ai()
querry = input("id(int or connected_english)> ")
d = deposit(querry)

print('[INFO]: Hello! If you need help, type help or h')

while(1):
    querry = ''
    querry = input('>>> ')
    if querry == 'exit' or querry == 'quit' or querry == 'q' or querry == 'e' or querry == 'exit()' or querry == 'quit()' or querry == 'q()' or querry == 'e()': 
        break
    elif querry == 'help' or querry == 'h':
        print('[INFO]: Help!\nshow, s: show your money\ndeposit, d: deposit your money\nwithdraw, w: withdraw your money\nstock_buy, sb: buy stock\nstock_sell, ss: sell stock\nnow_samsung, ns: show now samsung stock price\ntomorrow_samsung, next_samsung, ts: show tomorrow samsung stock price')
    elif querry == 'show' or querry == 's':
        d.show()
    elif querry == 'deposit' or querry == 'd':
        querry = int(input('howmuch> '))
        d.deposit(querry)
    elif querry == 'withdraw' or querry == 'w':
        querry = int(input('howmuch> '))
        d.withdraw(querry)
    elif querry == 'stock_buy' or querry == 'sb':
        querry = int(input('howmany> '))
        d.stock_buy(querry)
    elif querry == 'stock_sell' or querry == 'ss':
        querry = int(input('howmany> '))
        d.stock_sell(querry)
    elif querry == 'now_samsung' or querry == 'ns':
        print(f'{d.now_samsung()} KRW')
    elif querry == 'tomorrow_samsung' or querry == 'ts':
        print(f'{a.tomorrow_samsung()} KRW')
   