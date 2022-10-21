import cfscrape

def update_csv():
    r = cfscrape.create_scraper().get("https://query1.finance.yahoo.com/v7/finance/download/005930.KS?period1=1508457600&period2=1666224000&interval=1d&events=history&includeAdjustedClose=true")
    print(r.content.decode('utf-8'))
    with open(f'E:/Development/AI/StockBot/core/model/data/s.csv', 'wb') as f:
        f.write(r.content)