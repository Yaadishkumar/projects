from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
from datetime import datetime
import time
import imdb_scraper
BRAVE_PATH = r"C:\Users\yaadi\AppData\Local\BraveSoftware\Brave-Browser\Application\brave.exe"

options = Options()
options.binary_location = BRAVE_PATH
options.add_argument("--headless=new")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver.get("https://coinmarketcap.com/")
time.sleep(5)

rows = driver.find_elements("css selector", "table tbody tr")[1:11]

data = []

for row in rows:
    cells = row.find_elements("css selector", "td")
    name = cells[2].text.split("\n")[0]
    price = cells[3].text
    change_24h = cells[5].text
    market_cap = cells[7].text
    data.append([name, price, change_24h, market_cap])

driver.quit()

df = pd.DataFrame(data, columns=["Coin", "Price", "24h Change", "Market Cap"])

print(df.to_string(index=False))

df.to_csv("crypto_prices.csv", mode="a", index=False, header=not pd.io.common.file_exists("crypto_prices.csv"))
print("\nSaved to crypto_prices.csv")