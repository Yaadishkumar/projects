import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

URL = "https://www.imdb.com/chart/top/"


def build_driver(headless=True):
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
    )
    return driver


def scrape_top_250(driver):
    driver.get(URL)
    WebDriverWait(driver, 15).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "li.ipc-metadata-list-summary-item"))
    )
    time.sleep(1)

    movies = []
    rows = driver.find_elements(By.CSS_SELECTOR, "li.ipc-metadata-list-summary-item")

    for row in rows:
        try:
            rank = row.find_element(By.CSS_SELECTOR, "div.ipc-signpost__text").text.strip().lstrip("#")
            title = row.find_element(By.CSS_SELECTOR, "h4.ipc-title__text").text.strip()
            year = row.find_element(By.CSS_SELECTOR, "div.cli-title-metadata li.ipc-inline-list__item").text.strip()
            rating = row.find_element(By.CSS_SELECTOR, "span.ipc-rating-star--rating").text.strip()
            movies.append({"rank": rank, "title": title, "year": year, "rating": rating})
        except NoSuchElementException:
            continue

    return movies


def save_to_csv(movies, filename="imdb_top_250.csv"):
    pd.DataFrame(movies).to_csv(filename, mode="a",index=False, encoding="utf-8")
    print(f"Saved {len(movies)} movies to {filename}")


def main():
    driver = build_driver(headless=True)
    try:
        movies = scrape_top_250(driver)
        if not movies:
            print("No movies scraped.")
            return
        save_to_csv(movies)
    except TimeoutException:
        print("Timed out loading IMDb.")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()