# from bs4 import BeautifulSoup

import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

driver = webdriver.Chrome('./chromedriver')

# import requests

CAS_list = [
    '121-44-8', 
    '95-56-7',
    '591-20-8',
    '106-41-2',
    '121-446-8',
    '785-225-1145',
]

results = {}

get_url = lambda cas: f"https://www.sigmaaldrich.com/CZ/en/search/{cas}?focus=products&page=1&perpage=60&sort=relevance&term={cas}&type=product"

for i, cas in enumerate(CAS_list):
    url = get_url(cas)
    driver.get(url)

    if i == 0:
        # accept all cookies
        driver.implicitly_wait(10)
        driver.find_element(By.XPATH, "//button[@id='onetrust-accept-btn-handler']").click()

    # find number of results
    driver.implicitly_wait(3)
    count = None

    try:
        element = driver.find_element(By.XPATH, "//div[@data-testid='srp-result-count']")
        split = element.text.split(' ')
        count = int(split[3])  # take 4th word and parse it to int
        results[cas] = dict(available=True, count=count)
        print(f"Number of found products: {count}")
    except NoSuchElementException as e:
        print('CAS was not found.')
        results[cas] = dict(available=False, count=0)

    time.sleep(1)


print(results)

time.sleep(50)

# url = "https://dataquestio.github.io/web-scraping-pages/simple.html"

# page = requests.get(url, timeout=10)

# print(page.content)
# # soup = BeautifulSoup(page.content, "html.parser")


