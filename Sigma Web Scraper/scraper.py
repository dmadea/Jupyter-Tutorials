#!python3.10

import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

driver = webdriver.Chrome('./chromedriver')

with open('CAS.txt') as file:
    CAS_list = [line.rstrip() for line in file]


# CAS_list_test = [
#     '121-44-8', 
#     '95-56-7',
#     '591-20-8',
#     '106-41-2',
#     '121-446-8',
#     '785-225-1145',
# ]

# results = {}

get_url = lambda cas: f"https://www.sigmaaldrich.com/CZ/en/search/{cas}?focus=products&page=1&perpage=60&sort=relevance&term={cas}&type=product"


with open('results.txt', 'w') as file:

    def write_line(text):
        file.write(text + "\n")

    write_line("CAS\tCount\tLink to 1st product")
    for i, cas in enumerate(CAS_list):
        url = get_url(cas)
        driver.get(url)

        if i == 0:
            # accept all cookies
            driver.implicitly_wait(10)
            driver.find_element(By.XPATH, "//button[@id='onetrust-accept-btn-handler']").click()

        # find number of results
        driver.implicitly_wait(20)
        count = 0

        try:
            element = driver.find_element(By.XPATH, "//div[@data-testid='srp-result-count']")
            split = element.text.split(' ')
            count = int(split[3])  # take 4th word and parse it to int
            # results[cas] = dict(available=True, count=count)
            print(f"Number of found products for {cas}: {count}")
        except NoSuchElementException as e:
            print(f'CAS {cas} was not found.')
            # results[cas] = dict(available=False, count=0)

        driver.implicitly_wait(3)
        href = ""

        try:
            element = driver.find_element(By.XPATH, "//*[@id='__next']/div/div[2]/div[1]/div[1]/div/div[2]/div[4]/div[1]/div[1]/div[2]/ul/li/h3/a")
            href = element.get_attribute('href')
        except NoSuchElementException as e:
            print('No href found.')
        except Exception as e:
            print(e)

        write_line(f"{cas}\t{count}\t{href}")
        file.flush()

        time.sleep(1)

        if i > 0 and i % 10 == 0:
            time.sleep(20)

        if i > 0 and i % 100 == 0:
            time.sleep(300)

# time.sleep(50)


