from bs4 import BeautifulSoup
import requests
from selenium import webdriver
import pyautogui, time, os

requests.packages.urllib3.disable_warnings()



def scrape():
    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36"}
    session = requests.Session()
    url = 'https://services.montreal.ca/deneigement/avancement?fbclid=IwAR0hlSzBEyZ_wz55_j-tbEbHEfOmDSIRIj6s0OKW-Uz0Rg3FLnKuq6dWarM'
    session.headers.update(headers)

    r = session.get(url)

    soup = BeautifulSoup(r.content, "html.parser")
    posts = soup.find_all('div', {'class': 'evolution-arrondissement'})

    l = {}
    array = []
    for post in posts:
        for position in post.find_all('div', {'class': 'row data-arrond'}):
            for title, state in zip(position.find_all('div', {'class': 'name-arrond'}),
                                    position.find_all('p', {'class': 'aucune-operation'})):
                word = title.text
                word = word[:-2]  # to delete /t/t from the last of santence
                word1 = word.replace('─', ' ')
                word2 = word1.replace('-', ' ')
                if word2 == 'Côte des Neiges Notre Dame de Grâce                         ':
                    word2 = 'Côte des Neiges Notre Dame de Grâce'.replace('                         ','') # delete space
                word3 = word2.lower().replace('é','e')
                l[word3] = state.text
                array.append(word3)
                array.append(state.text)

    return l, array

    # my position


def display_ip():
    """  Function To Print GeoIP Latitude & Longitude """
    mylist = []
    list, mylist = scrape()
    ticket = 137

    # selenium declaration
    #session = os.path.abspath(__file__).lower().split('users')[1]
    #session = session.split("\ ".replace(' ', ''))[1].split("\ ".replace(' ', ''))[0]
    chromePath = r'C:\Users\nijao\Documents\chromedriver.exe'
    chromePath = chromePath.replace(' ', '')
    driver = webdriver.Chrome(chromePath)
    #test = '45.486112, -73.625248'
    URL = "https://www.google.ca/maps"
    #URL = "https://www.google.ca/maps/search/"+test 
    driver.get(URL)
    driver.maximize_window()
    screenWidth, screenHeight = pyautogui.size()  # Get the size of the primary monitor.
    pyautogui.moveTo(screenWidth / 2, screenHeight / 2)  # Move the mouse to XY coordinates.
    time.sleep(1)
    driver.find_element_by_id('widget-mylocation').click()
    time.sleep(3)
    pyautogui.click()  # Click the mouse.
    # driver.execute_script("window.scrollTo(0, 1000)")
    driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
    time.sleep(1)
    pyautogui.click()  # Click the mouse.
    for i in range(2):
        driver.find_element_by_id('widget-zoom-out').click()  # right-click the mouse
    time.sleep(2)
    pyautogui.moveTo((screenWidth / 2)-10, (screenHeight / 2)+10)  # Move the mouse to XY coordinates.
    pyautogui.click()  # Click the mouse.
    time.sleep(1)
    try:
        element = driver.find_element_by_css_selector(
            '#reveal-card > div > div.widget-reveal-card-container > button.widget-reveal-card-address > div.widget-reveal-card-address-line.widget-reveal-card-bold').text
    except:
        time.sleep(3)
        element = driver.find_element_by_css_selector(
            '#reveal-card > div > div.widget-reveal-card-container > button.widget-reveal-card-address > div.widget-reveal-card-address-line.widget-reveal-card-bold').text     
    element = element.lower().replace('é','e').replace('--', ' ').replace('-', ' ').replace('—', ' ')
    if element in mylist:
        if list.get(element) =='Aucune opération en cours' :
            print("You are good to go for 12 hours")
        else:
            print(" snow removal operation in the next 12hours!\n"
                  "Potential ticket : {} ".format(ticket))
    else:
        print("we don't have enought informatons")

    driver.close()


if __name__ == '__main__':
    
    display_ip()
