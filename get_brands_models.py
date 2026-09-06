"""
Step 1: Dynamically fetch all car brands (Makes) from PakWheels used-cars search page.
Goal: Confirm that no brand is missing before starting the full scrape.

How this works:
- The sidebar's 'Make' filter (id="collapse_5") shows only the top 5 brands by default,
  displayed as <a> links (brand name + PakWheels URL with numeric ID).
- A "more choices..." link opens a "Select Make" modal that loads the FULL brand list
  via AJAX. This modal uses checkboxes with labels (NOT <a> links), so it needs a
  different extraction method.
- We merge both sources into one final brand dictionary.

This script only collects the brand list (saved as JSON). No car-detail scraping yet.
"""

import time
import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

CHROMEDRIVER_PATH = r"D:/PycharmProjects/SummerProject/Car-Price-Prediction/chromedriver-3.exe"
USED_CARS_URL = "https://www.pakwheels.com/used-cars/search/-/"


def setup_driver():
    options = Options()
    options.add_argument("--start-maximized")
    # Headless is OFF for now, so we can manually verify the page is loading correctly
    # options.add_argument("--headless=new")
    service = Service(CHROMEDRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=options)
    return driver


def extract_brand_links_from_sidebar(container):
    """
    The default sidebar 'Make' section uses <a> links: brand name + PakWheels URL
    (which contains the brand's numeric ID, e.g. /used-cars/toyota/33).
    """
    brands = {}
    links = container.find_elements(By.TAG_NAME, "a")
    for link in links:
        href = link.get_attribute("href")
        name = link.text.strip()
        if href and name and "/used-cars/" in href:
            brands[name] = href
    return brands


def extract_brands_from_checkbox_modal(driver, modal):
    """
    The 'Select Make' modal uses checkboxes with labels (not <a> links).
    Each row looks roughly like: [checkbox] Brand Name [count].
    We read the checkbox's value attribute (brand slug) and the label text
    (brand display name) using the checkbox's closest label/li/div.
    """
    brands = {}
    checkboxes = modal.find_elements(By.CSS_SELECTOR, "input[type='checkbox']")
    for cb in checkboxes:
        try:
            row_text = driver.execute_script(
                "return arguments[0].closest('label, li, div').innerText;", cb
            )
        except Exception:
            row_text = None

        if not row_text:
            continue

        lines = [line.strip() for line in row_text.split("\n") if line.strip()]
        if not lines:
            continue

        name = lines[0]
        value = cb.get_attribute("value") or name
        brands[name] = value

    return brands


def get_all_brands(driver):
    """
    Extracts all brands from the 'Make' filter on PakWheels' used-cars search page,
    including the ones hidden behind the "more choices..." modal.
    """
    driver.get(USED_CARS_URL)
    time.sleep(3)  # let the page fully load

    brands = {}

    # Step A: grab the default top 5 brands visible in the sidebar
    try:
        make_section = driver.find_element(By.ID, "collapse_5")
        sidebar_brands = extract_brand_links_from_sidebar(make_section)
        brands.update(sidebar_brands)
        print(f"Brands found in default sidebar: {len(sidebar_brands)}")
    except Exception as e:
        print("Could not read default sidebar brand list:", e)

    # Dismiss the "Safety Precautions" popup if it appears, since it can block clicks
    try:
        safety_modal = driver.find_element(By.ID, "safety_precautions")
        if safety_modal.is_displayed():
            close_btn = safety_modal.find_element(By.CSS_SELECTOR, "button.close")
            driver.execute_script("arguments[0].click();", close_btn)
            time.sleep(1)
    except Exception:
        pass  # modal not present, nothing to close

    # Step B: click "more choices..." to load the FULL brand list via AJAX
    try:
        more_choices_link = driver.find_element(
            By.CSS_SELECTOR, "#collapse_5 .more-choice"
        )
        driver.execute_script("arguments[0].scrollIntoView(true);", more_choices_link)
        time.sleep(0.5)
        driver.execute_script("arguments[0].click();", more_choices_link)

        # Give the AJAX call time to populate the modal
        time.sleep(3)

        modal = driver.find_element(By.ID, "lb_more_choices_manufacturers")
        checkbox_count = len(modal.find_elements(By.CSS_SELECTOR, "input[type='checkbox']"))
        print(f"Checkboxes found inside modal: {checkbox_count}")

        if checkbox_count == 0:
            driver.save_screenshot("debug_modal_screenshot.png")
            print("Modal did not load any checkboxes. Saved debug_modal_screenshot.png for review.")
        else:
            full_brands = extract_brands_from_checkbox_modal(driver, modal)
            brands.update(full_brands)
            print(f"Total brands found after loading full list: {len(brands)}")

    except Exception as e:
        driver.save_screenshot("debug_modal_screenshot.png")
        print("Could not load full brand list from modal:", repr(e))
        print("Saved debug_modal_screenshot.png for review.")
        print("Proceeding with only the default sidebar brands found above.")

    return brands


def main():
    driver = setup_driver()
    try:
        brands = get_all_brands(driver)
        print(f"\nTotal brands found: {len(brands)}")
        for name, value in list(brands.items())[:20]:
            print(f" - {name}: {value}")

        # Save so we can review the full list
        with open("brands_found.json", "w", encoding="utf-8") as f:
            json.dump(brands, f, indent=2, ensure_ascii=False)

        print("\nSaved to brands_found.json. Please check this file.")

    finally:
        driver.quit()


if __name__ == "__main__":
    main()