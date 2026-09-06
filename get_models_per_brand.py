"""
Step 2: For every brand found in Step 1 (brands_found.json), fetch the FULL list
of models available for that brand on PakWheels.

How this works:
- We visit https://www.pakwheels.com/used-cars/search/-/mk_{brand_slug}/ for each brand.
  This filter URL works for any brand slug, without needing the brand's numeric ID.
- On this page, the sidebar has a "Model" filter section showing the top 5 models
  as <a> links, plus a "more choices..." link that opens a "Select Model" modal
  (loaded via AJAX) containing the FULL model list as checkboxes.
- We locate the "Model" section by its heading text (not by a hardcoded collapse_N id),
  since the id numbering can shift depending on which filters a brand has available.

Output: models_by_brand.json -> { "Suzuki": ["Alto", "Cultus", "Mehran", ...], ... }
"""

import time
import json
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

CHROMEDRIVER_PATH = r"D:/PycharmProjects/SummerProject/Car-Price-Prediction/chromedriver-3.exe"
BRANDS_FILE = "brands_found.json"
OUTPUT_FILE = "models_by_brand.json"


def setup_driver():
    options = Options()
    options.add_argument("--start-maximized")
    # Headless is OFF for now, so we can manually verify the page is loading correctly
    # options.add_argument("--headless=new")
    service = Service(CHROMEDRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=options)
    return driver


def slug_from_brand_value(value):
    """
    brands_found.json stores either a full PakWheels URL (e.g. containing '/toyota/33')
    or a plain slug (e.g. 'baic'). This function normalizes both cases to a slug.
    """
    if value.startswith("http"):
        # Extract the part right after '/used-cars/' and before the next '/'
        match = re.search(r"/used-cars/([a-z0-9\-]+)", value)
        if match:
            return match.group(1)
    return value


def dismiss_safety_popup(driver):
    try:
        safety_modal = driver.find_element(By.ID, "safety_precautions")
        if safety_modal.is_displayed():
            close_btn = safety_modal.find_element(By.CSS_SELECTOR, "button.close")
            driver.execute_script("arguments[0].click();", close_btn)
            time.sleep(1)
    except Exception:
        pass  # popup not present, nothing to close


def extract_names_from_checkbox_modal(driver, modal):
    """
    Reads brand/model names from a modal that uses checkboxes with labels
    (e.g. the 'Select Model' or 'Select Make' popups).
    """
    names = set()
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
        if lines:
            names.add(lines[0])

    return names


def extract_names_from_sidebar_links(container):
    """
    Reads model names from the default sidebar list, which uses <a> links.
    """
    names = set()
    links = container.find_elements(By.TAG_NAME, "a")
    for link in links:
        text = link.text.strip()
        if text:
            names.add(text)
    return names


def get_models_for_brand(driver, brand_slug):
    """
    Visits the brand's filtered search page and extracts all model names,
    merging the default sidebar list with the full "more choices" modal list.
    """
    url = f"https://www.pakwheels.com/used-cars/search/-/mk_{brand_slug}/"
    driver.get(url)
    time.sleep(2.5)

    dismiss_safety_popup(driver)

    models = set()

    # Step A: find the "Model" filter section by its heading text (not a hardcoded id)
    try:
        model_heading = driver.find_element(
            By.XPATH,
            "//a[contains(@class,'accordion-toggle') and normalize-space(.)='Model']"
        )
        model_body = model_heading.find_element(
            By.XPATH, "./parent::div/following-sibling::div[contains(@class,'accordion-body')]"
        )
    except Exception as e:
        print(f"  [!] Could not find Model section for '{brand_slug}': {e}")
        return sorted(models)

    # Default top models shown as <a> links
    models.update(extract_names_from_sidebar_links(model_body))

    # Step B: click "more choices..." within the Model section to load the full list
    try:
        more_choices_link = model_body.find_element(By.CSS_SELECTOR, ".more-choice")
        modal_target = more_choices_link.get_attribute("data-target")  # e.g. "#lb_more_choices_models"
        modal_id = modal_target.replace("#", "") if modal_target else "lb_more_choices_models"

        driver.execute_script("arguments[0].scrollIntoView(true);", more_choices_link)
        time.sleep(0.3)
        driver.execute_script("arguments[0].click();", more_choices_link)
        time.sleep(2.5)  # allow the AJAX call to populate the modal

        modal = driver.find_element(By.ID, modal_id)
        checkbox_count = len(modal.find_elements(By.CSS_SELECTOR, "input[type='checkbox']"))

        if checkbox_count > 0:
            models.update(extract_names_from_checkbox_modal(driver, modal))
        else:
            # Modal might use <a> links instead of checkboxes for some brands; try that too
            models.update(extract_names_from_sidebar_links(modal))

    except Exception as e:
        print(f"  [!] No 'more choices' for models on '{brand_slug}' (might only have a few models): {e}")

    return sorted(models)


def main():
    with open(BRANDS_FILE, "r", encoding="utf-8") as f:
        brands = json.load(f)

    # Resume support: if models_by_brand.json already exists from a previous run,
    # load it and skip brands that already have a non-empty model list.
    results = {}
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            results = json.load(f)
        already_done = sum(1 for v in results.values() if v)
        print(f"Found existing {OUTPUT_FILE} with {already_done} brands already fetched. Resuming...")
    except FileNotFoundError:
        pass

    # Manual overrides: brands with so few PakWheels listings that the site doesn't
    # render a 'Model' filter section at all, so scraping can't discover their models.
    MANUAL_MODEL_OVERRIDES = {
        "Hummer": ["H2", "H3"],
    }
    for brand_name, manual_models in MANUAL_MODEL_OVERRIDES.items():
        if not results.get(brand_name):
            results[brand_name] = manual_models
            print(f"Applied manual override for {brand_name}: {manual_models}")

    total = len(brands)

    driver = setup_driver()
    try:
        for i, (brand_name, brand_value) in enumerate(brands.items(), start=1):
            # Skip brands we already successfully fetched models for
            if results.get(brand_name):
                print(f"[{i}/{total}] Skipping {brand_name} (already have {len(results[brand_name])} models)")
                continue

            slug = slug_from_brand_value(brand_value)
            print(f"[{i}/{total}] Fetching models for {brand_name} ({slug})...")

            try:
                models = get_models_for_brand(driver, slug)
            except Exception as e:
                print(f"  [!] Failed for {brand_name}: {e}")
                models = []

            results[brand_name] = models
            print(f"  -> {len(models)} models found")

            # Save progress after every brand, so nothing is lost if it stops midway
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    finally:
        driver.quit()

    print(f"\nDone. Saved model lists for {len(results)} brands to {OUTPUT_FILE}.")


if __name__ == "__main__":
    main()