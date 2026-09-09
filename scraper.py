from playwright.sync_api import sync_playwright
import pandas as pd
import re
import time

# Labels that appear in the modal header block, in the order they appear.
# Used to split the raw modal text into individual fields.
HEADER_LABELS = [
    "Faculty supervisor",
    "Specialization",
    "Faculty Province",
    "Faculty University",
    "Faculty Campus",
    "Project Location",
    "Preferred student academic background",
    "Language",
    "Preferred start date",
]

# The 5 tabs, in the order they appear as buttons in the modal.
TAB_LABELS = [
    "Project Description",
    "Student Roles",
    "Required Skills",
    "Project Activities",
    "Additional Information",
]


def parse_header_fields(modal_text: str) -> dict:
    """Extract the label/value pairs from the top of the modal using text
    positions rather than CSS classes (robust against PrimeNG's generated
    class names, which we can't inspect without live DevTools access)."""
    fields = {}
    # Find where each label starts in the text, in order.
    positions = []
    for label in HEADER_LABELS:
        idx = modal_text.find(label)
        if idx != -1:
            positions.append((idx, label))
    positions.sort(key=lambda x: x[0])

    for i, (idx, label) in enumerate(positions):
        start = idx + len(label)
        end = positions[i + 1][0] if i + 1 < len(positions) else None
        value = modal_text[start:end] if end else modal_text[start:start + 300]
        fields[label] = value.strip(" :\n")
    return fields


def extract_active_tab_content(modal_text: str) -> str:
    """The tab bar labels are always present in the text (they're buttons),
    and the currently active tab's content immediately follows the LAST tab
    label ('Additional Information'). Splitting on that gives us just the
    content of whichever tab is currently selected."""
    marker = TAB_LABELS[-1]  # "Additional Information"
    idx = modal_text.rfind(marker)
    if idx == -1:
        return ""
    content = modal_text[idx + len(marker):]
    return content.strip()


def click_tab(page, tab_name: str):
    """Click a tab button by its visible text. Falls back through a couple
    of common locator strategies since we don't have live DevTools access
    to confirm the exact element type (button/div/li)."""
    locator = page.get_by_text(tab_name, exact=True).first
    locator.click()
    time.sleep(0.6)  # brief pause for the panel content to swap in


def scrape_project_detail(page) -> dict:
    """Click 'View Detail' on the currently open card is assumed to have
    already happened; this reads the now-open modal."""
    # Wait for a field we know is always present once the modal is open.
    page.wait_for_selector("text=Faculty supervisor", timeout=8000)
    time.sleep(0.3)

    detail = {}

    # --- Header fields (visible without clicking any tab) ---
    header_text = page.locator("body").inner_text()
    header_fields = parse_header_fields(header_text)
    detail.update({
        "Faculty_Supervisor": header_fields.get("Faculty supervisor", ""),
        "Specialization": header_fields.get("Specialization", ""),
        "Faculty_Province": header_fields.get("Faculty Province", ""),
        "Faculty_University": header_fields.get("Faculty University", ""),
        "Faculty_Campus": header_fields.get("Faculty Campus", ""),
        "Project_Location": header_fields.get("Project Location", ""),
        "Preferred_Academic_Background": header_fields.get(
            "Preferred student academic background", ""),
        "Language": header_fields.get("Language", ""),
        "Preferred_Start_Date": header_fields.get("Preferred start date", ""),
    })

    # --- Each of the 5 tabs ---
    for tab_name in TAB_LABELS:
        try:
            click_tab(page, tab_name)
            full_text = page.locator("body").inner_text()
            content = extract_active_tab_content(full_text)
            key = tab_name.replace(" ", "_")
            detail[key] = content
        except Exception as e:
            print(f"      ⚠️ Could not read tab '{tab_name}': {e}")
            detail[tab_name.replace(" ", "_")] = ""

    return detail


def close_modal(page):
    """PrimeNG dialogs close on Escape almost universally, which sidesteps
    needing the exact close-icon selector."""
    page.keyboard.press("Escape")
    time.sleep(0.5)


def run_scraper():
    scraped_projects = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        print("🌐 Opening Mitacs Portal...")
        page.goto("https://globalink.mitacs.ca/#/student/application/projects")

        print("\n⏳ ACTION REQUIRED:")
        print("1. Log in to your Mitacs account.")
        print("2. Set the two filters manually (École de Technologie Supérieure + Engg-Software).")
        print("3. Click 'Search and Filter'.")
        print("4. WAIT until you see 'Total number of projects: 35' on the screen.")
        input("👉 ONCE YOU SEE THE 35 PROJECTS, PRESS ENTER HERE TO START SCRAPING... ")

        total_pages = 12

        for current_page in range(1, total_pages + 1):
            print(f"📄 Scraping Page {current_page}/{total_pages}...")

            page.wait_for_selector(".p-dataview-content", timeout=12000)
            time.sleep(2)

            project_cards = page.locator(".p-dataview-content > div.col-12")
            count = project_cards.count()
            print(f"   Found {count} cards on page {current_page}.")

            for i in range(count):
                card = project_cards.nth(i)
                text = card.inner_text().strip()
                lines = [line.strip() for line in text.split("\n") if line.strip()]

                project_id = ""
                title = ""
                for line in lines:
                    if "Project ID" in line:
                        project_id = line.replace("Project ID", "").strip()
                        break
                for line in lines:
                    if "Project ID" not in line:
                        title = line
                        break

                record = {
                    "Project_ID": project_id,
                    "Title": title,
                }

                # --- Open the detail modal for this card ---
                try:
                    view_detail_btn = card.get_by_text("View Detail", exact=True)
                    view_detail_btn.click()
                    time.sleep(1)

                    detail = scrape_project_detail(page)
                    record.update(detail)

                    close_modal(page)
                except Exception as e:
                    print(f"   ⚠️ Failed to scrape detail for '{title}': {e}")

                scraped_projects.append(record)
                print(f"      ✅ [{i+1}/{count}] {title[:60]}")

            # --- Save incrementally after every page, in case of a crash ---
            df = pd.DataFrame(scraped_projects)
            df = df.drop_duplicates(subset=["Project_ID"])
            df.to_csv("mitacs_projects.csv", index=False, encoding="utf-8")

            if current_page < total_pages:
                print("⏭️ Moving to next page...")
                next_btn = page.locator(".p-paginator-next").first
                if next_btn.is_visible() and not next_btn.is_disabled():
                    next_btn.click()
                else:
                    page.locator(f".p-paginator-page:has-text('{current_page + 1}')").first.click()
                time.sleep(3)

        browser.close()

    df = pd.DataFrame(scraped_projects)
    df = df.drop_duplicates(subset=["Project_ID"])
    df.to_csv("mitacs_projects.csv", index=False, encoding="utf-8")
    print(f"\n✅ Done! Saved {len(df)} projects to 'mitacs_projects.csv'.")


if __name__ == "__main__":
    run_scraper()