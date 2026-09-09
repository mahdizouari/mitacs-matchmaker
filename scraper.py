from playwright.sync_api import sync_playwright
import pandas as pd
import time

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

        # --- Scraping Pages ---
        total_pages = 12

        for current_page in range(1, total_pages + 1):
            print(f"📄 Scraping Page {current_page}/{total_pages}...")

            # Wait for content container to render
            page.wait_for_selector(".p-dataview-content", timeout=12000)
            time.sleep(2) # Brief pause to ensure text is fully loaded

            project_cards = page.locator(".p-dataview-content > div.col-12")
            count = project_cards.count()
            print(f"   Found {count} cards on page {current_page}.")

            for i in range(count):
                card = project_cards.nth(i)
                text = card.inner_text().strip()
                lines = [line.strip() for line in text.split("\n") if line.strip()]

                if len(lines) >= 3:
                    # Extract Project ID
                    project_id = ""
                    for line in lines:
                        if "Project ID" in line:
                            project_id = line.replace("Project ID", "").strip()
                            break

                    # Extract Project Title
                    title = ""
                    for line in lines:
                        if "Project ID" not in line:
                            title = line
                            break

                    scraped_projects.append({
                        "Project_ID": project_id,
                        "Title": title,
                        "Description": "\n".join(lines)
                    })

            # Navigate to next page
            if current_page < total_pages:
                print("⏭️ Moving to next page...")
                next_btn = page.locator(".p-paginator-next").first
                if next_btn.is_visible() and not next_btn.is_disabled():
                    next_btn.click()
                else:
                    page.locator(f".p-paginator-page:has-text('{current_page + 1}')").first.click()
                time.sleep(3) # Wait for the next page to load

        browser.close()

    # --- Save to CSV ---
    df = pd.DataFrame(scraped_projects)
    df = df.drop_duplicates(subset=["Project_ID"])
    df.to_csv("mitacs_projects.csv", index=False, encoding="utf-8")
    print(f"\n✅ Done! Saved {len(df)} projects to 'mitacs_projects.csv'.")

if __name__ == "__main__":
    run_scraper()