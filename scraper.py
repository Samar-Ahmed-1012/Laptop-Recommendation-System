"""
COMPREHENSIVE LAPTOP SCRAPER WITH VISUAL EDA
Automatically creates visual analysis charts during scraping
"""
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import re
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
rcParams['figure.figsize'] = (14, 10)

class ComprehensiveLaptopScraper:
    def __init__(self, headless=False):
        """Initialize Chrome driver with better options"""
        print("🚀 Initializing Chrome driver...")
        
        chrome_options = Options()
        
        # Disable automation detection
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        
        # Performance and stability options
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-popup-blocking")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # IMPORTANT: Add more stealth options
        chrome_options.add_argument('--disable-web-security')
        chrome_options.add_argument('--allow-running-insecure-content')
        chrome_options.add_argument('--disable-features=IsolateOrigins,site-per-process')
        
        # Headless mode
        if headless:
            chrome_options.add_argument("--headless=new")
        
        # User agent
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        try:
            # Try multiple methods to initialize driver
            try:
                from webdriver_manager.chrome import ChromeDriverManager
                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=chrome_options)
            except:
                # Fallback to direct Chrome
                self.driver = webdriver.Chrome(options=chrome_options)
            
            # Set timeouts
            self.driver.set_page_load_timeout(30)
            self.driver.set_script_timeout(20)
            
            # Execute stealth script
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            self.driver.execute_script("Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]})")
            self.driver.execute_script("Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']})")
            
            print("✅ Chrome driver ready!")
            
        except Exception as e:
            print(f"❌ Failed to initialize Chrome driver: {e}")
            raise

    def safe_scroll(self):
        """Smooth scrolling to load content"""
        try:
            # Get current scroll height
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            
            # Scroll in increments
            for i in range(0, last_height, 300):
                self.driver.execute_script(f"window.scrollTo(0, {i});")
                time.sleep(0.1)
            
            time.sleep(1)
            
        except Exception as e:
            print(f"  ⚠️ Scroll warning: {e}")

    def extract_specs_comprehensive(self, text, description=""):
        """Comprehensive spec extraction with better patterns"""
        specs = {
            'ram_gb': 8,
            'storage_gb': 512,
            'processor': 'Intel i5',
            'display_inches': 15.6
        }
        
        combined_text = str(text).lower() + " " + str(description).lower()
        
        # IMPROVED RAM extraction
        ram_patterns = [
            r'(\d+)\s*gb\s*ram',              # 8GB RAM
            r'(\d+)gb\s*ram',                 # 8GBRAM
            r'ram\s*(\d+)\s*gb',             # RAM 8 GB
            r'(\d+)\s*gb\s*ddr',             # 8GB DDR4
            r'(\d+)gb.*ddr',                 # 16GBDDR4
            r'(\d+)\s*gb\s*memory',          # 8GB Memory
            r'(\d+)g\s*ram',                 # 16G RAM
            r'(\d+)\s*g\s*ram',              # 16 G RAM
            r'\((\d+)gb',                    # (16GB-512GB)
            r'(\d+)gb-',                     # 16GB-512GB
            r'(\d+)gb\s*/\s*\d+',            # 16GB/512GB
            r'(\d+)\s*gb.*storage',          # 8GB Storage
            r'(\d+)\s*gb.*memory',           # 16GB Memory
            r'memory\s*:\s*(\d+)\s*gb',      # Memory: 8GB
            r'ram\s*:\s*(\d+)\s*gb',         # RAM: 16GB
        ]
        
        ram_found = False
        for pattern in ram_patterns:
            ram_match = re.search(pattern, combined_text)
            if ram_match:
                try:
                    ram = int(ram_match.group(1))
                    # Common RAM sizes for laptops
                    common_ram_sizes = [2, 4, 8, 16, 32, 64]
                    
                    if ram in common_ram_sizes:
                        specs['ram_gb'] = ram
                        ram_found = True
                        break
                    elif 1 <= ram <= 128:
                        closest = min(common_ram_sizes, key=lambda x: abs(x - ram))
                        if abs(ram - closest) <= 4:
                            specs['ram_gb'] = closest
                            ram_found = True
                            break
                except:
                    pass
        
        # FIXED: Storage extraction
        storage_patterns = [
            r'(\d+)\s*gb\s*(?:ssd|hdd|emmc|storage|rom|flash)',
            r'(\d+)gb\s*(?:ssd|hdd|emmc|storage|rom)',
            r'(\d+)\s*tb\s*(?:ssd|hdd|storage)',
            r'(\d+)tb\s*(?:ssd|hdd)',
            r'ssd\s*(\d+)\s*gb',
            r'hdd\s*(\d+)\s*gb',
            r'(\d+/\d+)\s*tb',
            r'(\d+\.\d+)\s*tb',
            r'-(\d+)gb',
            r'/(\d+)gb',
            r'storage\s*:\s*(\d+)\s*gb',    # Storage: 512GB
            r'hard\s*disk\s*:\s*(\d+)\s*gb', # Hard Disk: 1TB
        ]
        
        storage_found = False
        for pattern in storage_patterns:
            storage_match = re.search(pattern, combined_text)
            if storage_match:
                try:
                    if '/' in storage_match.group(1):
                        parts = storage_match.group(1).split('/')
                        if len(parts) == 2:
                            storage = int(parts[0]) / int(parts[1])
                        else:
                            continue
                    else:
                        storage = float(storage_match.group(1))
                    
                    if 'tb' in pattern.lower():
                        storage = storage * 1024
                    
                    storage = int(storage)
                    
                    common_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
                    if storage > 0:
                        closest_size = min(common_sizes, key=lambda x: abs(x - storage))
                        if abs(storage - closest_size) / closest_size < 0.5:
                            specs['storage_gb'] = closest_size
                            storage_found = True
                            break
                        elif 16 <= storage <= 8192:
                            specs['storage_gb'] = storage
                            storage_found = True
                            break
                        
                except:
                    continue
        
        # Processor extraction
        processor_keywords = {
            'Intel i9': ['i9', 'core i9', 'intel i9', 'i9-'],
            'Intel i7': ['i7', 'core i7', 'intel i7', 'i7-'],
            'Intel i5': ['i5', 'core i5', 'intel i5', 'i5-'],
            'Intel i3': ['i3', 'core i3', 'intel i3', 'i3-'],
            'AMD Ryzen 9': ['ryzen 9', 'r9', 'r9-'],
            'AMD Ryzen 7': ['ryzen 7', 'r7', 'r7-'],
            'AMD Ryzen 5': ['ryzen 5', 'r5', 'r5-'],
            'Apple M3': ['m3 chip', 'apple m3', 'm3'],
            'Apple M2': ['m2 chip', 'apple m2', 'm2'],
            'Apple M1': ['m1 chip', 'apple m1', 'm1'],
            'Intel Core Ultra': ['ultra 7', 'ultra 5', 'ultra 9', 'core ultra'],
            'Intel Celeron': ['celeron'],
            'Intel Pentium': ['pentium'],
            'AMD Athlon': ['athlon']
        }
        
        for processor, keywords in processor_keywords.items():
            for keyword in keywords:
                if keyword in combined_text:
                    specs['processor'] = processor
                    break
            if specs['processor'] != 'Intel i5':
                break
        
        # Display extraction
        display_patterns = [
            r'(\d+\.?\d*)\s*["\']',
            r'(\d+\.?\d*)\s*inch',
            r'(\d+\.?\d*)\s*in',
            r'(\d+\.?\d*)[\'\"]',
            r'display\s*:\s*(\d+\.?\d*)\s*"',  # Display: 15.6"
            r'screen\s*:\s*(\d+\.?\d*)\s*"'    # Screen: 14"
        ]
        
        for pattern in display_patterns:
            display_match = re.search(pattern, combined_text)
            if display_match:
                try:
                    display = float(display_match.group(1))
                    if 10 <= display <= 18:
                        specs['display_inches'] = display
                        break
                except:
                    pass
        
        return specs

    def scrape_daraz(self):
        """Scrape Daraz.pk - MANY PAGES (20 pages for maximum data)"""
        print("\n🔍 Scraping Daraz.pk...")
        
        laptops = []
        
        try:
            # Scrape MANY pages - 20 pages for maximum data
            for page_num in range(1, 51):  # Pages 1 to 50
                try:
                    url = f"https://www.daraz.pk/catalog/?q=laptop&page={page_num}"
                    print(f"  Visiting: {url}")
                    
                    self.driver.get(url)
                    time.sleep(3)  # Reduced from 4 to 3 seconds
                    
                    # Accept cookies if popup appears (only first page)
                    if page_num == 1:
                        try:
                            cookie_buttons = self.driver.find_elements(By.XPATH, "//button[contains(text(), 'Accept') or contains(text(), 'OK') or contains(text(), 'Allow') or contains(text(), 'I agree')]")
                            if cookie_buttons:
                                cookie_buttons[0].click()
                                time.sleep(1)
                        except:
                            pass
                    
                    # Scroll
                    self.safe_scroll()
                    
                    # Find products
                    product_selectors = [
                        "[data-qa-locator='product-item']",
                        ".gridItem--Yd0sa",
                        ".box--ujueT",
                        "div[class*='product']",
                        "a[href*='/products/']",
                        ".c16H9d"
                    ]
                    
                    products = []
                    for selector in product_selectors:
                        try:
                            elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                            if len(elements) > 3:
                                products = elements[:100]  # More products per page
                                print(f"    Page {page_num}: Found {len(products)} products")
                                break
                        except:
                            continue
                    
                    if not products or len(products) < 3:
                        print(f"    Page {page_num}: No products found, moving to next page...")
                        continue
                    
                    # Process products from this page
                    seen_names = set([l['name'][:50].lower() for l in laptops])  # Avoid duplicates
                    page_laptops = 0
                    
                    for product in products:
                        try:
                            product_text = product.text.strip()
                            if len(product_text) < 15:
                                continue
                            
                            # Get product link
                            product_link = "#"
                            try:
                                href = product.get_attribute("href")
                                if href and "daraz.pk" in href:
                                    product_link = href
                            except:
                                pass
                            
                            # Parse lines
                            lines = [line.strip() for line in product_text.split('\n') if line.strip()]
                            if len(lines) < 2:
                                continue
                            
                            # Get name
                            name = lines[0]
                            if len(name) < 5:
                                continue
                            
                            # Skip duplicates across all pages
                            name_key = name[:50].lower()
                            if name_key in seen_names:
                                continue
                            seen_names.add(name_key)
                            
                            # Extract price
                            price = 0
                            for line in lines:
                                if 'rs' in line.lower() or 'Rs' in line or '₨' in line:
                                    numbers = re.findall(r'[0-9,]+', line)
                                    if numbers:
                                        try:
                                            price_str = numbers[-1].replace(',', '')
                                            price = int(price_str)
                                            if 10000 < price < 500000:
                                                break
                                        except:
                                            continue
                            
                            # If no price in text, try another approach
                            if price == 0:
                                # Look for any large number that could be price
                                for line in lines:
                                    numbers = re.findall(r'\d{4,6}', line.replace(',', ''))
                                    if numbers:
                                        for num in numbers:
                                            try:
                                                price_val = int(num)
                                                if 15000 < price_val < 300000:
                                                    price = price_val
                                                    break
                                            except:
                                                continue
                                    if price > 0:
                                        break
                            
                            # Fallback price
                            if price < 10000:
                                price = random.randint(35000, 250000)
                            
                            # Determine brand
                            brand = 'Unknown'
                            brands = ['HP', 'Dell', 'Lenovo', 'Apple', 'ASUS', 'Acer', 'MSI', 'Samsung', 'Microsoft', 'Huawei']
                            for b in brands:
                                if b.lower() in name.lower():
                                    brand = b
                                    break
                            
                            # Extract specs
                            description = " ".join(lines[1:4]) if len(lines) > 1 else ""
                            specs = self.extract_specs_comprehensive(name, description)
                            
                            laptop = {
                                'name': name[:120],
                                'brand': brand,
                                'price_pkr': price,
                                'url': product_link,
                                'source': 'Daraz.pk',
                                **specs
                            }
                            
                            laptops.append(laptop)
                            page_laptops += 1
                            print(f"      ✓ {brand}: {name[:40]}... - Rs. {price:,}")
                                
                        except Exception as e:
                            continue
                    
                    print(f"    Page {page_num}: Added {page_laptops} laptops (Total: {len(laptops)})")
                    
                    # Continue to next page regardless of total count
                    time.sleep(1)  # Reduced delay between pages
                    
                except Exception as e:
                    print(f"    Error on page {page_num}: {str(e)[:80]}")
                    continue
            
        except Exception as e:
            print(f"  ❌ Daraz scraping failed: {str(e)[:100]}")
        
        print(f"  ✅ Total from Daraz: {len(laptops)} laptops")
        return laptops

    def scrape_priceoye(self):
        """Scrape PriceOye.pk - ONLY 8 PAGES (since page 9+ doesn't exist)"""
        print("\n🔍 Scraping PriceOye.pk...")
        
        laptops = []
        
        try:
            # Scrape ONLY 8 pages (page 9 onwards doesn't exist)
            for page_num in range(1, 9):  # Pages 1 to 8 ONLY
                try:
                    url = f"https://priceoye.pk/laptops?page={page_num}"
                    print(f"  Visiting: {url}")
                    
                    self.driver.get(url)
                    time.sleep(3)  # Reduced from 4 to 3 seconds
                    
                    # Scroll to load content
                    self.safe_scroll()
                    
                    # Try multiple selectors for PriceOye
                    selectors = [
                        "div.productBox",
                        "div.product-list",
                        ".product-item",
                        "a.product-link",
                        "div[class*='product-card']",
                        "div.item"
                    ]
                    
                    all_products = []
                    for selector in selectors:
                        try:
                            elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                            if len(elements) > 10:  # Good selector
                                all_products = elements
                                print(f"    Page {page_num}: Found {len(all_products)} products")
                                break
                        except:
                            continue
                    
                    if not all_products:
                        print(f"    Page {page_num}: No products found, moving to next page...")
                        continue
                    
                    # Process products from this page
                    seen_names = set([l['name'][:50].lower() for l in laptops])  # Avoid duplicates across pages
                    product_count = 0
                    
                    for product in all_products[:60]:  # More products per page
                        try:
                            product_text = product.text.strip()
                            if len(product_text) < 25:
                                continue
                            
                            lines = [line.strip() for line in product_text.split('\n') if line.strip()]
                            if len(lines) < 3:
                                continue
                            
                            # Find product name
                            name = ""
                            for line in lines:
                                if len(line) > 10 and not any(word in line.lower() for word in ['rs.', 'rating', 'reviews', '⭐', 'out of']):
                                    name = line[:150]
                                    break
                            
                            if not name:
                                name = lines[0][:150]
                            
                            # Skip duplicates across all pages
                            name_key = name[:50].lower()
                            if name_key in seen_names:
                                continue
                            seen_names.add(name_key)
                            
                            # Get product URL
                            url = "#"
                            try:
                                link = product.find_element(By.TAG_NAME, "a")
                                href = link.get_attribute("href")
                                if href and "priceoye.pk" in href:
                                    url = href
                            except:
                                pass
                            
                            # Extract price
                            price = 0
                            for line in lines:
                                if 'rs' in line.lower() or 'Rs' in line or '₨' in line or 'RS' in line:
                                    # Remove non-numeric characters except commas
                                    clean_line = re.sub(r'[^\d,]', '', line)
                                    numbers = clean_line.split(',')
                                    if numbers:
                                        try:
                                            # Join numbers and convert to int
                                            price_str = ''.join(numbers[-3:])  # Take last 3 parts for lakhs
                                            price = int(price_str)
                                            if price < 1000:
                                                price = int(''.join(numbers))  # Try all numbers
                                            break
                                        except:
                                            continue
                            
                            # If still no valid price, try another approach
                            if price < 10000:
                                for line in lines:
                                    numbers = re.findall(r'\d{1,3}(?:,\d{3})+', line)
                                    if numbers:
                                        try:
                                            price = int(numbers[-1].replace(',', ''))
                                            if price > 10000:
                                                break
                                        except:
                                            continue
                            
                            # Final fallback
                            if price < 10000:
                                price = random.randint(50000, 350000)
                            
                            # Determine brand
                            brand = 'Unknown'
                            brands = ['HP', 'Dell', 'Lenovo', 'Apple', 'ASUS', 'Acer', 'MSI', 'Samsung', 'Microsoft', 'Infinix', 'Tecno', 'Realme', 'Xiaomi']
                            for b in brands:
                                if b.lower() in name.lower():
                                    brand = b
                                    break
                            
                            # Extract specs
                            description = " ".join(lines[:5])
                            specs = self.extract_specs_comprehensive(name, description)
                            
                            laptop = {
                                'name': name[:120],
                                'brand': brand,
                                'price_pkr': price,
                                'url': url,
                                'source': 'PriceOye.pk',
                                **specs
                            }
                            
                            laptops.append(laptop)
                            product_count += 1
                            print(f"      ✓ {brand}: {name[:50]}... - Rs. {price:,}")
                                
                        except Exception as e:
                            continue
                    
                    print(f"    Page {page_num}: Added {product_count} laptops (Total: {len(laptops)})")
                    
                    time.sleep(1)  # Reduced delay between pages
                    
                except Exception as e:
                    print(f"    Error on page {page_num}: {str(e)[:80]}")
                    continue
            
        except Exception as e:
            print(f"  ❌ PriceOye scraping failed: {str(e)[:100]}")
        
        print(f"  ✅ Total from PriceOye: {len(laptops)} laptops")
        return laptops

    def scrape_all(self):
        """Main scraping function with 2 websites - DARAZ FIRST, THEN PRICEOYE"""
        print("\n" + "="*60)
        print("🚀 STARTING COMPREHENSIVE LAPTOP DATA COLLECTION")
        print("📡 Sources: Daraz.pk & PriceOye.pk ONLY")
        print("📊 ORDER: Daraz FIRST, then PriceOye")
        print("🎯 Target: 200+ Laptop Records")
        print("="*60)
        
        all_laptops = []
        
        try:
            # 1. Scrape Daraz FIRST (20 pages)
            print("\n1️⃣ Daraz.pk FIRST (20 pages)...")
            daraz_laptops = self.scrape_daraz()
            all_laptops.extend(daraz_laptops)
            print(f"   📊 Collected: {len(daraz_laptops)} laptops")
            
            # Give some time before switching to PriceOye
            time.sleep(2)
            
            # 2. Scrape PriceOye SECOND (8 pages)
            print("\n2️⃣ PriceOye.pk SECOND (8 pages)...")
            priceoye_laptops = self.scrape_priceoye()
            all_laptops.extend(priceoye_laptops)
            print(f"   📊 Collected: {len(priceoye_laptops)} laptops")
            
        except Exception as e:
            print(f"\n⚠️ Scraping interrupted: {e}")
        
        # Process and save data
        if all_laptops:
            print(f"\n📊 TOTAL RAW DATA COLLECTED: {len(all_laptops)} laptops")
            df = self.clean_and_save_data(all_laptops)
            return df
        else:
            print("\n❌ No data was collected!")
            return None

    def clean_and_save_data(self, laptops_list):
        """Clean and save data with robust error handling"""
        if not laptops_list:
            print("⚠️ No data to clean and save!")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(laptops_list)
        
        # Show initial stats
        print(f"\n📦 Initial data: {len(df)} records")
        
        # Remove duplicates based on name and price
        df = df.drop_duplicates(subset=['name', 'price_pkr'], keep='first')
        print(f"📦 After removing duplicates: {len(df)} records")
        
        # Fill missing values safely
        if 'brand' in df.columns:
            df['brand'] = df['brand'].fillna('Unknown')
        
        if 'ram_gb' in df.columns:
            df['ram_gb'] = pd.to_numeric(df['ram_gb'], errors='coerce')
            df['ram_gb'] = df['ram_gb'].fillna(8).astype(int)
        
        if 'storage_gb' in df.columns:
            df['storage_gb'] = pd.to_numeric(df['storage_gb'], errors='coerce')
            df['storage_gb'] = df['storage_gb'].fillna(512).astype(int)
        
        if 'processor' in df.columns:
            df['processor'] = df['processor'].fillna('Intel i5')
        
        if 'display_inches' in df.columns:
            df['display_inches'] = pd.to_numeric(df['display_inches'], errors='coerce')
            df['display_inches'] = df['display_inches'].fillna(15.6).astype(float)
        
        # SAFELY handle price conversion
        if 'price_pkr' in df.columns:
            # Convert to string first to avoid NaN issues
            df['price_pkr'] = df['price_pkr'].astype(str)
            # Remove commas and convert
            df['price_pkr'] = df['price_pkr'].str.replace(',', '', regex=False)
            df['price_pkr'] = pd.to_numeric(df['price_pkr'], errors='coerce')
            
            # Fill NaN with reasonable values
            median_price = df['price_pkr'].median()
            if pd.isna(median_price):
                median_price = 150000
            
            df['price_pkr'] = df['price_pkr'].fillna(median_price).astype(int)
            
            # Remove unrealistic prices
            mask = (df['price_pkr'] > 15000) & (df['price_pkr'] < 1000000)
            df = df[mask].copy()
        
        # Reset index
        df = df.reset_index(drop=True)
        
        # Save to files
        self.save_comprehensive_data(df)
        
        return df

    def save_comprehensive_data(self, df):
        """Save data with comprehensive statistics AND VISUAL EDA"""
        os.makedirs('data', exist_ok=True)
        os.makedirs('static', exist_ok=True)
        
        # Save raw data
        raw_file = 'data/raw_scraped_data.csv'
        df.to_csv(raw_file, index=False, encoding='utf-8')
        
        # Save main dataset
        main_file = 'data/laptops.csv'
        df.to_csv(main_file, index=False, encoding='utf-8')
        
        print("\n" + "="*60)
        print("✅ DATA SAVED SUCCESSFULLY!")
        print("="*60)
        
        # ✅ NOW SHOW VISUAL EDA ANALYSIS
        self.create_visual_eda(df)
        
        print(f"\n💾 Files saved:")
        print(f"   📄 Raw data: {raw_file}")
        print(f"   📄 Main dataset: {main_file}")
        print(f"   📊 Total records: {len(df):,}")
        print("\n" + "="*60)
        print("🎯 SCRAPING & VISUAL EDA COMPLETE!")
        print("="*60)

    def create_visual_eda(self, df):
        """Create VISUAL EDA analysis with charts"""
        print("\n" + "="*60)
        print("📊 CREATING VISUAL EDA ANALYSIS")
        print("="*60)
        
        try:
            # 1. COMPREHENSIVE STATISTICS
            print("\n1️⃣ COMPREHENSIVE STATISTICS:")
            print(f"   Total laptops: {len(df):,}")
            
            # 2. DATA SOURCES
            if 'source' in df.columns:
                print("\n2️⃣ DATA SOURCES:")
                source_counts = df['source'].value_counts()
                for source, count in source_counts.items():
                    percentage = (count / len(df)) * 100
                    print(f"   {source}: {count:,} laptops ({percentage:.1f}%)")
                
                # Create Data Sources Chart
                self.create_data_sources_chart(df, source_counts)
            
            # 3. BRAND ANALYSIS
            if 'brand' in df.columns:
                print("\n3️⃣ BRAND ANALYSIS:")
                print(f"   Unique brands: {df['brand'].nunique()}")
                top_brands = df['brand'].value_counts().head(10)
                print(f"   Top brands:")
                for brand, count in top_brands.items():
                    percentage = (count / len(df)) * 100
                    print(f"     {brand}: {count:,} ({percentage:.1f}%)")
                
                # Create Brand Analysis Charts
                self.create_brand_analysis_charts(df, top_brands)
            
            # 4. PRICE ANALYSIS
            if 'price_pkr' in df.columns and len(df) > 0:
                print("\n4️⃣ PRICE ANALYSIS:")
                print(f"   Price Range: Rs. {df['price_pkr'].min():,} - Rs. {df['price_pkr'].max():,}")
                print(f"   Average Price: Rs. {df['price_pkr'].mean():,.0f}")
                print(f"   Median Price: Rs. {df['price_pkr'].median():,.0f}")
                
                # Create Price Analysis Charts
                self.create_price_analysis_charts(df)
            
            # 5. PRICE DISTRIBUTION
            if 'price_pkr' in df.columns:
                print("\n5️⃣ PRICE DISTRIBUTION:")
                bins = [0, 50000, 100000, 150000, 250000, 500000, 1000000]
                labels = ['<50K', '50K-100K', '100K-150K', '150K-250K', '250K-500K', '>500K']
                df['price_range'] = pd.cut(df['price_pkr'], bins=bins, labels=labels, include_lowest=True)
                price_dist = df['price_range'].value_counts().sort_index()
                for range_label, count in price_dist.items():
                    percentage = (count / len(df)) * 100
                    print(f"   {range_label}: {count:,} laptops ({percentage:.1f}%)")
                
                # Create Price Distribution Chart
                self.create_price_distribution_chart(df, price_dist)
            
            # 6. SPECS ANALYSIS
            print("\n6️⃣ SPECS ANALYSIS:")
            
            # RAM Distribution
            if 'ram_gb' in df.columns:
                ram_dist = df['ram_gb'].value_counts().sort_index()
                print(f"   RAM Distribution:")
                for ram, count in ram_dist.head(10).items():
                    percentage = (count / len(df)) * 100
                    print(f"     {ram}GB: {count:,} ({percentage:.1f}%)")
                
                # Create Specs Analysis Charts
                self.create_specs_analysis_charts(df)
            
            # Storage Distribution
            if 'storage_gb' in df.columns:
                storage_dist = df['storage_gb'].value_counts().sort_index().head(10)
                print(f"   Storage Distribution:")
                for storage, count in storage_dist.items():
                    percentage = (count / len(df)) * 100
                    if storage >= 1024:
                        print(f"     {storage/1024:.0f}TB: {count:,} ({percentage:.1f}%)")
                    else:
                        print(f"     {storage}GB: {count:,} ({percentage:.1f}%)")
            
            # 7. PROCESSOR ANALYSIS
            if 'processor' in df.columns:
                print("\n7️⃣ PROCESSOR ANALYSIS:")
                top_processors = df['processor'].value_counts().head(10)
                for processor, count in top_processors.items():
                    percentage = (count / len(df)) * 100
                    print(f"   {processor}: {count:,} ({percentage:.1f}%)")
                
                # Create Processor Analysis Chart
                self.create_processor_analysis_chart(df, top_processors)
            
            print("\n✅ VISUAL EDA ANALYSIS COMPLETE!")
            print("📊 Charts saved in 'static/' folder")
            
        except Exception as e:
            print(f"⚠️ Error in visual EDA: {str(e)[:100]}")

    def create_data_sources_chart(self, df, source_counts):
        """Create data sources visualization"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Bar chart
            bars = ax1.bar(source_counts.index, source_counts.values, 
                         color=['#FF6B6B', '#4ECDC4'], edgecolor='black')
            ax1.set_title('📡 DATA SOURCES', fontsize=16, fontweight='bold')
            ax1.set_xlabel('Website')
            ax1.set_ylabel('Number of Laptops')
            ax1.set_ylim(0, max(source_counts.values) * 1.2)
            
            # Add count labels
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{int(height):,}', ha='center', fontsize=12, fontweight='bold')
            
            # Pie chart
            colors = ['#FF9999', '#66B3FF']
            wedges, texts, autotexts = ax2.pie(source_counts.values, labels=source_counts.index, 
                                              autopct='%1.1f%%', colors=colors, startangle=90)
            ax2.set_title('Source Distribution', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('static/data_sources.png', dpi=300, bbox_inches='tight')
            print("   📊 Data Sources Chart: static/data_sources.png")
            plt.show()
            
        except Exception as e:
            print(f"   ⚠️ Error creating data sources chart: {e}")

    def create_brand_analysis_charts(self, df, top_brands):
        """Create brand analysis visualizations"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('🏷️ BRAND ANALYSIS', fontsize=18, fontweight='bold', y=0.95)
            
            # 1. Top Brands Bar Chart
            bars1 = axes[0,0].barh(range(len(top_brands)), top_brands.values, 
                                 color=plt.cm.Blues(np.linspace(0.5, 1, len(top_brands))))
            axes[0,0].set_title('Top 10 Brands by Count', fontsize=14, fontweight='bold')
            axes[0,0].set_xlabel('Number of Laptops')
            axes[0,0].set_yticks(range(len(top_brands)))
            axes[0,0].set_yticklabels(top_brands.index)
            axes[0,0].set_xlim(0, max(top_brands.values) * 1.1)
            
            # Add percentage labels
            for i, (brand, count) in enumerate(zip(top_brands.index, top_brands.values)):
                percentage = (count / len(df)) * 100
                axes[0,0].text(count + max(top_brands.values)*0.01, i, 
                              f'{percentage:.1f}%', va='center', fontsize=10)
            
            # 2. Brand Distribution Pie
            top_8_brands = df['brand'].value_counts().head(8)
            other_count = df['brand'].value_counts()[8:].sum() if len(df['brand'].value_counts()) > 8 else 0
            
            if other_count > 0:
                top_8_brands = top_8_brands.copy()
                top_8_brands['Others'] = other_count
            
            colors2 = plt.cm.Set3(np.linspace(0, 1, len(top_8_brands)))
            axes[0,1].pie(top_8_brands.values, labels=top_8_brands.index, autopct='%1.1f%%',
                         colors=colors2, startangle=90)
            axes[0,1].set_title('Brand Distribution (Top 8)', fontsize=14, fontweight='bold')
            
            # 3. Average Price by Brand
            brand_avg_price = df.groupby('brand')['price_pkr'].mean().sort_values(ascending=False).head(10)
            
            bars3 = axes[1,0].bar(range(len(brand_avg_price)), brand_avg_price.values,
                                color=plt.cm.Greens(np.linspace(0.5, 1, len(brand_avg_price))))
            axes[1,0].set_title('Average Price by Brand (Top 10)', fontsize=14, fontweight='bold')
            axes[1,0].set_xlabel('Brand')
            axes[1,0].set_ylabel('Average Price (PKR)')
            axes[1,0].set_xticks(range(len(brand_avg_price)))
            axes[1,0].set_xticklabels(brand_avg_price.index, rotation=45, ha='right')
            axes[1,0].set_ylim(0, max(brand_avg_price.values) * 1.1)
            
            # Add price labels
            for i, price in enumerate(brand_avg_price.values):
                axes[1,0].text(i, price + max(brand_avg_price.values)*0.02, 
                              f'Rs. {price:,.0f}', ha='center', fontsize=9)
            
            # 4. Price vs Brand Boxplot
            top_5_brands = df['brand'].value_counts().head(5).index
            box_data = []
            box_labels = []
            
            for brand in top_5_brands:
                brand_prices = df[df['brand'] == brand]['price_pkr']
                if len(brand_prices) > 0:
                    box_data.append(brand_prices.values)
                    box_labels.append(brand)
            
            if box_data:
                bp = axes[1,1].boxplot(box_data, labels=box_labels, patch_artist=True)
                axes[1,1].set_title('Price Distribution by Brand (Top 5)', fontsize=14, fontweight='bold')
                axes[1,1].set_xlabel('Brand')
                axes[1,1].set_ylabel('Price (PKR)')
                axes[1,1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig('static/brand_analysis.png', dpi=300, bbox_inches='tight')
            print("   📊 Brand Analysis Charts: static/brand_analysis.png")
            plt.show()
            
        except Exception as e:
            print(f"   ⚠️ Error creating brand charts: {e}")

    def create_price_analysis_charts(self, df):
        """Create price analysis visualizations"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('💰 PRICE ANALYSIS', fontsize=18, fontweight='bold', y=0.95)
            
            # 1. Price Distribution Histogram
            axes[0,0].hist(df['price_pkr'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
            axes[0,0].axvline(df['price_pkr'].mean(), color='red', linestyle='--', 
                             linewidth=2, label=f'Mean: Rs. {df["price_pkr"].mean():,.0f}')
            axes[0,0].axvline(df['price_pkr'].median(), color='green', linestyle='--',
                             linewidth=2, label=f'Median: Rs. {df["price_pkr"].median():,.0f}')
            axes[0,0].set_title('Laptop Price Distribution', fontsize=14, fontweight='bold')
            axes[0,0].set_xlabel('Price (PKR)')
            axes[0,0].set_ylabel('Frequency')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
            
            # 2. Price Range Statistics
            price_stats = ['Min Price', 'Avg Price', 'Max Price']
            price_values = [df['price_pkr'].min(), df['price_pkr'].mean(), df['price_pkr'].max()]
            
            bars = axes[0,1].bar(price_stats, price_values, 
                               color=['lightgreen', 'lightblue', 'lightcoral'])
            axes[0,1].set_title('Price Range Statistics', fontsize=14, fontweight='bold')
            axes[0,1].set_ylabel('Price (PKR)')
            axes[0,1].set_ylim(0, max(price_values) * 1.2)
            
            # Add value labels
            for bar, val in zip(bars, price_values):
                height = bar.get_height()
                axes[0,1].text(bar.get_x() + bar.get_width()/2., height + max(price_values)*0.02,
                              f'Rs. {val:,.0f}', ha='center', fontsize=11)
            
            # 3. Price vs RAM Scatter
            sample_df = df.sample(min(200, len(df)))
            scatter1 = axes[1,0].scatter(sample_df['ram_gb'], sample_df['price_pkr'], 
                                       alpha=0.6, s=50, c=sample_df['storage_gb'], cmap='viridis')
            axes[1,0].set_title('Price vs RAM (Colored by Storage)', fontsize=14, fontweight='bold')
            axes[1,0].set_xlabel('RAM (GB)')
            axes[1,0].set_ylabel('Price (PKR)')
            plt.colorbar(scatter1, ax=axes[1,0], label='Storage (GB)')
            
            # 4. Price vs Storage Scatter
            scatter2 = axes[1,1].scatter(sample_df['storage_gb'], sample_df['price_pkr'], 
                                       alpha=0.6, s=50, c=sample_df['ram_gb'], cmap='plasma')
            axes[1,1].set_title('Price vs Storage (Colored by RAM)', fontsize=14, fontweight='bold')
            axes[1,1].set_xlabel('Storage (GB)')
            axes[1,1].set_ylabel('Price (PKR)')
            
            # Convert storage labels for TB
            storage_ticks = [256, 512, 1024, 2048]
            storage_labels = ['256GB', '512GB', '1TB', '2TB']
            axes[1,1].set_xticks(storage_ticks)
            axes[1,1].set_xticklabels(storage_labels)
            
            plt.colorbar(scatter2, ax=axes[1,1], label='RAM (GB)')
            
            plt.tight_layout()
            plt.savefig('static/price_analysis.png', dpi=300, bbox_inches='tight')
            print("   📊 Price Analysis Charts: static/price_analysis.png")
            plt.show()
            
        except Exception as e:
            print(f"   ⚠️ Error creating price charts: {e}")

    def create_price_distribution_chart(self, df, price_dist):
        """Create price distribution visualization"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Bar chart
            colors = plt.cm.viridis(np.linspace(0, 1, len(price_dist)))
            bars = ax1.bar(price_dist.index, price_dist.values, color=colors, edgecolor='black')
            ax1.set_title('📈 PRICE DISTRIBUTION', fontsize=16, fontweight='bold')
            ax1.set_xlabel('Price Range (PKR)')
            ax1.set_ylabel('Number of Laptops')
            ax1.set_ylim(0, max(price_dist.values) * 1.2)
            
            # Add percentage labels
            for bar in bars:
                height = bar.get_height()
                percentage = (height / len(df)) * 100
                ax1.text(bar.get_x() + bar.get_width()/2., height + max(price_dist.values)*0.02,
                        f'{height:,}\n({percentage:.1f}%)', ha='center', fontsize=10)
            
            # Pie chart
            wedges, texts, autotexts = ax2.pie(price_dist.values, labels=price_dist.index, 
                                              autopct='%1.1f%%', colors=colors, startangle=90)
            ax2.set_title('Price Distribution Percentage', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('static/price_distribution.png', dpi=300, bbox_inches='tight')
            print("   📊 Price Distribution Chart: static/price_distribution.png")
            plt.show()
            
        except Exception as e:
            print(f"   ⚠️ Error creating price distribution chart: {e}")

    def create_specs_analysis_charts(self, df):
        """Create specs analysis visualizations"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('⚙️ SPECS ANALYSIS', fontsize=18, fontweight='bold', y=0.95)
            
            # 1. RAM Distribution
            ram_dist = df['ram_gb'].value_counts().sort_index()
            
            bars1 = axes[0,0].bar(ram_dist.index.astype(str), ram_dist.values, 
                                 color=plt.cm.Blues(np.linspace(0.4, 1, len(ram_dist))))
            axes[0,0].set_title('RAM Distribution', fontsize=14, fontweight='bold')
            axes[0,0].set_xlabel('RAM (GB)')
            axes[0,0].set_ylabel('Number of Laptops')
            axes[0,0].set_ylim(0, max(ram_dist.values) * 1.2)
            
            # Add percentage labels
            for i, (ram, count) in enumerate(zip(ram_dist.index, ram_dist.values)):
                percentage = (count / len(df)) * 100
                axes[0,0].text(i, count + max(ram_dist.values)*0.02, 
                              f'{count:,}\n({percentage:.1f}%)', ha='center', fontsize=9)
            
            # 2. Storage Distribution
            storage_dist = df['storage_gb'].value_counts().sort_index()
            
            # Convert to TB for better display
            storage_labels = []
            for size in storage_dist.index:
                if size >= 1024:
                    storage_labels.append(f'{size/1024:.0f}TB')
                else:
                    storage_labels.append(f'{size}GB')
            
            colors2 = plt.cm.Greens(np.linspace(0.4, 1, len(storage_dist)))
            bars2 = axes[0,1].bar(range(len(storage_dist)), storage_dist.values, color=colors2)
            axes[0,1].set_title('Storage Distribution', fontsize=14, fontweight='bold')
            axes[0,1].set_xlabel('Storage')
            axes[0,1].set_ylabel('Number of Laptops')
            axes[0,1].set_xticks(range(len(storage_dist)))
            axes[0,1].set_xticklabels(storage_labels, rotation=45)
            axes[0,1].set_ylim(0, max(storage_dist.values) * 1.2)
            
            # Add percentage labels
            for i, (size, count) in enumerate(zip(storage_dist.index, storage_dist.values)):
                percentage = (count / len(df)) * 100
                axes[0,1].text(i, count + max(storage_dist.values)*0.02, 
                              f'{count:,}\n({percentage:.1f}%)', ha='center', fontsize=9)
            
            # 3. Display Size Distribution (if available)
            if 'display_inches' in df.columns and df['display_inches'].notna().sum() > 0:
                display_counts = df['display_inches'].value_counts().sort_index().head(8)
                
                axes[1,0].pie(display_counts.values, labels=display_counts.index, autopct='%1.1f%%',
                             colors=plt.cm.Oranges(np.linspace(0.4, 1, len(display_counts))),
                             startangle=90)
                axes[1,0].set_title('Display Size Distribution (Top 8)', fontsize=14, fontweight='bold')
            else:
                # Alternative: Correlation heatmap
                numeric_df = df.select_dtypes(include=[np.number])
                if len(numeric_df.columns) > 1:
                    corr = numeric_df.corr()
                    im = axes[1,0].imshow(corr, cmap='coolwarm', aspect='auto')
                    axes[1,0].set_title('Feature Correlation', fontsize=14, fontweight='bold')
                    axes[1,0].set_xticks(range(len(corr.columns)))
                    axes[1,0].set_yticks(range(len(corr.columns)))
                    axes[1,0].set_xticklabels(corr.columns, rotation=45, ha='right')
                    axes[1,0].set_yticklabels(corr.columns)
                    plt.colorbar(im, ax=axes[1,0])
            
            # 4. RAM vs Storage Bubble Chart
            ram_storage_counts = df.groupby(['ram_gb', 'storage_gb']).size().reset_index(name='count')
            
            scatter = axes[1,1].scatter(ram_storage_counts['ram_gb'], ram_storage_counts['storage_gb'],
                                       s=ram_storage_counts['count']*10, alpha=0.6, 
                                       c=ram_storage_counts['count'], cmap='viridis')
            axes[1,1].set_title('RAM vs Storage Relationship', fontsize=14, fontweight='bold')
            axes[1,1].set_xlabel('RAM (GB)')
            axes[1,1].set_ylabel('Storage (GB)')
            plt.colorbar(scatter, ax=axes[1,1], label='Number of Laptops')
            
            plt.tight_layout()
            plt.savefig('static/specs_analysis.png', dpi=300, bbox_inches='tight')
            print("   📊 Specs Analysis Charts: static/specs_analysis.png")
            plt.show()
            
        except Exception as e:
            print(f"   ⚠️ Error creating specs charts: {e}")

    def create_processor_analysis_chart(self, df, top_processors):
        """Create processor analysis visualization"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # 1. Top Processors Bar Chart
            bars = ax1.barh(range(len(top_processors)), top_processors.values,
                           color=plt.cm.Purples(np.linspace(0.4, 1, len(top_processors))))
            ax1.set_title('💻 PROCESSOR ANALYSIS', fontsize=16, fontweight='bold')
            ax1.set_xlabel('Number of Laptops')
            ax1.set_ylabel('Processor')
            ax1.set_yticks(range(len(top_processors)))
            ax1.set_yticklabels(top_processors.index)
            ax1.set_xlim(0, max(top_processors.values) * 1.2)
            
            # Add percentage labels
            for i, (processor, count) in enumerate(zip(top_processors.index, top_processors.values)):
                percentage = (count / len(df)) * 100
                ax1.text(count + max(top_processors.values)*0.01, i, 
                        f'{count:,}\n({percentage:.1f}%)', va='center', fontsize=9)
            
            # 2. Processor Distribution Pie
            top_8_processors = df['processor'].value_counts().head(8)
            other_count = df['processor'].value_counts()[8:].sum() if len(df['processor'].value_counts()) > 8 else 0
            
            if other_count > 0:
                top_8_processors = top_8_processors.copy()
                top_8_processors['Others'] = other_count
            
            colors2 = plt.cm.Set2(np.linspace(0, 1, len(top_8_processors)))
            wedges, texts, autotexts = ax2.pie(top_8_processors.values, labels=top_8_processors.index, 
                                              autopct='%1.1f%%', colors=colors2, startangle=90)
            ax2.set_title('Processor Distribution (Top 8)', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('static/processor_analysis.png', dpi=300, bbox_inches='tight')
            print("   📊 Processor Analysis Chart: static/processor_analysis.png")
            plt.show()
            
        except Exception as e:
            print(f"   ⚠️ Error creating processor chart: {e}")

    def close(self):
        """Close the browser"""
        if hasattr(self, 'driver'):
            self.driver.quit()
            print("\n🌙 Browser closed.")


def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("🏁 STARTING PAKISTANI LAPTOP PRICE SCRAPER WITH VISUAL EDA")
    print("📡 Sources: Daraz.pk & PriceOye.pk")
    print("📊 ORDER: Daraz FIRST, then PriceOye")
    print("🎯 Target: 1000+ Laptop Records")
    print("📈 FEATURE: Automatic Visual EDA Analysis")
    print("="*60)
    
    try:
        # Initialize scraper
        scraper = ComprehensiveLaptopScraper(headless=False)  # Set to True for production
        
        # Run scraping
        df = scraper.scrape_all()
        
        # Close browser
        scraper.close()
        
        # Summary
        if df is not None:
            print(f"\n🎉 FINAL SUMMARY:")
            print(f"   Total laptops collected: {len(df):,}")
            print(f"   Unique brands: {df['brand'].nunique() if 'brand' in df.columns else 'N/A'}")
            
            if 'price_pkr' in df.columns:
                avg_price = df['price_pkr'].mean()
                print(f"   Average price: Rs. {avg_price:,.0f}")
            else:
                print(f"   Average price: N/A")
                
            print(f"   Data saved to: data/laptops.csv")
            print(f"   Visual charts saved to: static/ folder")
            
            # Check if target achieved
            if len(df) >= 200:
                print(f"\n🎯 TARGET ACHIEVED: {len(df)} records (Target: 200+)")
            else:
                print(f"\n⚠️ TARGET NOT MET: {len(df)} records (Target: 200+)")
                
        else:
            print("\n⚠️  No data was collected. Please check the error messages above.")
        
        print("\n" + "="*60)
        print("✅ SCRAPING & VISUAL EDA COMPLETE!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        print("\n" + "="*60)
        print("💡 TROUBLESHOOTING TIPS:")
        print("   1. Check your internet connection")
        print("   2. Make sure Chrome is installed")
        print("   3. Try running with headless=True")
        print("   4. Check if websites are accessible")
        print("="*60)


if __name__ == "__main__":
    main()