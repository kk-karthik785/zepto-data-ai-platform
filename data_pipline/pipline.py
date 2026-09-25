import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import re
from pathlib import Path
from urllib.parse import urljoin


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR = 105.50

CURRENT_DIR = Path(__file__).resolve().parent

DATABASE_FILE = CURRENT_DIR / "books.db"
CSV_FILE = CURRENT_DIR / "books_cleaned.csv"
SQL_FILE = CURRENT_DIR / "queries.sql"
QUERY_OUTPUT_FILE = CURRENT_DIR / "query_outputs.txt"


# ============================================================
# 1. GET WEB PAGE
# ============================================================

def get_soup(url):
    """Download a webpage and return BeautifulSoup object."""

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=20
    )

    response.raise_for_status()

    return BeautifulSoup(response.text, "html.parser")


# ============================================================
# 2. CONVERT RATING TEXT TO NUMBER
# ============================================================

def convert_rating(rating_text):

    rating_values = {
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5
    }

    return rating_values.get(rating_text)


# ============================================================
# 3. CONVERT PRICE TO FLOAT
# ============================================================

def convert_price(price_text):

    if price_text is None:
        return None

    try:
        value = re.sub(r"[^0-9.]", "", price_text)

        if value == "":
            return None

        return float(value)

    except (ValueError, TypeError):
        return None


# ============================================================
# 4. SCRAPE BOOKS
# ============================================================

def scrape_books():

    all_books = []

    print("=" * 60)
    print("STARTING WEB SCRAPING")
    print("=" * 60)

    # First 5 pages of All Products
    for page_number in range(1, 6):

        page_url = urljoin(
            BASE_URL,
            f"catalogue/page-{page_number}.html"
        )

        print(f"\nScraping page {page_number}...")
        print(page_url)

        try:
            soup = get_soup(page_url)

        except requests.RequestException as error:
            print(
                f"Could not download page {page_number}: {error}"
            )
            continue

        books = soup.select("article.product_pod")

        print(f"Books found on page: {len(books)}")

        for book in books:

            # -------------------------
            # TITLE
            # -------------------------

            title_tag = book.select_one("h3 a")

            if title_tag:
                title = title_tag.get("title", "").strip()

                detail_url = urljoin(
                    page_url,
                    title_tag.get("href", "")
                )
            else:
                title = ""
                detail_url = ""

            # -------------------------
            # PRICE
            # -------------------------

            price_tag = book.select_one(".price_color")

            if price_tag:
                price = price_tag.get_text(strip=True)
            else:
                price = None

            # -------------------------
            # RATING
            # -------------------------

            rating_tag = book.select_one(".star-rating")

            rating_text = None

            if rating_tag:

                classes = rating_tag.get("class", [])

                if len(classes) > 1:
                    rating_text = classes[1]

            # -------------------------
            # AVAILABILITY
            # -------------------------

            availability_tag = book.select_one(
                ".availability"
            )

            if availability_tag:
                availability = availability_tag.get_text(
                    " ",
                    strip=True
                )
            else:
                availability = None

            # -------------------------
            # CATEGORY
            # -------------------------

            category = "Unknown"

            if detail_url:

                try:
                    detail_soup = get_soup(detail_url)

                    breadcrumb_links = detail_soup.select(
                        "ul.breadcrumb li a"
                    )

                    if len(breadcrumb_links) >= 3:

                        category = breadcrumb_links[
                            2
                        ].get_text(strip=True)

                except requests.RequestException:
                    category = "Unknown"

            # -------------------------
            # STORE RECORD
            # -------------------------

            all_books.append(
                {
                    "title": title,
                    "price": price,
                    "star_rating": rating_text,
                    "availability": availability,
                    "category": category
                }
            )

    dataframe = pd.DataFrame(all_books)

    return dataframe


# ============================================================
# 5. CLEAN DATA
# ============================================================

def clean_data(df):

    print("\n" + "=" * 60)
    print("CLEANING DATA")
    print("=" * 60)

    df = df.copy()

    # -------------------------
    # PRICE
    # -------------------------

    df["price_gbp"] = df["price"].apply(
        convert_price
    )

    # -------------------------
    # RATING
    # -------------------------

    df["rating"] = df["star_rating"].apply(
        convert_rating
    )

    # -------------------------
    # AVAILABILITY
    # -------------------------

    df["in_stock"] = (
        df["availability"]
        .fillna("")
        .str.contains(
            "In stock",
            case=False,
            na=False
        )
    )

    # -------------------------
    # REMOVE INVALID TITLE
    # -------------------------

    df = df[
        df["title"].notna()
        & (df["title"].str.strip() != "")
    ]

    # -------------------------
    # REMOVE UNKNOWN CATEGORY
    # -------------------------

    df = df[
        df["category"].notna()
        & (df["category"] != "Unknown")
    ]

    # -------------------------
    # MEDIAN IMPUTATION
    # -------------------------

    if df["price_gbp"].isna().any():

        median_price = df["price_gbp"].median()

        df["price_gbp"] = df[
            "price_gbp"
        ].fillna(median_price)

        print(
            f"Missing price values replaced "
            f"using median: {median_price}"
        )

    if df["rating"].isna().any():

        median_rating = df["rating"].median()

        df["rating"] = df[
            "rating"
        ].fillna(median_rating)

        print(
            f"Missing rating values replaced "
            f"using median: {median_rating}"
        )

    # Convert rating to integer
    df["rating"] = df["rating"].round().astype(int)

    # -------------------------
    # GBP TO INR
    # -------------------------

    df["price_inr"] = (
        df["price_gbp"] * GBP_TO_INR
    ).round(2)

    # -------------------------
    # FINAL COLUMNS
    # -------------------------

    df = df[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category"
        ]
    ]

    return df.reset_index(drop=True)


# ============================================================
# 6. CREATE SQLITE DATABASE
# ============================================================

def create_database(df):

    print("\n" + "=" * 60)
    print("CREATING SQLITE DATABASE")
    print("=" * 60)

    # Delete old database
    if DATABASE_FILE.exists():
        DATABASE_FILE.unlink()

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    cursor = connection.cursor()

    # Enable foreign keys
    cursor.execute(
        "PRAGMA foreign_keys = ON"
    )

    # -------------------------
    # CATEGORIES TABLE
    # -------------------------

    cursor.execute("""
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT NOT NULL UNIQUE
        )
    """)

    # -------------------------
    # BOOKS TABLE
    # -------------------------

    cursor.execute("""
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL,
            in_stock INTEGER NOT NULL,
            category_id INTEGER NOT NULL,
            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        )
    """)

    # -------------------------
    # INSERT CATEGORIES
    # -------------------------

    categories = sorted(
        df["category"].unique()
    )

    for category in categories:

        cursor.execute(
            """
            INSERT INTO categories(category_name)
            VALUES (?)
            """,
            (category,)
        )

    # -------------------------
    # INSERT BOOKS
    # -------------------------

    for _, row in df.iterrows():

        cursor.execute(
            """
            SELECT category_id
            FROM categories
            WHERE category_name = ?
            """,
            (row["category"],)
        )

        result = cursor.fetchone()

        if result is None:
            continue

        category_id = result[0]

        cursor.execute(
            """
            INSERT INTO books
            (
                title,
                price_gbp,
                price_inr,
                rating,
                in_stock,
                category_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                row["title"],
                float(row["price_gbp"]),
                float(row["price_inr"]),
                int(row["rating"]),
                int(row["in_stock"]),
                category_id
            )
        )

    connection.commit()

    print("Database created successfully.")

    return connection


# ============================================================
# 7. RUN SQL QUERIES AND SAVE OUTPUTS
# ============================================================

def run_sql_queries(connection):

    print("\n" + "=" * 60)
    print("SQL QUERY RESULTS")
    print("=" * 60)

    queries = {

        "Q1_SELECT_WHERE_ORDER_LIMIT": """
            SELECT title, price_gbp, rating
            FROM books
            WHERE rating >= 4
            ORDER BY price_gbp DESC
            LIMIT 10;
        """,

        "Q2_ORDER_BY_LIMIT": """
            SELECT title, price_inr
            FROM books
            ORDER BY price_inr DESC
            LIMIT 10;
        """,

        "Q3_DISTINCT": """
            SELECT DISTINCT rating
            FROM books
            ORDER BY rating;
        """,

        "Q4_BETWEEN": """
            SELECT title, price_gbp
            FROM books
            WHERE price_gbp BETWEEN 20 AND 40
            ORDER BY price_gbp;
        """,

        "Q5_JOIN": """
            SELECT
                b.title,
                b.price_gbp,
                b.rating,
                c.category_name
            FROM books AS b
            JOIN categories AS c
                ON b.category_id = c.category_id
            ORDER BY b.rating DESC
            LIMIT 20;
        """
    }

    results = {}

    # Save query outputs
    with open(
        QUERY_OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as output_file:

        for query_name, query in queries.items():

            print("\n" + "-" * 60)
            print(query_name)
            print("-" * 60)

            result = pd.read_sql(
                query,
                connection
            )

            print(
                result.to_string(index=False)
            )

            # Save query name
            output_file.write(
                "\n" + "=" * 60 + "\n"
            )

            output_file.write(
                query_name + "\n"
            )

            output_file.write(
                "=" * 60 + "\n"
            )

            # Save SQL query
            output_file.write(
                "SQL Query:\n"
            )

            output_file.write(
                query.strip() + "\n\n"
            )

            # Save output
            output_file.write(
                "Output:\n"
            )

            output_file.write(
                result.to_string(index=False)
            )

            output_file.write("\n\n")

            results[query_name] = result

    print(
        f"\nSQL query outputs saved to: "
        f"{QUERY_OUTPUT_FILE}"
    )

    return results


# ============================================================
# 8. SQL JOIN VS PANDAS MERGE
# ============================================================

def verify_join(connection):

    print("\n" + "=" * 60)
    print("SQL JOIN VS PANDAS MERGE")
    print("=" * 60)

    # SQL JOIN result
    sql_result = pd.read_sql(
        """
        SELECT
            b.book_id,
            b.title,
            b.price_gbp,
            b.rating,
            c.category_name
        FROM books AS b
        JOIN categories AS c
            ON b.category_id = c.category_id
        ORDER BY b.book_id;
        """,
        connection
    )

    # Read individual tables
    books_df = pd.read_sql(
        """
        SELECT
            book_id,
            title,
            price_gbp,
            rating,
            category_id
        FROM books;
        """,
        connection
    )

    categories_df = pd.read_sql(
        """
        SELECT
            category_id,
            category_name
        FROM categories;
        """,
        connection
    )

    # Pandas merge
    pandas_result = pd.merge(
        books_df,
        categories_df,
        on="category_id",
        how="inner"
    )

    pandas_result = pandas_result[
        [
            "book_id",
            "title",
            "price_gbp",
            "rating",
            "category_name"
        ]
    ]

    pandas_result = pandas_result.sort_values(
        "book_id"
    ).reset_index(drop=True)

    sql_result = sql_result.reset_index(
        drop=True
    )

    # Compare
    equivalent = sql_result.equals(
        pandas_result
    )

    print("\nSQL JOIN:")
    print(
        sql_result.head(10).to_string(
            index=False
        )
    )

    print("\nPandas MERGE:")
    print(
        pandas_result.head(10).to_string(
            index=False
        )
    )

    print(
        "\nAre both results equivalent?",
        equivalent
    )

    return equivalent


# ============================================================
# 9. SAVE SQL QUERIES
# ============================================================

def save_sql_queries():

    sql_content = """
-- Q1: SELECT + WHERE + ORDER BY + LIMIT

SELECT title, price_gbp, rating
FROM books
WHERE rating >= 4
ORDER BY price_gbp DESC
LIMIT 10;


-- Q2: ORDER BY + LIMIT

SELECT title, price_inr
FROM books
ORDER BY price_inr DESC
LIMIT 10;


-- Q3: DISTINCT

SELECT DISTINCT rating
FROM books
ORDER BY rating;


-- Q4: BETWEEN

SELECT title, price_gbp
FROM books
WHERE price_gbp BETWEEN 20 AND 40
ORDER BY price_gbp;


-- Q5: JOIN

SELECT
    b.title,
    b.price_gbp,
    b.rating,
    c.category_name
FROM books AS b
JOIN categories AS c
    ON b.category_id = c.category_id
ORDER BY b.rating DESC
LIMIT 20;
"""

    with open(
        SQL_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(sql_content)

    print(
        f"\nSQL queries saved to: {SQL_FILE}"
    )


# ============================================================
# 10. MAIN PROGRAM
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("BOOKS TO SCRAPE - DATA PIPELINE")
    print("=" * 60)

    # --------------------------------------------------------
    # STEP 1: SCRAPE
    # --------------------------------------------------------

    raw_df = scrape_books()

    print(
        f"\nTotal books scraped: {len(raw_df)}"
    )

    if len(raw_df) < 60:

        raise ValueError(
            "ERROR: Less than 60 books were scraped."
        )

    category_count = raw_df[
        "category"
    ].nunique()

    print(
        f"Number of categories: {category_count}"
    )

    if category_count < 3:

        raise ValueError(
            "ERROR: Less than 3 categories found."
        )

    # --------------------------------------------------------
    # STEP 2: CLEAN
    # --------------------------------------------------------

    clean_df = clean_data(raw_df)

    # --------------------------------------------------------
    # STEP 3: VALIDATION
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL DATASET")
    print("=" * 60)

    print(
        clean_df.head(10).to_string(
            index=False
        )
    )

    print("\nData types:")
    print(clean_df.dtypes)

    print(
        "\nNumber of rows:",
        len(clean_df)
    )

    print(
        "Number of categories:",
        clean_df["category"].nunique()
    )

    # --------------------------------------------------------
    # STEP 4: SAVE CSV
    # --------------------------------------------------------

    clean_df.to_csv(
        CSV_FILE,
        index=False
    )

    print(
        f"\nCSV saved: {CSV_FILE}"
    )

    # --------------------------------------------------------
    # STEP 5: CREATE DATABASE
    # --------------------------------------------------------

    connection = create_database(
        clean_df
    )

    # --------------------------------------------------------
    # STEP 6: SQL QUERIES
    # --------------------------------------------------------

    run_sql_queries(
        connection
    )

    # --------------------------------------------------------
    # STEP 7: PANDAS MERGE
    # --------------------------------------------------------

    equivalent = verify_join(
        connection
    )

    # --------------------------------------------------------
    # STEP 8: SAVE SQL
    # --------------------------------------------------------

    save_sql_queries()

    # --------------------------------------------------------
    # CLOSE DATABASE
    # --------------------------------------------------------

    connection.close()

    # --------------------------------------------------------
    # FINAL MESSAGE
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    if equivalent:

        print(
            "SQL JOIN and Pandas MERGE are equivalent."
        )

    else:

        print(
            "WARNING: SQL JOIN and Pandas MERGE differ."
        )


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":
    main()