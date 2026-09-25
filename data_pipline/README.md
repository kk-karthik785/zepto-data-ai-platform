# Module 1 - Data Collection and Pipeline

## Objective

The objective of this module is to collect book data from the public website Books to Scrape, clean and transform the data, store it in a normalized SQLite database, and demonstrate SQL and Pandas data analysis.

## Data Source

Website:

https://books.toscrape.com/

The first five pages of the All Products section were scraped using Python.

The scraping was performed using:

- Requests
- BeautifulSoup
- Pandas

## Data Collected

The following fields were collected:

- Book title
- Price in GBP
- Star rating
- Availability
- Category

The final cleaned dataset also contains:

- `price_gbp`
- `price_inr`
- `rating`
- `in_stock`
- `category`

## Data Cleaning

### Price

The original price contains the GBP currency symbol.

The currency symbol and other non-numeric characters were removed and the value was converted to a floating-point number.

Example:

`£51.77` → `51.77`

### Rating

The website provides ratings as text such as:

- One
- Two
- Three
- Four
- Five

These values were converted to integers from 1 to 5.

### Availability

Availability text was converted into a Boolean value:

- `True` = In stock
- `False` = Not in stock

### Missing Numeric Values

If numeric parsing produced missing values, median imputation was used.

Median imputation was selected because it is less affected by extreme values than the mean.

### Currency Conversion

A fixed conversion rate was used:

`1 GBP = 105.50 INR`

Therefore:

`price_inr = price_gbp × 105.50`

No external currency API was used.

## SQLite Database

The cleaned data is stored in:

`books.db`

The database contains two normalized tables.

### categories

| Column | Description |
|---|---|
| category_id | Primary key |
| category_name | Category name |

### books

| Column | Description |
|---|---|
| book_id | Primary key |
| title | Book title |
| price_gbp | Price in GBP |
| price_inr | Price in INR |
| rating | Rating from 1 to 5 |
| in_stock | Availability |
| category_id | Foreign key |

The `category_id` column in the `books` table references the `categories` table.

## SQL Queries

Five SQL queries were implemented in `queries.sql`.

They demonstrate:

1. SELECT and WHERE
2. ORDER BY
3. LIMIT
4. DISTINCT
5. BETWEEN
6. JOIN

The SQL queries are executed using Python and Pandas.

## Pandas SQL Analysis

`pd.read_sql()` was used to read SQL query results into Pandas DataFrames.

The SQL JOIN result was independently reproduced using:

`pd.merge()`

The two results were compared to verify their equivalence.

## Files

```text
data_pipeline/
├── pipeline.py
├── books.db
├── books_cleaned.csv
├── queries.sql
└── README.md