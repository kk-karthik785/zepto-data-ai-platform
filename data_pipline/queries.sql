
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
