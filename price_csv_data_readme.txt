The dataset contains 20,703 entries and 93 columns. Here's a brief overview of the structure and some of the columns included in the data:

Unnamed: 0: An index or identifier column.
1. car_code: A unique code for each car.
2. car_last_price_in_brl: The last recorded price of the car in Brazilian Real (BRL).
3. car_manufacturer: The manufacturer of the car.
4. car_model: The model of the car.
5. car_model_year: The model year of the car.
6. first_year_of_tracking: The first year the car's price was tracked.
7. last_factory_price: The last known factory price of the car.
8. last_factory_price_year: The year of the last known factory price.
9. Columns year_01 to year_20: Prices of the car for 20 years, possibly representing annual data or specific years of tracking.

Other columns related to adjustments, year-over-year changes, etc.
Given the wide range of data available, the analysis could cover various aspects,
such as trends in car prices over time, comparisons between manufacturers, or the depreciation of car values.

Year-over-year (YoY) growth is a common financial performance comparison metric that measures the change in a variable over a 12-month period.
It's often used to compare the performance of financial metrics, such as revenue, profit, or, in this case, car prices, from one year to the next.
The formula to calculate the Year-over-Year (YoY) growth rate is typically:

YoY Growth= ((Value in Current Year − Value in Previous Year) / Value in Previous Year) * 100


This formula gives the percentage change from one year to the next.
In the context of the dataset, if we consider "year_01_yoy" as an example,
it would represent the percentage change in price from the "first_year_of_tracking" to the first year for which there's data.
For subsequent years, like "year_02_yoy", it would calculate the change from "year_01" to "year_02", and so on.

However, based on the dataset overview, it seems that the YoY columns
directly follow the pattern of naming according to the years (e.g., year_01_yoy, year_02_yoy),
indicating that these columns contain pre-calculated YoY growth values rather than raw yearly prices.
These values likely represent the growth rate of the car prices from one year to the next, calculated using the formula mentioned above.
