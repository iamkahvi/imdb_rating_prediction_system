# Analysis of Netflix Data

## Development

1. Download data from https://www.kaggle.com/netflix-inc/netflix-prize-data

2. Convert .txt files to .csv to `to_csv.sh` (`gawk` must be installed)

3. `jupyter notebook .`

## Goal

The qualifying.txt is the final output that we submit to get evaluated. 

The probe.txt is used to test what we have.

For example, if the qualifying dataset looked like:

> 111:  
3245,2005-12-19  
5666,2005-12-23  
6789,2005-03-14  
225:  
1234,2005-05-26  
3456,2005-11-07  

*the magic happens*

then a prediction file should look something like:  
> 111:  
3.0  
3.4  
4.0  
225:  
1.0  
2.0  

which predicts that customer 3245 would have rated movie 111 3.0 stars on the
19th of Decemeber, 2005, that customer 5666 would have rated it slightly higher
at 3.4 stars on the 23rd of Decemeber, 2005, etc.

## Dataset Description

### combined_data_{1,2,3,4}.txt

per movie
- customer id
- rating
- date (YYYY-MM-DD)

### qualifying.txt

per movie
- customer id
- date

> MovieID1:  
CustomerID11,Date11  
CustomerID12,Date12  
...  
MovieID2:  
CustomerID21,Date21  
CustomerID22,Date22  

### movie_title.csv

- movieId
- year
- title
