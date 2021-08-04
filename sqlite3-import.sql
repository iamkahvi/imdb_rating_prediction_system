-- Process for inserting the top actor from each movie into the movies table.
-- sqlite3

-- Import the tables
.mode csv
.import ./content/title_principals.csv princ
.schema princ

-- sqlite3 has trouble importing the other two files for some reason. Use the csv-clean.py script to
-- convert those to use the ; separator instead of ,. Then run these lines
.separator ;
.import ./content/names-sep-new.csv names
.schema names
.import ./content/movies-sep-new.csv movies
.schema movies
