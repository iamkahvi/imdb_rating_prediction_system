- Now that we have our tables we can do the data transformation

.headers on
.separator ,
.output movies_with_top_actor.csv

-- Select the actors and their ranks
WITH top_actors AS (SELECT
  princ.imdb_title_id,
  princ.imdb_name_id,
  name,
  ROW_NUMBER() OVER (PARTITION BY princ.imdb_title_id ORDER BY CAST(ordering AS INT) ASC) AS rank
FROM
  princ
INNER JOIN
  names ON princ.imdb_name_id = names.imdb_name_id
WHERE category = "actor" OR category = "actress")

-- Select only the top actor/actress
SELECT
  movies.imdb_title_id,
  movies.title,
  movies.original_title,
  movies.year,
  movies.date_published,
  movies.genre,
  movies.duration,
  movies.country,
  movies.language,
  movies.director,
  movies.writer,
  movies.production_company,
  movies.actors,
  movies.description,
  movies.avg_vote,
  movies.votes,
  movies.budget,
  movies.usa_gross_income,
  movies.worlwide_gross_income,
  movies.metascore,
  movies.reviews_from_users,
  movies.reviews_from_critics,
  top_actors.name AS top_actor
FROM
  movies
INNER JOIN
  top_actors ON top_actors.imdb_title_id = movies.imdb_title_id
WHERE
  top_actors.rank = 1;
