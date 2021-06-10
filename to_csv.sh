#!/bin/bash

gawk '
    BEGIN { print "movieId,customerId,rating,date" }
    { n=match($0, /^(.*):/, arr); 
    if (n==1)
        movie=arr[1]
    else
        print movie","$0 }' $1 > $1.csv