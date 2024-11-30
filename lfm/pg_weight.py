#!/usr/bin/env python3


# IMPORTANT: Run after lastfm_*_extraction*.py


# TODO: Write query to run this


"""
CALL apoc.periodic.iterate(
  'MATCH ()-[r]->() RETURN r',
  'WITH r
   SET r.pg_weight = CASE
       WHEN r.weight IS NOT NULL THEN r.weight
       WHEN r.count IS NOT NULL THEN r.count
       ELSE 1
     END',
  {concurrency: 12, batchSize: 100000}
);
"""
