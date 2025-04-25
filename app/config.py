# Database configuration
import os

DB_CONFIG = {
  "host":   os.environ["PGHOST"],
  "port":   os.environ["PGPORT"],
  "user":   os.environ["PGUSER"],
  "password": os.environ["PGPASSWORD"],
  "dbname": os.environ["PGDATABASE"],
}