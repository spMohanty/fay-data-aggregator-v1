SELECT EXISTS (
  SELECT 1
  FROM pg_tables t
  WHERE
    t.tablename = %s AND
    t.schemaname = 'public'
)
