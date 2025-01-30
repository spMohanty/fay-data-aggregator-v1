WITH morning_surveys_fixed AS (
  SELECT
    s.user_id,
    s.survey_date AS loc_occurred_on,
    s.sleeping_start AS loc_start_time,
    s.sleeping_end AS loc_end_time,
    EXTRACT(HOUR FROM s.sleeping_start) + EXTRACT(MINUTE FROM s.sleeping_start) / 60.0 AS loc_start_hour,
    EXTRACT(HOUR FROM s.sleeping_end) + EXTRACT(MINUTE FROM s.sleeping_end) / 60.0 AS loc_end_hour
  FROM morning_surveys s
  WHERE
    s.sleeping_start NOTNULL AND
    s.sleeping_end NOTNULL AND
    s.survey_date NOTNULL
)
SELECT
  s.user_id,
  s.loc_occurred_on,
  s.loc_start_hour,
  s.loc_end_hour,
  (s.loc_occurred_on - MAKE_INTERVAL(days := (s.loc_start_time >= s.loc_end_time)::INT) + loc_start_time) AS raw_started_at,
  (s.loc_occurred_on + loc_end_time) AS raw_ended_at
FROM morning_surveys_fixed s
