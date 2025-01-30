WITH sleep_fixed_a AS (
  SELECT
    s.user_id_fk AS user_id,
    (s.start_date::TIMESTAMPTZ AT TIME ZONE 'UTC')::TIMESTAMPTZ AS started_at,
    (s.stop_date::TIMESTAMPTZ AT TIME ZONE 'UTC')::TIMESTAMPTZ AS ended_at
  FROM pa_sleep s
  WHERE
    s.start_date NOTNULL AND
    s.stop_date NOTNULL
),
sleep_fixed_b AS (
  SELECT
    s.*,
    p.cohort,
    s.started_at AT TIME ZONE p.timezone AS loc_started_at,
    s.ended_at AT TIME ZONE p.timezone AS loc_ended_at
  FROM sleep_fixed_a s
  JOIN faydesc_participants p ON p.id = s.user_id
  WHERE
    s.ended_at BETWEEN
      GREATEST(p.mfr_started_at, p.glucose_started_at) AND
      LEAST(p.mfr_ended_at, p.glucose_ended_at)
)
SELECT
  s.user_id,
  s.started_at,
  s.ended_at,
  s.cohort,
  s.loc_ended_at::DATE AS loc_occurred_on,
  EXTRACT(HOUR FROM s.loc_started_at) + EXTRACT(MINUTE FROM s.loc_started_at) / 60.0  AS loc_start_hour,
  EXTRACT(HOUR FROM s.loc_ended_at) + EXTRACT(MINUTE FROM s.loc_ended_at) / 60.0  AS loc_end_hour,
  EXTRACT(EPOCH FROM ended_at - started_at) / 3600.0 AS duration_hours
FROM sleep_fixed_b s
